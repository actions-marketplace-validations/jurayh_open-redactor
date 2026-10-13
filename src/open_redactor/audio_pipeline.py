"""Audio pipeline. One audio file in, share-safe audio file out.

An audio file has no pixels to detect. Its identifying channels are
the words spoken, the voice itself, and the tags in the container.
This pipeline covers all three with the same layers the video path
uses: spoken PII spans are found through the speech layer and muted,
the whole track can be silenced or pitch shifted, and container
metadata is inspected, stripped, and reported through the metadata
layer. Durations stand in for frames in the shared summary shape, one
unit per second, so audits and gates read audio summaries exactly
like video ones.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SUPPORTED_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a"}

# Re-encode targets per format, used whenever a filter runs. With no
# filter the stream is copied and only the metadata mapping changes.
ENCODE_ARGS = {
    ".mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
    ".wav": ["-c:a", "pcm_s16le"],
    ".m4a": ["-c:a", "aac", "-b:a", "192k"],
}


def probe_audio(path: Path) -> Tuple[float, bool]:
    """Return (duration in seconds, has an audio stream) from ffprobe."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration:stream=codec_type",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(proc.stdout or "{}")
    except Exception:
        return 0.0, False
    has_audio = any(
        isinstance(s, dict) and s.get("codec_type") == "audio"
        for s in data.get("streams", [])
    )
    try:
        duration = float(data.get("format", {}).get("duration", 0.0))
    except Exception:
        duration = 0.0
    return duration, has_audio


def _render_audio(
    input_path: Path,
    output_path: Path,
    filters: List[str],
) -> None:
    """Render the output with ffmpeg. Failure raises and writes nothing.

    The output goes to a temp sibling first and is moved into place
    only on success, so a failed transform never leaves a file that
    still carries the original audio.
    """
    suffix = input_path.suffix.lower()
    codec = ENCODE_ARGS[suffix] if filters else ["-c:a", "copy"]
    tmp = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(input_path)]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-map_metadata", "-1", *codec, str(tmp)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call(cmd)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Audio render failed for {input_path}: {exc}") from exc
    tmp.replace(output_path)


def run_audio_pipeline(
    input_path: Path,
    output_path: Path,
    audio_mode: str = "keep",
    pitch_factor: float = 0.8,
    speech_pii: bool = False,
    speech_soft: bool = False,
    transcript_path: Optional[Path] = None,
    whisper_model: str = "base",
    keep_metadata: bool = False,
    shadow: bool = False,
) -> Dict[str, object]:
    """Run the audio redaction pipeline and return a summary dict.

    Speech PII muting, whole track audio modes, and metadata handling
    mirror the video pipeline. Shadow mode writes the audit and summary
    and renders nothing.
    """
    if input_path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        raise ValueError(f"Unsupported audio format '{input_path.suffix}'. Supported audio inputs: {supported}.")
    if audio_mode not in ("keep", "mute", "pitch"):
        raise ValueError(f"Unknown audio mode '{audio_mode}'. Choose keep, mute, or pitch.")

    duration, has_audio = probe_audio(input_path)
    if not has_audio:
        raise ValueError(f"Input has no readable audio stream: {input_path}")
    total_seconds = max(1, int(round(duration)))

    print(f"Input: {input_path} {duration:.1f} seconds of audio")
    print(f"Audio mode: {audio_mode} pitch_factor={pitch_factor}")

    # Spoken PII layer, same contract as the video path. Explicit
    # requests fail loudly when no transcriber is available, while the
    # --sensitive bundle enables the layer only when it can run.
    speech_hits: List = []
    speech_requested = speech_pii or transcript_path is not None
    if speech_requested or speech_soft:
        from .speech import find_speech_pii, load_transcript, transcribe, transcriber_available

        words = None
        if transcript_path is not None:
            words = load_transcript(transcript_path)
            print(f"Speech: using transcript {transcript_path}")
        elif transcriber_available():
            print("Speech: transcribing audio locally for spoken PII")
            words = transcribe(input_path, model=whisper_model)
        elif speech_requested:
            raise ValueError(
                "Speech redaction needs a transcriber. Install it with "
                'pip install "open-redactor[speech]", or pass --transcript with a Whisper JSON file.'
            )
        else:
            print("Speech layer skipped: no transcriber installed, so spoken numbers were not scanned")
        if words is not None:
            speech_hits = find_speech_pii(words)
            for hit in speech_hits:
                print(f"Speech: {hit.kind} at {hit.start:.1f}s to {hit.end:.1f}s will be muted")
            if not speech_hits:
                print("Speech: no verified PII found in the audio")

    # Metadata layer, same contract as the video path: findings are
    # reported in the summary, stripped by default, kept only on
    # request, and listed as elements in audits and keep runs so the
    # gate can fail on a tagged original.
    from .metadata import inspect_metadata, labels as metadata_labels

    meta_findings = inspect_metadata(input_path)
    found_labels = metadata_labels(meta_findings)
    metadata_info = {"found": found_labels, "stripped": bool(meta_findings) and not keep_metadata and not shadow}
    metadata_elements: List[dict] = []
    if meta_findings:
        if shadow:
            print(f"Metadata: {', '.join(found_labels)} present in the source")
        elif keep_metadata:
            print(f"Metadata: {', '.join(found_labels)} found in the source and kept in the output by request")
        else:
            print(f"Metadata: {', '.join(found_labels)} found in the source, stripped from the output")
        if shadow or keep_metadata:
            from .analytics import SEVERITY_WEIGHTS

            for n, finding in enumerate(meta_findings, start=1):
                level = finding["severity"]
                metadata_elements.append(
                    {
                        "track": f"metadata:{finding['label'].lower()}:{n}",
                        "kind": finding["label"].lower(),
                        "severity": level,
                        "severity_weight": SEVERITY_WEIGHTS[level],
                        "frames_detected": total_seconds,
                        "first_frame": 0,
                        "last_frame": max(0, total_seconds - 1),
                        "share_of_clip": 1.0,
                    }
                )

    from .analytics import SEVERITY_WEIGHTS, build_summary, severity_for, summary_text, write_summary_files

    speech_elements: List[dict] = []
    for n, hit in enumerate(speech_hits, start=1):
        level = severity_for(hit.kind)
        first = int(hit.start)
        last = int(hit.end)
        speech_elements.append(
            {
                "track": f"speech:{hit.kind}:{n}",
                "kind": hit.kind,
                "severity": level,
                "severity_weight": SEVERITY_WEIGHTS[level],
                "frames_detected": max(1, last - first),
                "first_frame": first,
                "last_frame": last,
                "share_of_clip": round(max(1, last - first) / total_seconds, 3),
            }
        )
    summary = build_summary(
        {},
        total_seconds,
        audio_mode=audio_mode,
        shadow=shadow,
        kept_visible=[],
        sam_mask_source="none",
        extra_elements=speech_elements + metadata_elements,
        metadata_info=metadata_info if meta_findings else None,
    )

    if shadow:
        paths = write_summary_files(summary, output_path, shadow=True)
        print(summary_text(summary))
        print(f"Audit report: {paths['audit']}")
        print(f"Summary JSON: {paths['json']}")
        print("Shadow mode: no redacted audio was rendered and the original is untouched.")
        return {
            "input": str(input_path),
            "output": None,
            "audit": str(paths["audit"]),
            "summary": str(paths["json"]),
            "frames": total_seconds,
            "objects": 0,
            "shadow": True,
            "preview": False,
        }

    filters: List[str] = []
    if audio_mode == "pitch":
        filters.append(f"rubberband=pitch={pitch_factor}")
        print(f"Audio: pitch shifted by factor {pitch_factor}, tempo preserved")
    elif audio_mode == "mute":
        filters.append("volume=0")
        print("Audio: muted, output track is silent")
    if speech_hits:
        from .speech import mute_span_filter

        filters.append(mute_span_filter(speech_hits))
        print(f"Audio: {len(speech_hits)} spoken PII span(s) muted")

    _render_audio(input_path, output_path, filters)

    if meta_findings:
        if keep_metadata:
            from .metadata import transplant_metadata

            if not transplant_metadata(input_path, output_path):
                print("Metadata: could not copy the source tags onto the output")
        else:
            survivors = [
                l for l in metadata_labels(inspect_metadata(output_path))
                if l in found_labels
            ]
            if survivors:
                print(f"Metadata warning: {', '.join(survivors)} still present in the output")

    print(f"Wrote: {output_path}")
    paths = write_summary_files(summary, output_path, shadow=False)
    print(summary_text(summary))
    print(f"Summary JSON: {paths['json']}")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "frames": total_seconds,
        "objects": 0,
        "summary": str(paths["json"]),
        "preview": False,
    }
