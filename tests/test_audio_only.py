"""Audio only inputs: speech muting, audio modes, and metadata handling."""

import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from open_redactor.api import audit_media, redact_media
from open_redactor.audio_pipeline import run_audio_pipeline
from open_redactor.cli import main as cli_main
from open_redactor.metadata import inspect_metadata, labels

RATE = 16000


def write_tone_wav(path: Path, seconds: float = 7.0, freq: float = 440.0) -> Path:
    t = np.arange(int(RATE * seconds)) / RATE
    samples = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.tobytes())
    return path


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(float) / 32768.0


def rms_between(path: Path, start: float, end: float) -> float:
    samples = read_wav(path)
    seg = samples[int(start * RATE):int(end * RATE)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


def write_transcript(path: Path) -> Path:
    words = [
        {"word": "call", "start": 0.5, "end": 0.8},
        {"word": "me", "start": 0.9, "end": 1.1},
    ]
    digits = ["four", "one", "five", "five", "five", "five", "two", "six", "seven", "one"]
    for i, word in enumerate(digits):
        words.append({"word": word, "start": 2.0 + 0.3 * i, "end": 2.3 + 0.3 * i})
    path.write_text(json.dumps({"segments": [{"words": words}]}))
    return path


def make_tagged_mp3(tmp_path: Path) -> Path:
    wav = write_tone_wav(tmp_path / "tone.wav", seconds=3.0)
    mp3 = tmp_path / "tagged.mp3"
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(wav),
            "-metadata", "title=Secret Title",
            "-metadata", "artist=Test Person",
            "-metadata", "date=2024-01-02",
            str(mp3),
        ]
    )
    return mp3


def load_summary_for(output: Path) -> dict:
    return json.loads(output.with_suffix(".summary.json").read_text())


def test_speech_muting_silences_span_and_reports_element(tmp_path: Path):
    wav = write_tone_wav(tmp_path / "note.wav")
    transcript = write_transcript(tmp_path / "transcript.json")
    out = tmp_path / "note.redacted.wav"
    run_audio_pipeline(wav, out, speech_pii=True, transcript_path=transcript)
    assert out.exists()
    inside = rms_between(out, 2.5, 4.5)
    outside = rms_between(out, 6.0, 6.9)
    assert outside > 0.2
    assert inside < outside * 0.01
    summary = load_summary_for(out)
    tracks = [e["track"] for e in summary["elements"]]
    assert "speech:phone number:1" in tracks


def test_metadata_found_and_stripped_by_default(tmp_path: Path):
    mp3 = make_tagged_mp3(tmp_path)
    assert "title and description" in labels(inspect_metadata(mp3))
    out = tmp_path / "tagged.redacted.mp3"
    run_audio_pipeline(mp3, out)
    summary = load_summary_for(out)
    assert "title and description" in summary["metadata"]["found"]
    assert "creation time" in summary["metadata"]["found"]
    assert summary["metadata"]["stripped"] is True
    assert "title and description" not in labels(inspect_metadata(out))
    assert summary["elements"] == []


def test_keep_metadata_preserves_tags_and_reports_elements(tmp_path: Path):
    mp3 = make_tagged_mp3(tmp_path)
    out = tmp_path / "tagged.redacted.mp3"
    run_audio_pipeline(mp3, out, keep_metadata=True)
    summary = load_summary_for(out)
    assert summary["metadata"]["stripped"] is False
    assert "title and description" in labels(inspect_metadata(out))
    tracks = [e["track"] for e in summary["elements"]]
    assert any(t.startswith("metadata:title and description:") for t in tracks)


def test_shadow_audit_reports_elements_and_renders_nothing(tmp_path: Path):
    mp3 = make_tagged_mp3(tmp_path)
    transcript = write_transcript(tmp_path / "transcript.json")
    base = tmp_path / "tagged.audit.mp3"
    details = run_audio_pipeline(mp3, base, transcript_path=transcript, shadow=True)
    assert details["output"] is None
    assert not base.exists()
    summary = json.loads(Path(details["summary"]).read_text())
    tracks = [e["track"] for e in summary["elements"]]
    assert "speech:phone number:1" in tracks
    assert any(t.startswith("metadata:") for t in tracks)
    assert Path(details["audit"]).exists()


def test_full_mute_pitch_and_mode_errors(tmp_path: Path):
    wav = write_tone_wav(tmp_path / "tone.wav", seconds=3.0)
    muted = tmp_path / "tone.muted.wav"
    run_audio_pipeline(wav, muted, audio_mode="mute")
    assert rms_between(muted, 0.5, 2.5) < 0.001

    pitched = tmp_path / "tone.pitched.wav"
    run_audio_pipeline(wav, pitched, audio_mode="pitch", pitch_factor=0.8)
    assert rms_between(pitched, 0.5, 2.5) > 0.1

    with pytest.raises(ValueError):
        run_audio_pipeline(wav, tmp_path / "x.wav", audio_mode="loud")
    with pytest.raises(ValueError):
        run_audio_pipeline(tmp_path / "tone.flac", tmp_path / "x.flac")


def test_unreadable_audio_is_rejected_without_output(tmp_path: Path):
    fake = tmp_path / "fake.mp3"
    fake.write_bytes(b"this is not audio at all")
    out = tmp_path / "fake.redacted.mp3"
    with pytest.raises(ValueError):
        run_audio_pipeline(fake, out)
    assert not out.exists()


def test_api_redact_and_audit_audio(tmp_path: Path):
    wav = write_tone_wav(tmp_path / "note.wav")
    transcript = write_transcript(tmp_path / "transcript.json")
    result = redact_media(wav, options={"speech_pii": True, "transcript_path": str(transcript)})
    assert result.output_path is not None
    assert result.output_path.endswith("note.redacted.wav")
    assert Path(result.output_path).exists()
    tracks = [e["track"] for e in result.summary["elements"]]
    assert "speech:phone number:1" in tracks

    mp3 = make_tagged_mp3(tmp_path)
    audit = audit_media(mp3, options={"transcript_path": str(transcript)})
    assert audit.output_path is None
    assert audit.audit_path and Path(audit.audit_path).exists()
    tracks = [e["track"] for e in audit.summary["elements"]]
    assert "speech:phone number:1" in tracks
    assert any(t.startswith("metadata:") for t in tracks)


def test_cli_audio_dispatch_and_shadow(tmp_path: Path):
    wav = write_tone_wav(tmp_path / "note.wav")
    transcript = write_transcript(tmp_path / "transcript.json")
    out = tmp_path / "out.wav"
    code = cli_main([str(wav), "--speech-pii", "--transcript", str(transcript), "--output", str(out)])
    assert code == 0
    assert out.exists()
    assert rms_between(out, 2.5, 4.5) < rms_between(out, 6.0, 6.9) * 0.01

    mp3 = make_tagged_mp3(tmp_path)
    code = cli_main([str(mp3), "--shadow"])
    assert code == 0
    assert not (tmp_path / "tagged.redacted.mp3").exists()
    summary = json.loads((tmp_path / "tagged.redacted.summary.json").read_text())
    assert any(e["track"].startswith("metadata:") for e in summary["elements"])
