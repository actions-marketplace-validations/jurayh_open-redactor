"""Metadata inspection and handling for media files.

Pixels are only half the file. Photos carry EXIF blocks with GPS
coordinates, exact timestamps, and device details. Videos carry
container tags with creation time and location. A redacted copy whose
metadata still names the place and the phone is not share safe, so
every run inspects the source, guarantees the output is clean, and
reports what was found.

Outputs are rebuilt from decoded frames today, which drops source
metadata as a side effect. This module makes that a checked promise
instead of an accident: renders pin the guarantee with an explicit
metadata mapping, keep mode transplants source tags deliberately and
says so, and audits surface tagged originals as findings.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}

# label, severity for each class of finding
GPS = ("GPS location", "high")
CREATION = ("creation time", "medium")
DEVICE = ("device info", "medium")
TEXT = ("title and description", "medium")
EMBEDDED_TEXT = ("embedded text", "medium")


def _finding(label_sev) -> Dict[str, str]:
    return {"label": label_sev[0], "severity": label_sev[1]}


def _dedupe(findings: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: Dict[str, Dict[str, str]] = {}
    for f in findings:
        seen.setdefault(f["label"], f)
    order = [GPS[0], CREATION[0], DEVICE[0], TEXT[0], EMBEDDED_TEXT[0]]
    return [seen[k] for k in order if k in seen]


def inspect_photo(path: Path) -> List[Dict[str, str]]:
    """Classify the sensitive EXIF and text metadata in a photo."""
    findings: List[Dict[str, str]] = []
    try:
        from PIL import Image

        with Image.open(path) as img:
            exif = img.getexif()
            if exif:
                try:
                    gps = exif.get_ifd(0x8825)
                except Exception:
                    gps = {}
                if gps:
                    findings.append(_finding(GPS))
                if any(tag in exif for tag in (306, 36867, 36868)):
                    findings.append(_finding(CREATION))
                if any(tag in exif for tag in (271, 272)):
                    findings.append(_finding(DEVICE))
            text = getattr(img, "text", None)
            if isinstance(text, dict) and text:
                findings.append(_finding(EMBEDDED_TEXT))
    except Exception:
        return []
    return _dedupe(findings)


def inspect_video(path: Path) -> List[Dict[str, str]]:
    """Classify the sensitive container tags in a video or audio file."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(proc.stdout or "{}")
    except Exception:
        return []
    tags: Dict[str, str] = {}
    fmt = data.get("format", {}).get("tags", {})
    if isinstance(fmt, dict):
        tags.update({str(k).lower(): str(v) for k, v in fmt.items()})
    for stream in data.get("streams", []):
        st = stream.get("tags", {})
        if isinstance(st, dict):
            tags.update({str(k).lower(): str(v) for k, v in st.items()})
    findings: List[Dict[str, str]] = []
    for key in tags:
        if key.startswith("location"):
            findings.append(_finding(GPS))
        elif key in ("creation_time", "date", "recording_time"):
            findings.append(_finding(CREATION))
        elif key in ("make", "model", "manufacturer", "com.apple.quicktime.make", "com.apple.quicktime.model"):
            findings.append(_finding(DEVICE))
        elif key in ("title", "description", "comment", "synopsis", "artist", "album", "composer"):
            findings.append(_finding(TEXT))
    return _dedupe(findings)


def inspect_metadata(path: Path) -> List[Dict[str, str]]:
    """All sensitive metadata findings for any supported input."""
    if Path(path).suffix.lower() in IMAGE_SUFFIXES:
        return inspect_photo(path)
    return inspect_video(path)


def transplant_metadata(source: Path, output: Path) -> bool:
    """Copy container metadata from source onto a video output."""
    tmp = output.with_name(output.stem + ".meta-tmp" + output.suffix)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(output), "-i", str(source),
        "-map", "0", "-map_metadata", "1", "-c", "copy",
        str(tmp),
    ]
    try:
        subprocess.check_call(cmd)
        tmp.replace(output)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False


def labels(findings: List[Dict[str, str]]) -> List[str]:
    return [f["label"] for f in findings]
