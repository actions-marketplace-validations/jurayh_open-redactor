"""Metadata layer: source tags are found, stripped, reported, or kept on request."""

import json
import subprocess
from pathlib import Path

import piexif
from PIL import Image

from open_redactor.cli import main as cli_main
from open_redactor.image_pipeline import run_image_pipeline
from open_redactor.metadata import inspect_metadata, labels


def tagged_photo(path: Path) -> None:
    img = Image.new("RGB", (320, 240), (90, 120, 160))
    img.save(path)
    exif = {
        "0th": {
            piexif.ImageIFD.Make: "TestPhone",
            piexif.ImageIFD.Model: "TestPhone 12",
            piexif.ImageIFD.DateTime: "2026:09:15 10:30:00",
        },
        "GPS": {
            piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
            piexif.GPSIFD.GPSLatitudeRef: "N",
            piexif.GPSIFD.GPSLatitude: ((37, 1), (20, 1), (30, 1)),
            piexif.GPSIFD.GPSLongitudeRef: "W",
            piexif.GPSIFD.GPSLongitude: ((122, 1), (5, 1), (40, 1)),
        },
    }
    piexif.insert(piexif.dump(exif), str(path))


def tagged_video(path: Path) -> None:
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=gray:s=64x48:r=10:d=1",
        "-metadata", "creation_time=2026-09-15T10:30:00Z",
        "-metadata", "location=+37.3382-122.0503/",
        "-metadata", "title=Family day",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ])


def video_tags(path: Path) -> str:
    return subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True,
    ).stdout


def test_inspect_photo_finds_gps_device_and_time(tmp_path: Path):
    photo = tmp_path / "tagged.jpg"
    tagged_photo(photo)
    found = labels(inspect_metadata(photo))
    assert "GPS location" in found
    assert "device info" in found
    assert "creation time" in found
    severities = {f["label"]: f["severity"] for f in inspect_metadata(photo)}
    assert severities["GPS location"] == "high"


def test_inspect_video_finds_location_time_and_title(tmp_path: Path):
    video = tmp_path / "tagged.mp4"
    tagged_video(video)
    found = labels(inspect_metadata(video))
    assert "GPS location" in found
    assert "creation time" in found
    assert "title and description" in found


def test_inspect_clean_file_has_no_findings(tmp_path: Path):
    photo = tmp_path / "clean.jpg"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(photo)
    assert inspect_metadata(photo) == []


def test_photo_run_strips_and_reports(tmp_path: Path):
    photo = tmp_path / "tagged.jpg"
    tagged_photo(photo)
    out = tmp_path / "out.jpg"
    result = run_image_pipeline(photo, out, targets=[], local=True)
    assert Image.open(out).getexif() == {}
    assert result["metadata"]["stripped"] is True
    assert "GPS location" in result["metadata"]["found"]


def test_photo_keep_metadata_reattaches_exif(tmp_path: Path):
    photo = tmp_path / "tagged.jpg"
    tagged_photo(photo)
    out = tmp_path / "out.jpg"
    result = run_image_pipeline(photo, out, targets=[], local=True, keep_metadata=True)
    exif = Image.open(out).getexif()
    assert exif.get(271) == "TestPhone"
    assert 34853 in exif
    # Orientation was consumed on read, so it must not come back
    assert 274 not in exif
    assert result["metadata"]["stripped"] is False


def test_video_run_strips_and_reports(tmp_path: Path):
    video = tmp_path / "tagged.mp4"
    tagged_video(video)
    code = cli_main([str(video), "--local", "--no-cache"])
    assert code == 0
    tags = video_tags(tmp_path / "tagged.redacted.mp4")
    assert "TAG:location=" not in tags
    assert "TAG:creation_time=" not in tags
    summary = json.loads((tmp_path / "tagged.redacted.summary.json").read_text())
    assert summary["metadata"]["stripped"] is True
    assert "GPS location" in summary["metadata"]["found"]
    # Stripped findings are handled, so they are not residual elements
    assert not any(e["track"].startswith("metadata:") for e in summary["elements"])


def test_video_keep_metadata_transplants_tags(tmp_path: Path):
    video = tmp_path / "tagged.mp4"
    tagged_video(video)
    code = cli_main([str(video), "--local", "--no-cache", "--keep-metadata"])
    assert code == 0
    tags = video_tags(tmp_path / "tagged.redacted.mp4")
    assert "TAG:location=" in tags
    assert "TAG:creation_time=" in tags
    summary = json.loads((tmp_path / "tagged.redacted.summary.json").read_text())
    assert summary["metadata"]["stripped"] is False
    meta_elements = [e for e in summary["elements"] if e["track"].startswith("metadata:")]
    assert any(e["kind"] == "gps location" and e["severity"] == "high" for e in meta_elements)


def test_shadow_audit_lists_metadata_findings(tmp_path: Path):
    video = tmp_path / "tagged.mp4"
    tagged_video(video)
    code = cli_main([str(video), "--local", "--no-cache", "--shadow"])
    assert code == 0
    summary = json.loads((tmp_path / "tagged.redacted.summary.json").read_text())
    assert summary["metadata"]["stripped"] is False
    meta_elements = [e for e in summary["elements"] if e["track"].startswith("metadata:")]
    assert any(e["kind"] == "gps location" for e in meta_elements)
