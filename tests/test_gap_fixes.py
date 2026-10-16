"""Gap fixes: photo shadow audits and mixed-media batch mode."""

import shutil
import struct
import wave
from pathlib import Path

import segno
from PIL import Image

from open_redactor import audit_media
from open_redactor.cli import main as cli_main

QR_VIDEO = Path(__file__).parent.parent / "eval" / "data" / "qr.mp4"


def make_qr_photo(path: Path) -> Path:
    import io

    qr = segno.make("https://example.com/secret")
    buf = io.BytesIO()
    qr.save(buf, kind="png", scale=8, border=2)
    buf.seek(0)
    qr_img = Image.open(buf).convert("RGB").resize((200, 200))
    canvas = Image.new("RGB", (320, 320), "white")
    canvas.paste(qr_img, (60, 60))
    canvas.save(path)
    return path


def make_tagged_photo(path: Path) -> Path:
    import piexif

    exif = piexif.load(str(path))
    exif["GPS"] = {
        piexif.GPSIFD.GPSLatitudeRef: "N",
        piexif.GPSIFD.GPSLatitude: ((37, 1), (49, 1), (0, 1)),
        piexif.GPSIFD.GPSLongitudeRef: "W",
        piexif.GPSIFD.GPSLongitude: ((122, 1), (28, 1), (0, 1)),
    }
    piexif.insert(piexif.dump(exif), str(path))
    return path


def make_tone_wav(path: Path) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"".join(struct.pack("<h", 2000) for _ in range(8000)))
    return path


def test_api_photo_audit_reports_code_and_metadata(tmp_path: Path):
    photo = make_tagged_photo(make_qr_photo(tmp_path / "scene.jpg"))
    result = audit_media(photo, options={"sensitive": True})
    assert result.output_path is None
    assert result.audit_path and Path(result.audit_path).exists()
    assert result.summary is not None
    tracks = [e["track"] for e in result.summary["elements"]]
    assert any(t.startswith("code:") for t in tracks)
    assert any(t.startswith("metadata:gps location:") for t in tracks)
    assert result.summary["metadata"]["found"] == ["GPS location"]
    assert result.summary["metadata"]["stripped"] is False
    audit_text = Path(result.audit_path).read_text()
    assert "reported only, nothing was rendered" in audit_text
    assert not (tmp_path / "scene.redacted.jpg").exists()


def test_cli_photo_shadow_writes_audit_only(tmp_path: Path):
    photo = make_qr_photo(tmp_path / "scene.jpg")
    code = cli_main([str(photo), "--shadow", "--sensitive"])
    assert code == 0
    assert (tmp_path / "scene.audit.audit.txt").exists()
    assert (tmp_path / "scene.audit.summary.json").exists()
    assert not (tmp_path / "scene.redacted.jpg").exists()


def test_batch_processes_video_photo_and_audio(tmp_path: Path):
    shutil.copy(QR_VIDEO, tmp_path / "clip.mp4")
    make_qr_photo(tmp_path / "scene.jpg")
    make_tone_wav(tmp_path / "note.wav")
    code = cli_main([str(tmp_path), "--batch", "--sensitive"])
    assert code == 0
    assert (tmp_path / "clip.redacted.mp4").exists()
    assert (tmp_path / "scene.redacted.jpg").exists()
    assert (tmp_path / "note.redacted.wav").exists()
    assert (tmp_path / "batch-summary.json").exists()


def test_batch_shadow_audits_photo_and_audio(tmp_path: Path):
    make_qr_photo(tmp_path / "scene.jpg")
    make_tone_wav(tmp_path / "note.wav")
    code = cli_main([str(tmp_path), "--batch", "--shadow", "--sensitive"])
    assert code == 0
    assert (tmp_path / "scene.audit.audit.txt").exists()
    assert (tmp_path / "note.audit.audit.txt").exists()
    assert not (tmp_path / "scene.redacted.jpg").exists()
    assert not (tmp_path / "note.redacted.wav").exists()
