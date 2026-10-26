"""Location audit and GPS coordinate text detection.

The estimator itself is never called in tests: pipeline tests stub
``run_location_audit`` at the module seam the pipelines import from,
and module tests exercise parsing, block building, and the failure
path with no key configured.
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

import open_redactor.location_audit as location_audit
from open_redactor.analytics import build_summary, kind_from_track_key, severity_for, summary_text
from open_redactor.api import RedactionOptions
from open_redactor.location_audit import (
    LocationEstimate,
    parse_estimate,
    run_location_audit,
    summary_block,
    text_lines,
)
from open_redactor.pii import find_pii_in_text

CANNED = LocationEstimate(
    latitude=48.8566,
    longitude=2.3522,
    country="France",
    city="Paris",
    confidence_25km=0.25,
    cues=["Haussmann facades", "blue street name plaques"],
)


def fake_audit(**kwargs):
    return summary_block(CANNED, kwargs.get("basis", "test"))


# --- GPS coordinate text detection ---


def kinds(text: str):
    return [h.kind for h in find_pii_in_text(text)]


def test_gps_stamp_from_dashcam_overlay():
    assert kinds("2026-08-26 -30.920099 146.96307E speed 42") == ["gps coordinates"]


def test_gps_plain_pair_and_hemisphere_forms():
    assert kinds("taken at 37.774929, -122.419418 yesterday") == ["gps coordinates"]
    assert kinds("N 51.50722 W 0.12750") == ["gps coordinates"]
    assert kinds("146.963070 -30.920099") == ["gps coordinates"]  # lon, lat order


def test_gps_rejects_lookalikes():
    assert "gps coordinates" not in kinds("total 12.99 and 45.00 due")
    assert "gps coordinates" not in kinds("GPS 0.0000 0.0000 no fix")
    assert "gps coordinates" not in kinds("value 91.12345 and 200.12345")
    assert "gps coordinates" not in kinds("version 1.2.3456 build 7.8.9012")


def test_gps_does_not_steal_other_pii():
    assert kinds("card 4111 1111 1111 1111 on file") == ["credit card number"]
    assert kinds("call 415-555-0132 now") == ["phone number"]
    assert kinds("write to a.b@example.com") == ["email"]


def test_gps_severity_and_kind():
    assert severity_for("gps coordinates") == "high"
    assert kind_from_track_key("pii:gps coordinates:1") == "gps coordinates"


# --- Estimator parsing and reporting ---


def test_parse_estimate_from_output_items():
    payload = {
        "output": [
            {
                "content": [
                    {
                        "text": json.dumps(
                            {
                                "lat": 48.8566,
                                "lon": 2.3522,
                                "country": "France",
                                "city": "Paris",
                                "confidence_25km": 25,
                                "cues": ["Haussmann facades"],
                            }
                        )
                    }
                ]
            }
        ]
    }
    estimate = parse_estimate(payload)
    assert estimate.country == "France"
    assert estimate.confidence_25km == 0.25
    assert estimate.cues == ["Haussmann facades"]


def test_parse_estimate_rejects_garbage():
    with pytest.raises(RuntimeError):
        parse_estimate({"output": [{"content": [{"text": "no json here"}]}]})


def test_summary_block_and_text_lines_ok():
    block = summary_block(CANNED, "processed photo")
    assert block["status"] == "ok"
    assert block["estimated_city"] == "Paris"
    lines = "\n".join(text_lines(block))
    assert "Location audit" in lines
    assert "Paris, France" in lines
    assert "within 25 km: 25%" in lines
    assert "does not remove it" in lines


def test_summary_block_failed_is_explicit():
    block = summary_block(None, "processed photo", error="no API key found")
    assert block["status"] == "failed"
    lines = "\n".join(text_lines(block))
    assert "estimate failed" in lines
    assert "do not treat this output as location-checked" in lines


def test_run_location_audit_without_key_fails_soft(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    block = run_location_audit(image_path=Path("whatever.jpg"), api_key_env="MODEL_API_KEY")
    assert block["status"] == "failed"
    assert "MODEL_API_KEY" in block["error"]


def test_analytics_summary_carries_location_block():
    block = summary_block(CANNED, "processed photo")
    summary = build_summary({}, 1, location_info=block)
    assert summary["location_audit"]["estimated_country"] == "France"
    assert "Location audit" in summary_text(summary)


# --- Option validation ---


def test_location_audit_option_validation():
    RedactionOptions(location_audit=True).validate()
    with pytest.raises(ValueError):
        RedactionOptions(location_audit=True, local=True).validate()
    with pytest.raises(ValueError):
        RedactionOptions(location_audit=True, backend="local").validate()
    with pytest.raises(ValueError):
        RedactionOptions(location_audit=True, provider="grounding-sam").validate()


# --- Pipeline integration (estimator stubbed) ---


def make_photo(path: Path):
    img = np.full((120, 160, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_photo_render_writes_location_summary(tmp_path: Path, monkeypatch):
    from open_redactor.image_pipeline import run_image_pipeline

    monkeypatch.setattr(location_audit, "run_location_audit", fake_audit)
    src = tmp_path / "photo.jpg"
    make_photo(src)
    out = tmp_path / "photo.redacted.jpg"
    result = run_image_pipeline(
        input_path=src, output_path=out, targets=["face"], local=True, provider="sam",
        location_audit=True,
    )
    assert result["location_audit"]["status"] == "ok"
    assert result["location_audit"]["basis"] == "processed photo"
    summary = json.loads(Path(result["summary"]).read_text())
    assert summary["location_audit"]["estimated_city"] == "Paris"


def test_photo_shadow_audit_text_includes_location(tmp_path: Path, monkeypatch):
    from open_redactor.image_pipeline import run_image_pipeline

    monkeypatch.setattr(location_audit, "run_location_audit", fake_audit)
    src = tmp_path / "photo.jpg"
    make_photo(src)
    out = tmp_path / "photo.audit.jpg"
    result = run_image_pipeline(
        input_path=src, output_path=out, targets=["face"], local=True, provider="sam",
        shadow=True, location_audit=True,
    )
    assert result["location_audit"]["basis"] == "source photo"
    audit_text = Path(result["audit"]).read_text()
    assert "Location audit" in audit_text
    assert "does not remove it" in audit_text


def make_video(path: Path):
    subprocess.check_call([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=gray:s=64x48:r=10:d=1",
        "-pix_fmt", "yuv420p", str(path),
    ])


def test_video_render_location_block(tmp_path: Path, monkeypatch):
    from open_redactor.pipeline import run_pipeline

    monkeypatch.setattr(location_audit, "run_location_audit", fake_audit)
    src = tmp_path / "clip.mp4"
    make_video(src)
    out = tmp_path / "clip.redacted.mp4"
    result = run_pipeline(
        input_path=src, output_path=out, targets=["face"], local=True, provider="sam",
        location_audit=True,
    )
    assert result["location_audit"]["basis"] == "one sampled frame of the processed output"
    summary = json.loads(Path(result["summary"]).read_text())
    assert summary["location_audit"]["status"] == "ok"


def test_video_shadow_location_block(tmp_path: Path, monkeypatch):
    from open_redactor.pipeline import run_pipeline

    monkeypatch.setattr(location_audit, "run_location_audit", fake_audit)
    src = tmp_path / "clip.mp4"
    make_video(src)
    out = tmp_path / "clip.audit.mp4"
    result = run_pipeline(
        input_path=src, output_path=out, targets=["face"], local=True, provider="sam",
        shadow=True, location_audit=True,
    )
    assert result["location_audit"]["basis"] == "one sampled frame of the source"
    audit_text = Path(result["audit"]).read_text()
    assert "Location audit" in audit_text


# --- CLI ---


def test_cli_location_audit_rejects_audio(tmp_path: Path):
    import wave

    from open_redactor.cli import main as cli_main

    wav_path = tmp_path / "note.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 800)
    assert cli_main([str(wav_path), "--location-audit"]) == 2
