"""Location audit and GPS coordinate text detection.

The estimator itself is never called in tests: pipeline tests stub
``run_location_audit`` at the module seam the pipelines import from,
and module tests exercise parsing, block building, and the failure
path with no key configured.
"""

import base64
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import open_redactor.image_pipeline as image_pipeline
import open_redactor.location_audit as location_audit
import open_redactor.pipeline as video_pipeline
from open_redactor.analytics import build_summary, kind_from_track_key, severity_for, summary_text
from open_redactor.api import RedactionOptions
from open_redactor.location_audit import (
    LocationEstimate,
    estimate_location,
    parse_estimate,
    run_location_audit,
    summary_block,
    text_lines,
)
from open_redactor.pii import find_pii_in_text
from open_redactor.sam_client import LocalSamStub

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


def stub_sam_api(monkeypatch):
    """Route the pipelines' SAM API client at the local stub.

    Location audit runs are API runs now (the local guard rejects the
    combination everywhere), so integration tests fake the SAM client
    instead of passing local=True the way they used to.
    """

    class FakeSamApiClient:
        @staticmethod
        def from_env(*args, **kwargs):
            return LocalSamStub()

    monkeypatch.setattr(image_pipeline, "SamApiClient", FakeSamApiClient)
    monkeypatch.setattr(video_pipeline, "SamApiClient", FakeSamApiClient)


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

    stub_sam_api(monkeypatch)
    monkeypatch.setattr(location_audit, "run_location_audit", fake_audit)
    src = tmp_path / "photo.jpg"
    make_photo(src)
    out = tmp_path / "photo.redacted.jpg"
    result = run_image_pipeline(
        input_path=src, output_path=out, targets=["face"], provider="sam",
        location_audit=True,
    )
    assert result["location_audit"]["status"] == "ok"
    assert result["location_audit"]["basis"] == "processed photo"
    summary = json.loads(Path(result["summary"]).read_text())
    assert summary["location_audit"]["estimated_city"] == "Paris"


def test_photo_shadow_audit_text_includes_location(tmp_path: Path, monkeypatch):
    from open_redactor.image_pipeline import run_image_pipeline

    stub_sam_api(monkeypatch)
    monkeypatch.setattr(location_audit, "run_location_audit", fake_audit)
    src = tmp_path / "photo.jpg"
    make_photo(src)
    out = tmp_path / "photo.audit.jpg"
    result = run_image_pipeline(
        input_path=src, output_path=out, targets=["face"], provider="sam",
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

    stub_sam_api(monkeypatch)
    monkeypatch.setattr(location_audit, "run_location_audit", fake_audit)
    src = tmp_path / "clip.mp4"
    make_video(src)
    out = tmp_path / "clip.redacted.mp4"
    result = run_pipeline(
        input_path=src, output_path=out, targets=["face"], provider="sam",
        location_audit=True,
    )
    assert result["location_audit"]["basis"] == "one sampled frame of the processed output"
    summary = json.loads(Path(result["summary"]).read_text())
    assert summary["location_audit"]["status"] == "ok"


def test_video_shadow_location_block(tmp_path: Path, monkeypatch):
    from open_redactor.pipeline import run_pipeline

    stub_sam_api(monkeypatch)
    monkeypatch.setattr(location_audit, "run_location_audit", fake_audit)
    src = tmp_path / "clip.mp4"
    make_video(src)
    out = tmp_path / "clip.audit.mp4"
    result = run_pipeline(
        input_path=src, output_path=out, targets=["face"], provider="sam",
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


# --- GPS overlap masking and formats (0.2.19) ---


def test_gps_neighbouring_decimal_cannot_mask_pair():
    assert kinds("999.1234 37.774929 -122.419418") == ["gps coordinates"]
    assert kinds("200.12345 37.774929 -122.419418") == ["gps coordinates"]


def test_gps_neighbouring_decimal_cannot_steal_latitude():
    hits = find_pii_in_text("1.2345 37.774929 -122.419418")
    assert [h.kind for h in hits] == ["gps coordinates"]
    # The whole run is one covered span: the longitude is inside it.
    assert "-122.419418" in hits[0].text
    assert hits[0].end == len("1.2345 37.774929 -122.419418")


def test_gps_label_and_separator_formats():
    assert kinds("LAT 37.774929 LON -122.419418") == ["gps coordinates"]
    assert kinds("LAT: 37.774929, LON: -122.419418") == ["gps coordinates"]
    assert kinds("37.774929\t-122.419418") == ["gps coordinates"]
    assert kinds("37.774929 / -122.419418") == ["gps coordinates"]
    assert kinds("37.774929; -122.419418") == ["gps coordinates"]
    assert kinds("37.774929° -122.419418°") == ["gps coordinates"]
    assert kinds("37.774929n, -122.419418w") == ["gps coordinates"]


def test_gps_decimal_comma_formats():
    assert kinds("52,520008 13,404954") == ["gps coordinates"]
    assert kinds("52,520008, 13,404954") == ["gps coordinates"]
    assert kinds("37,774929 -122,419418") == ["gps coordinates"]


def test_gps_hit_span_has_no_surrounding_whitespace():
    hits = find_pii_in_text("at 37.774929 -122.419418 now")
    assert hits[0].text == "37.774929 -122.419418"


# --- Estimator verdict validation (0.2.19) ---


def _verdict_payload(**overrides):
    verdict = {
        "lat": 48.8566,
        "lon": 2.3522,
        "country": "France",
        "city": "Paris",
        "confidence_25km": 25,
        "cues": ["facades"],
    }
    verdict.update(overrides)
    return {"output_text": json.dumps(verdict)}


def test_parse_estimate_rejects_out_of_range_and_nan():
    with pytest.raises(RuntimeError):
        parse_estimate(_verdict_payload(lat=999.0, lon=-999.0))
    with pytest.raises(RuntimeError):
        parse_estimate({"output_text": '{"lat": NaN, "lon": 2.0, "country": "X", "city": "Y", "confidence_25km": 25, "cues": []}'})
    with pytest.raises(RuntimeError):
        parse_estimate(_verdict_payload(confidence_25km=150))


def test_parse_estimate_cues_string_becomes_one_cue():
    estimate = parse_estimate(_verdict_payload(cues="blue street name plaques"))
    assert estimate.cues == ["blue street name plaques"]


def test_parse_estimate_caps_cues():
    estimate = parse_estimate(_verdict_payload(cues=[f"cue {i}" for i in range(20)]))
    assert len(estimate.cues) == 12


def test_location_estimate_rejects_invalid_direct_construction():
    with pytest.raises(ValueError):
        LocationEstimate(latitude=91.0, longitude=0.0, country="X", city="Y", confidence_25km=0.5)


# --- Estimator payload hygiene (0.2.19) ---


class _FakeEstimatorResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return _verdict_payload()


class _FakeEstimatorClient:
    captured: dict = {}

    def __init__(self, timeout=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, endpoint, json=None, headers=None):
        _FakeEstimatorClient.captured = json
        return _FakeEstimatorResponse()


def _captured_payload_bytes() -> bytes:
    url = _FakeEstimatorClient.captured["input"][0]["content"][1]["image_url"]
    assert url.startswith("data:image/jpeg;base64,")
    return base64.b64decode(url.split(",", 1)[1])


def test_estimate_location_reencodes_webp_as_jpeg(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(location_audit, "httpx", SimpleNamespace(Client=_FakeEstimatorClient))
    src = tmp_path / "photo.webp"
    make_photo(src)
    estimate = estimate_location(src, "test-key", endpoint="http://127.0.0.1/mock")
    assert estimate.city == "Paris"
    raw = _captured_payload_bytes()
    assert raw[:2] == b"\xff\xd8"  # JPEG magic, whatever the source format


def test_estimate_location_never_sends_exif(monkeypatch, tmp_path: Path):
    import piexif
    from PIL import Image

    monkeypatch.setattr(location_audit, "httpx", SimpleNamespace(Client=_FakeEstimatorClient))
    src = tmp_path / "tagged.jpg"
    Image.new("RGB", (160, 120), (90, 120, 160)).save(src)
    exif = {
        "GPS": {
            piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
            piexif.GPSIFD.GPSLatitudeRef: "N",
            piexif.GPSIFD.GPSLatitude: ((37, 1), (20, 1), (30, 1)),
            piexif.GPSIFD.GPSLongitudeRef: "W",
            piexif.GPSIFD.GPSLongitude: ((122, 1), (5, 1), (40, 1)),
        }
    }
    piexif.insert(piexif.dump(exif), str(src))
    estimate_location(src, "test-key", endpoint="http://127.0.0.1/mock")
    raw = _captured_payload_bytes()
    assert raw != src.read_bytes()
    assert piexif.load(raw).get("GPS", {}) == {}


def test_run_location_audit_prints_endpoint(monkeypatch, capsys):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.setenv("OPEN_REDACTOR_ENDPOINT", "http://127.0.0.1:9/audit")
    monkeypatch.setattr(location_audit, "estimate_location", lambda *a, **k: CANNED)
    block = run_location_audit(image_path=Path("x.jpg"), api_key_env="MODEL_API_KEY")
    assert block["status"] == "ok"
    assert "Location audit endpoint: http://127.0.0.1:9/audit" in capsys.readouterr().out


# --- Local guard on every surface (0.2.19) ---


def test_pipelines_reject_local_location_audit(tmp_path: Path):
    from open_redactor.image_pipeline import run_image_pipeline
    from open_redactor.pipeline import run_pipeline

    src = tmp_path / "photo.jpg"
    make_photo(src)
    with pytest.raises(ValueError):
        run_image_pipeline(
            input_path=src, output_path=tmp_path / "out.jpg", targets=["face"],
            local=True, provider="sam", location_audit=True,
        )
    clip = tmp_path / "clip.mp4"
    make_video(clip)
    with pytest.raises(ValueError):
        run_pipeline(
            input_path=clip, output_path=tmp_path / "out.mp4", targets=["face"],
            local=True, provider="sam", location_audit=True,
        )


def test_cli_rejects_local_location_audit(tmp_path: Path):
    from open_redactor.cli import main as cli_main

    src = tmp_path / "photo.jpg"
    make_photo(src)
    assert cli_main([str(src), "--local", "--location-audit"]) == 2
    assert cli_main([str(src), "--provider", "yunet", "--location-audit"]) == 2
    clip = tmp_path / "clip.mp4"
    make_video(clip)
    assert cli_main([str(clip), "--backend", "local", "--location-audit"]) == 2
    assert cli_main([str(tmp_path), "--batch", "--local", "--location-audit"]) == 2


def test_photo_shadow_audit_passes_decoded_frame(tmp_path: Path, monkeypatch):
    from open_redactor.image_pipeline import run_image_pipeline

    stub_sam_api(monkeypatch)
    recorded: dict = {}

    def recording_audit(**kwargs):
        recorded.update(kwargs)
        return summary_block(CANNED, kwargs.get("basis", "test"))

    monkeypatch.setattr(location_audit, "run_location_audit", recording_audit)
    src = tmp_path / "photo.jpg"
    make_photo(src)
    run_image_pipeline(
        input_path=src, output_path=tmp_path / "photo.audit.jpg", targets=["face"],
        provider="sam", shadow=True, location_audit=True,
    )
    assert isinstance(recorded.get("frame"), np.ndarray)
    assert recorded.get("image_path") is None


# --- Video audit samples the rendered output (0.2.19, the Critical) ---


def test_video_render_audit_frame_comes_from_output(tmp_path: Path, monkeypatch):
    from open_redactor.pipeline import run_pipeline
    from open_redactor.sam_client import DetectedObject, SegmentationResult

    class MaskSamClient:
        @staticmethod
        def from_env(*args, **kwargs):
            return MaskSamClient()

        def segment_video(self, video_path, phrase, shape):
            mask = np.zeros(shape, dtype=bool)
            mask[:, : shape[1] // 2] = True
            obj = DetectedObject(
                track_id="0", frame_index=0,
                box=(0, 0, shape[1] // 2, shape[0]), mask=mask, phrase=phrase,
            )
            return SegmentationResult(
                phrase=phrase, objects=[obj],
                tracks={"0": {i: mask for i in range(10)}}, mask_source="pixel",
            )

    monkeypatch.setattr(video_pipeline, "SamApiClient", MaskSamClient)
    recorded: dict = {}

    def recording_audit(**kwargs):
        recorded.update(kwargs)
        return summary_block(CANNED, kwargs.get("basis", "test"))

    monkeypatch.setattr(location_audit, "run_location_audit", recording_audit)

    # Left half is a 1px checkerboard (extreme contrast), right half flat.
    src = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    yy, xx = np.mgrid[0:48, 0:64]
    frame = np.where(xx < 32, ((xx + yy) % 2) * 255, 128).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for _ in range(10):
        writer.write(frame)
    writer.release()

    out = tmp_path / "clip.redacted.mp4"
    result = run_pipeline(
        input_path=src, output_path=out, targets=["face"], provider="sam",
        location_audit=True, mode="blur", strength=31,
    )
    assert result["location_audit"]["basis"] == "one sampled frame of the processed output"
    sampled = recorded.get("frame")
    assert isinstance(sampled, np.ndarray)
    # The masked checkerboard must arrive blurred, not as the source's
    # black/white extremes: the audit frame is the processed output.
    region = sampled[:, :32].astype(float)
    assert region.std() < 40


def test_video_preview_result_includes_location_key(tmp_path: Path):
    from open_redactor.pipeline import run_pipeline

    src = tmp_path / "clip.mp4"
    make_video(src)
    result = run_pipeline(
        input_path=src, output_path=tmp_path / "clip.redacted.mp4", targets=["face"],
        local=True, provider="sam", preview=True,
    )
    assert result["location_audit"] is None


def test_batch_rollup_carries_location_blocks(tmp_path: Path):
    from open_redactor.cli import write_batch_rollup

    block = summary_block(CANNED, "processed photo")
    path = tmp_path / "a.summary.json"
    path.write_text(json.dumps({
        "elements_found": 0,
        "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        "frames_with_sensitive_elements": 0,
        "kept_visible": [],
        "exposure_score": 0.0,
        "location_audit": block,
    }))
    dest = write_batch_rollup([path], tmp_path)
    rollup = json.loads(dest.read_text())
    assert rollup["files"][0]["location_audit"]["estimated_city"] == "Paris"
