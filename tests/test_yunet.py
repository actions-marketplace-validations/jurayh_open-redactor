"""YuNet fast face provider: gating, model integrity, and dispatch."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from open_redactor import yunet
from open_redactor.cli import main as cli_main
from open_redactor.yunet import YunetFaceClient, box_mask


def test_non_face_phrases_are_skipped_without_a_model(tmp_path: Path, monkeypatch):
    # The gate must fire before any model download is attempted
    def boom():
        raise AssertionError("model fetch must not happen for non-face phrases")

    monkeypatch.setattr(yunet, "ensure_model", boom)
    client = YunetFaceClient()
    photo = tmp_path / "x.jpg"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(photo)
    result = client.segment_image(photo, "license plate", shape=(48, 64))
    assert result.objects == []
    assert result.tracks == {}


def test_model_integrity_check(tmp_path: Path, monkeypatch):
    payload = b"fake-model-bytes"
    target = tmp_path / "model.onnx"
    target.write_bytes(payload)
    monkeypatch.setattr(yunet, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(yunet, "model_cache_path", lambda: target)
    assert yunet.ensure_model() == target
    target.write_bytes(b"corrupted")
    monkeypatch.setattr(yunet.urllib.request, "urlretrieve", lambda url, dest: Path(dest).write_bytes(b"also-wrong"))
    with pytest.raises(RuntimeError, match="integrity check"):
        yunet.ensure_model()


def test_box_mask_region():
    mask = box_mask((10, 12), (2, 3, 6, 7))
    assert mask.shape == (10, 12)
    assert mask[3:7, 2:6].all()
    assert mask.sum() == 16
    assert mask[0, 0] == 0


class FakeYunetClient:
    """Stands in for the real detector: one face box in the middle."""

    def segment_image(self, image_path, phrase, shape):
        from open_redactor.sam_client import DetectedObject, SegmentationResult

        if phrase.strip().lower() not in ("face", "faces"):
            return SegmentationResult(phrase=phrase)
        h, w = shape
        box = (w // 3, h // 3, 2 * w // 3, 2 * h // 3)
        result = SegmentationResult(phrase=phrase, mask_source="box")
        result.objects.append(DetectedObject(track_id="0", frame_index=0, box=box, phrase=phrase))
        result.tracks["0"] = {0: box_mask(shape, box)}
        return result

    def segment_video(self, video_path, phrase, shape):
        return self.segment_image(video_path, phrase, shape)


def noise_photo(path: Path) -> np.ndarray:
    """Save a noise photo and return its decoded pixels.

    Pure noise is the worst case for JPEG, so the only honest baseline
    is what a reader decodes from the file, not the pre-save array.
    """
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return np.asarray(Image.open(path)).copy()


def test_cli_dispatches_yunet_for_photos(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(yunet, "YunetFaceClient", FakeYunetClient)
    photo = tmp_path / "crowd.jpg"
    before = noise_photo(photo)
    out = tmp_path / "out.jpg"
    code = cli_main([str(photo), "--provider", "yunet", "--target", "face", "--output", str(out)])
    assert code == 0
    after = np.asarray(Image.open(out)).astype(float)
    # The fake face box covers the center: blur must smooth the noise there
    assert after[32:64, 32:64].std() < before[32:64, 32:64].astype(float).std() * 0.6
    # Outside the box the photo is essentially untouched
    assert np.abs(after[0:8, 0:8] - before[0:8, 0:8].astype(float)).mean() < 12.0


def test_cli_yunet_ignores_other_targets(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(yunet, "YunetFaceClient", FakeYunetClient)
    photo = tmp_path / "plate.jpg"
    before = noise_photo(photo)
    out = tmp_path / "out.jpg"
    code = cli_main([str(photo), "--provider", "yunet", "--target", "license plate", "--output", str(out)])
    assert code == 0
    after = np.asarray(Image.open(out)).astype(float)
    assert np.abs(after - before.astype(float)).mean() < 12.0
