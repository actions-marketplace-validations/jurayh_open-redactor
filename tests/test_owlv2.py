"""OWLv2 local provider: box handling, tracking, and dispatch."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from open_redactor import owlv2
from open_redactor.api import RedactionOptions
from open_redactor.cli import main as cli_main
from open_redactor.owlv2 import Owlv2Client, clean_box
from open_redactor.sam_client import SegmentationResult


def test_clean_box_clamps_and_rejects_empty_regions():
    assert clean_box((-5, 3.2, 25.6, 40), (50, 30)) == (0, 3, 26, 40)
    assert clean_box((10, 10, 10, 20), (50, 30)) is None
    assert clean_box((40, 10, 20, 20), (50, 30)) is None


class FakeProcessor:
    def __call__(self, text, images, return_tensors):
        assert text == [["license plate"]]
        assert return_tensors == "pt"
        return {"fake": "inputs"}

    def post_process_grounded_object_detection(self, outputs, target_sizes, threshold, text_labels):
        assert threshold == 0.05
        assert target_sizes == [(48, 64)]
        return [{"boxes": [[8, 6, 40, 30], [50, 30, 90, 60], [20, 20, 20, 30]]}]


class FakeModel:
    def __call__(self, **inputs):
        assert inputs == {"fake": "inputs"}
        return {"fake": "outputs"}


def test_detect_returns_clean_boxes():
    client = Owlv2Client()
    client._processor = FakeProcessor()
    client._model = FakeModel()
    boxes = client._detect(object(), "license plate", shape=(48, 64))
    assert boxes == [(8, 6, 40, 30), (50, 30, 64, 48)]


def test_segment_image_returns_box_tracks(tmp_path: Path, monkeypatch):
    client = Owlv2Client()
    monkeypatch.setattr(
        client,
        "_detect",
        lambda image, phrase, shape: [(6, 5, 26, 20)],
    )
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (64, 48), (20, 30, 40)).save(photo)
    result = client.segment_image(photo, "person", shape=(48, 64))
    assert result.mask_source == "box"
    assert [obj.box for obj in result.objects] == [(6, 5, 26, 20)]
    assert result.tracks["0"][0].sum() == 20 * 15


def test_segment_video_links_boxes_into_one_track(tmp_path: Path, monkeypatch):
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    calls = {"count": 0}

    def fake_detect(image, phrase, shape):
        offset = calls["count"]
        calls["count"] += 1
        return [(10 + offset, 8, 34 + offset, 30)]

    client = Owlv2Client()
    monkeypatch.setattr(client, "_detect", fake_detect)
    result = client.segment_video(video, "person", shape=(48, 64))
    assert list(result.tracks) == ["0"]
    assert sorted(result.tracks["0"]) == [0, 1, 2]
    assert [obj.box for obj in result.objects] == [
        (10, 8, 34, 30),
        (11, 8, 35, 30),
        (12, 8, 36, 30),
    ]


def test_api_accepts_owlv2_provider():
    RedactionOptions(provider="owlv2").validate()


class FakeOwlv2Client:
    def segment_image(self, image_path, phrase, shape):
        from open_redactor.sam_client import DetectedObject
        from open_redactor.yunet import box_mask

        h, w = shape
        box = (w // 3, h // 3, 2 * w // 3, 2 * h // 3)
        result = SegmentationResult(phrase=phrase, mask_source="box")
        result.objects.append(DetectedObject(track_id="0", frame_index=0, box=box, phrase=phrase))
        result.tracks["0"] = {0: box_mask(shape, box)}
        return result

    def segment_video(self, video_path, phrase, shape):
        return self.segment_image(video_path, phrase, shape)


def test_cli_dispatches_owlv2_for_photos(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(owlv2, "Owlv2Client", FakeOwlv2Client)
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
    photo = tmp_path / "scene.jpg"
    Image.fromarray(arr).save(photo)
    before = np.asarray(Image.open(photo)).astype(float)
    out = tmp_path / "out.jpg"
    code = cli_main([str(photo), "--provider", "owlv2", "--target", "person", "--output", str(out)])
    assert code == 0
    after = np.asarray(Image.open(out)).astype(float)
    assert after[32:64, 32:64].std() < before[32:64, 32:64].std() * 0.6
