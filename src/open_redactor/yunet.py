"""Fast local face provider (YuNet).

The basic tier of the provider ladder. OpenCV ships a small DNN face
detector, YuNet, that runs in milliseconds on CPU with no torch stack
and no API key. It detects faces only, front and profile, and returns
boxes, so it suits the most common job (blur the faces in this clip)
at a fraction of the cost of a full promptable run.

Faces only is a hard limit, stated plainly: any other phrase returns
zero matches with a note. The model file comes from the OpenCV model
zoo (Apache-2.0), is fetched once into the local cache, and is
verified against a pinned SHA-256 before use, so a corrupted or
swapped download is refused rather than loaded.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .local_gsam import IouTracker
from .sam_client import DetectedObject, SegmentationResult

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{MODEL_NAME}"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
SCORE_THRESHOLD = 0.5


def model_cache_path() -> Path:
    return Path.home() / ".cache" / "open-redactor" / "models" / MODEL_NAME


def _hash_ok(path: Path) -> bool:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return False
    return digest == MODEL_SHA256


def ensure_model() -> Path:
    """Return the verified model path, downloading it once if needed."""
    path = model_cache_path()
    if path.exists() and _hash_ok(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".onnx.part")
    try:
        print(f"Downloading the YuNet face model once ({MODEL_NAME})")
        urllib.request.urlretrieve(MODEL_URL, tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "The YuNet face model is not cached and could not be downloaded. "
            "Run once with network access, or choose another provider. "
            f"Download failed: {exc}"
        ) from exc
    if not _hash_ok(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("The downloaded YuNet model failed its integrity check and was discarded.")
    tmp.replace(path)
    return path


def box_mask(shape: Tuple[int, int], box: Tuple[int, int, int, int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = box
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = 1
    return mask


class YunetFaceClient:
    """Face detection with OpenCV YuNet. Any phrase other than face
    returns zero matches with a printed note, never a guess."""

    def __init__(self, score_threshold: float = SCORE_THRESHOLD) -> None:
        self.score_threshold = score_threshold
        self._detector = None
        self._size: Optional[Tuple[int, int]] = None

    def _detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        import cv2

        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                str(ensure_model()), "", (320, 320),
                score_threshold=self.score_threshold, nms_threshold=0.3, top_k=5000,
            )
        height, width = frame.shape[:2]
        if self._size != (width, height):
            self._detector.setInputSize((width, height))
            self._size = (width, height)
        _, faces = self._detector.detect(frame)
        boxes: List[Tuple[int, int, int, int]] = []
        if faces is not None:
            for f in faces:
                x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                boxes.append((x, y, x + w, y + h))
        return boxes

    @staticmethod
    def _is_face_phrase(phrase: str) -> bool:
        return phrase.strip().lower() in ("face", "faces")

    def segment_image(self, image_path: Path, phrase: str, shape: tuple[int, int]) -> SegmentationResult:
        import cv2

        if not self._is_face_phrase(phrase):
            print(f"YuNet provider detects faces only. Skipping '{phrase}'.")
            return SegmentationResult(phrase=phrase)
        frame = cv2.imread(str(image_path))
        if frame is None:
            return SegmentationResult(phrase=phrase)
        result = SegmentationResult(phrase=phrase, mask_source="box")
        for track_id, box in enumerate(self._detect(frame)):
            key = str(track_id)
            result.objects.append(DetectedObject(track_id=key, frame_index=0, box=box, phrase=phrase))
            result.tracks[key] = {0: box_mask(shape, box)}
        return result

    def segment_video(self, video_path: Path, phrase: str, shape: tuple[int, int]) -> SegmentationResult:
        import cv2

        if not self._is_face_phrase(phrase):
            print(f"YuNet provider detects faces only. Skipping '{phrase}'.")
            return SegmentationResult(phrase=phrase)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return SegmentationResult(phrase=phrase)
        result = SegmentationResult(phrase=phrase, mask_source="box")
        tracker = IouTracker(min_iou=0.3)
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes = self._detect(frame)
            ids = tracker.assign([(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in boxes])
            for track_id, box in zip(ids, boxes):
                key = str(track_id)
                result.objects.append(DetectedObject(track_id=key, frame_index=frame_index, box=box, phrase=phrase))
                result.tracks.setdefault(key, {})[frame_index] = box_mask(shape, box)
            frame_index += 1
        cap.release()
        return result
