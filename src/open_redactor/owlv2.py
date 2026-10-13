"""Local OWLv2 open vocabulary provider.

OWLv2 is the second rung of the local provider ladder. It finds named
objects with a text query and returns boxes rather than segmentation
masks. The boxes are linked across video frames with the same IoU
tracker used by the other local providers, and every result records
its mask source as "box" so a fast detector run never presents itself
as a pixel mask SAM run.

The default checkpoint is Google's Apache-2.0 base ensemble model.
Install the shared local stack with: pip install "open-redactor[local]"
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .local_gsam import IouTracker
from .sam_client import DetectedObject, SegmentationResult
from .yunet import box_mask

MODEL_ID = "google/owlv2-base-patch16-ensemble"
SCORE_THRESHOLD = 0.05

Box = Tuple[int, int, int, int]


def clean_box(box: Sequence[float], shape: tuple[int, int]) -> Optional[Box]:
    """Clamp one detector box to the frame and reject empty regions."""
    height, width = shape
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except Exception:
        return None
    cleaned = (
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    )
    if cleaned[2] <= cleaned[0] or cleaned[3] <= cleaned[1]:
        return None
    return cleaned


class Owlv2Client:
    """Open vocabulary box detection with OWLv2.

    A missing local stack returns zero matches with a clear install
    message. A model load failure is reported once and also returns
    zero matches rather than fabricating detections.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        score_threshold: float = SCORE_THRESHOLD,
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.score_threshold = score_threshold
        self.device = device
        self._processor = None
        self._model = None
        self._load_attempted = False

    def _load(self) -> bool:
        if self._model is not None and self._processor is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except Exception as exc:
            print(
                "OWLv2 provider needs the optional local stack. "
                'Install with: pip install "open-redactor[local]" '
                f"Import failed: {exc}"
            )
            return False

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        try:
            print(f"Local OWLv2 on {device}: {self.model_id}")
            self._processor = Owlv2Processor.from_pretrained(self.model_id)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_id).to(device)
            if hasattr(self._model, "eval"):
                self._model.eval()
        except Exception as exc:
            self._processor = None
            self._model = None
            print(
                "OWLv2 provider could not load its model. "
                "Check the local install and network access for the first download. "
                f"Load failed: {exc}"
            )
            return False
        return True

    def _detect(self, image, phrase: str, shape: tuple[int, int]) -> List[Box]:
        if not self._load():
            return []
        try:
            import torch

            inference_context = torch.no_grad()
        except Exception:
            inference_context = nullcontext()

        inputs = self._processor(text=[[phrase]], images=image, return_tensors="pt")
        if hasattr(inputs, "to") and self.device is not None:
            inputs = inputs.to(self.device)
        with inference_context:
            outputs = self._model(**inputs)
        detections = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=[shape],
            threshold=self.score_threshold,
            text_labels=[[phrase]],
        )[0]
        raw_boxes = detections.get("boxes", []) if hasattr(detections, "get") else []
        if hasattr(raw_boxes, "tolist"):
            raw_boxes = raw_boxes.tolist()
        boxes: List[Box] = []
        for raw_box in raw_boxes:
            if hasattr(raw_box, "tolist"):
                raw_box = raw_box.tolist()
            cleaned = clean_box(raw_box, shape)
            if cleaned is not None:
                boxes.append(cleaned)
        return boxes

    def segment_image(self, image_path: Path, phrase: str, shape: tuple[int, int]) -> SegmentationResult:
        import cv2
        from PIL import Image

        frame = cv2.imread(str(image_path))
        if frame is None:
            return SegmentationResult(phrase=phrase)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = SegmentationResult(phrase=phrase, mask_source="box")
        for track_id, box in enumerate(self._detect(image, phrase, shape)):
            key = str(track_id)
            result.objects.append(
                DetectedObject(track_id=key, frame_index=0, box=box, phrase=phrase)
            )
            result.tracks[key] = {0: box_mask(shape, box)}
        return result

    def segment_video(self, video_path: Path, phrase: str, shape: tuple[int, int]) -> SegmentationResult:
        import cv2
        from PIL import Image

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
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes = self._detect(image, phrase, shape)
            ids = tracker.assign(
                [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in boxes]
            )
            for track_id, box in zip(ids, boxes):
                key = str(track_id)
                result.objects.append(
                    DetectedObject(
                        track_id=key,
                        frame_index=frame_index,
                        box=box,
                        phrase=phrase,
                    )
                )
                result.tracks.setdefault(key, {})[frame_index] = box_mask(shape, box)
            frame_index += 1
        cap.release()
        print(
            f"Local OWLv2: '{phrase}' produced {len(result.tracks)} tracks "
            f"over {frame_index} frames"
        )
        return result
