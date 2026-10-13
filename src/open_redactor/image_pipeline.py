"""Photo pipeline. One image in, redacted image out, same layers as video."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .masks import build_per_frame_masks, select_tracks
from .pipeline import apply_redaction_to_frame, apply_secure_fill
from .sam_client import LocalSamStub, SamApiClient, SegmentationResult

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
HEIC_SUFFIXES = {".heic", ".heif"}

HEIC_INSTALL_HINT = (
    'HEIC and HEIF photos need the optional decoder. Install with: pip install "open-redactor[heic]"'
)


def _load_heif_frame(path: Path) -> np.ndarray:
    """Decode a HEIC or HEIF photo to a BGR array via Pillow and pillow-heif."""
    try:
        import pillow_heif
        from PIL import Image, ImageOps
    except Exception as exc:
        raise ValueError(f"{HEIC_INSTALL_HINT} Import failed: {exc}") from exc
    pillow_heif.register_heif_opener()
    with Image.open(path) as img:
        upright = ImageOps.exif_transpose(img)
        rgb = np.asarray(upright.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def _exif_orientation(path: Path) -> int:
    """The EXIF orientation tag of a photo, 1 when absent or unreadable."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            value = img.getexif().get(274, 1)
            return int(value) if value else 1
    except Exception:
        return 1


# Correction transforms keyed by EXIF orientation value. OpenCV decodes
# the stored pixels and ignores the tag, so without these a rotated
# phone photo would be detected and redacted sideways.
_ORIENTATION_FIXES = {
    2: lambda a: np.fliplr(a),
    3: lambda a: np.rot90(a, 2),
    4: lambda a: np.flipud(a),
    5: lambda a: np.swapaxes(a, 0, 1),
    6: lambda a: np.rot90(a, 3),
    7: lambda a: np.rot90(np.swapaxes(a, 0, 1), 2),
    8: lambda a: np.rot90(a, 1),
}


def apply_exif_orientation(frame: np.ndarray, path: Path) -> np.ndarray:
    """Rotate or flip a decoded frame to its visual orientation."""
    fix = _ORIENTATION_FIXES.get(_exif_orientation(path))
    if fix is None:
        return frame
    return fix(frame).copy()


def read_photo(path: Path) -> np.ndarray:
    """Read a photo to a BGR array. HEIC and HEIF go through pillow-heif
    because OpenCV cannot decode them. EXIF orientation is applied on
    every path so phone photos are processed upright. Newer OpenCV
    builds apply the tag inside imread and older ones ignore it, so the
    decode pins the stored pixels and the correction happens here,
    identically on every build."""
    flags = cv2.IMREAD_COLOR
    if hasattr(cv2, "IMREAD_IGNORE_ORIENTATION"):
        flags |= cv2.IMREAD_IGNORE_ORIENTATION
    frame = cv2.imread(str(path), flags)
    if frame is not None:
        return apply_exif_orientation(frame, path)
    if path.suffix.lower() in HEIC_SUFFIXES:
        return _load_heif_frame(path)
    raise ValueError(f"Could not read image: {path}")


def _reattach_exif(source: Path, output: Path) -> None:
    """Copy source EXIF onto a written photo for --keep-metadata.

    The decoded frame was already transposed upright on read, so the
    Orientation tag is dropped: keeping it would rotate the photo a
    second time in viewers.
    """
    try:
        from PIL import Image

        with Image.open(source) as src_img:
            exif = src_img.getexif()
            if not exif:
                return
            if 274 in exif:
                del exif[274]
            exif_bytes = exif.tobytes()
        with Image.open(output) as out_img:
            out_img.save(output, exif=exif_bytes)
    except Exception as exc:
        print(f"Metadata: could not copy EXIF onto the output ({exc})")


def write_photo(path: Path, frame_bgr: np.ndarray) -> None:
    """Write a BGR frame, keeping the input format. HEIC and HEIF outputs
    are encoded through Pillow with pillow-heif registered."""
    if path.suffix.lower() in HEIC_SUFFIXES:
        try:
            import pillow_heif
            from PIL import Image
        except Exception as exc:
            raise ValueError(f"{HEIC_INSTALL_HINT} Import failed: {exc}") from exc
        pillow_heif.register_heif_opener()
        rgb = frame_bgr[:, :, ::-1]
        Image.fromarray(rgb).save(str(path), quality=90)
        return
    if not cv2.imwrite(str(path), frame_bgr):
        raise RuntimeError(f"Could not write output image: {path}")


def run_image_pipeline(
    input_path: Path,
    output_path: Path,
    targets: List[str],
    mode: str = "blur",
    strength: int = 21,
    mask_margin: int = 10,
    local: bool = False,
    api_key_env: str = "MODEL_API_KEY",
    backend: str | None = None,
    endpoint: str | None = None,
    provider: str | None = None,
    pii_text: bool = False,
    codes: bool = False,
    keep_tracks: Optional[List[str]] = None,
    exclude_tracks: Optional[List[str]] = None,
    keep_metadata: bool = False,
) -> Dict[str, object]:
    frame = read_photo(input_path)
    height, width = frame.shape[:2]
    shape = (height, width)
    print(f"Input photo: {input_path} {width}x{height}")
    print(f"Targets: {', '.join(targets)}")

    # Detection clients read a file path and the API expects PNG or JPEG
    # bytes, so HEIC inputs are transcoded to a temp PNG for detection
    # only. Rotated photos of any format take the same path, because
    # detection has to run on the upright frame the redaction uses.
    # Redaction applies to the decoded original frame and the output
    # keeps the source format.
    segment_path = input_path
    _detect_tmp = None
    if input_path.suffix.lower() in HEIC_SUFFIXES or _exif_orientation(input_path) != 1:
        import tempfile

        _detect_tmp = tempfile.TemporaryDirectory(prefix="open-redactor-detect-")
        segment_path = Path(_detect_tmp.name) / "detect.png"
        cv2.imwrite(str(segment_path), frame)

    resolved_backend = backend or ("local" if local else "api")
    resolved_provider = provider or ("grounding-sam" if resolved_backend == "local" else "sam")
    if resolved_provider in ("yunet", "owlv2"):
        resolved_backend = "local"
    client: object
    if resolved_backend == "local":
        if resolved_provider == "grounding-sam":
            from .local_gsam import LocalGroundingSamClient

            client = LocalGroundingSamClient()
        elif resolved_provider == "yunet":
            from .yunet import YunetFaceClient

            client = YunetFaceClient()
        elif resolved_provider == "owlv2":
            from .owlv2 import Owlv2Client

            client = Owlv2Client()
        else:
            client = LocalSamStub()
    else:
        client = SamApiClient.from_env(api_key_env, endpoint=endpoint)
        if resolved_backend == "hosted":
            print(f"Hosted backend endpoint: {client.endpoint}")  # type: ignore[attr-defined]

    all_tracks: Dict[str, Dict[int, np.ndarray]] = {}
    total_objects = 0
    sam_sources: List[str] = []
    for phrase in targets:
        try:
            result: SegmentationResult = client.segment_image(segment_path, phrase, shape=shape)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"Warning: SAM failed for phrase '{phrase}': {exc}")
            result = SegmentationResult(phrase=phrase)
        total_objects += len(result.objects)
        if result.objects:
            sam_sources.append(result.mask_source)
        for track_id, frame_map in result.tracks.items():
            merged: Dict[int, np.ndarray] = {}
            for _fidx, m in frame_map.items():
                merged[0] = m
            if merged:
                all_tracks[f"{phrase}:{track_id}"] = merged

    frames = [frame]
    if pii_text:
        from .pii import PiiScanner

        for key, frame_boxes in PiiScanner(sample_every=1).scan_frames(frames).items():
            for fidx, boxes in frame_boxes.items():
                m = np.zeros(shape, dtype=bool)
                for x1, y1, x2, y2 in boxes:
                    m[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
                if np.any(m):
                    all_tracks[f"pii:{key}"] = {0: m}
                    total_objects += 1
    if codes:
        from .codes import CodeScanner

        for key, frame_boxes in CodeScanner(sample_every=1).scan_frames(frames).items():
            for fidx, boxes in frame_boxes.items():
                m = np.zeros(shape, dtype=bool)
                for x1, y1, x2, y2 in boxes:
                    m[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
                if np.any(m):
                    all_tracks[f"code:{key}"] = {0: m}
                    total_objects += 1

    # Keep or exclude tracks by key, same selection as the video pipeline
    left_visible: List[str] = []
    if keep_tracks or exclude_tracks:
        all_tracks, kept_visible, excluded, unmatched = select_tracks(
            all_tracks, keep=keep_tracks, exclude=exclude_tracks
        )
        left_visible = kept_visible + excluded
        for key in kept_visible:
            print(f"Kept visible, not redacted: {key}")
        for key in excluded:
            print(f"Excluded, not redacted: {key}")
        for pattern in unmatched:
            available = ", ".join(sorted(all_tracks.keys())) or "none"
            print(
                f"Warning: no track matches '{pattern}'. "
                f"Track keys in this photo: {available}"
            )

    if total_objects == 0:
        print("Nothing matched. Writing a clean copy.")
    masks = build_per_frame_masks(
        tracks=all_tracks, total_frames=1, shape=shape, margin=mask_margin, carry_frames=0, smooth_radius=1
    )
    if mode == "replace":
        from .replace import apply_replacement
        redacted = frame
        for track_key, track_map in all_tracks.items():
            single = build_per_frame_masks(
                tracks={track_key: track_map}, total_frames=1, shape=shape,
                margin=mask_margin, carry_frames=0, smooth_radius=1,
            )
            if np.any(single[0]):
                redacted = apply_replacement(redacted, single[0], track_key)
    else:
        base_frame = frame
        code_tracks = {key: value for key, value in all_tracks.items() if key.startswith("code:")}
        if code_tracks:
            code_masks = build_per_frame_masks(
                tracks=code_tracks, total_frames=1, shape=shape,
                margin=mask_margin, carry_frames=0, smooth_radius=1,
            )
            base_frame = apply_secure_fill(frame, code_masks[0])
            print("Codes: opaque fill applied so the payload cannot be decoded from a blurred pattern")
        if mode == "blur" and all_tracks:
            from .pipeline import build_blur_layers

            layers = build_blur_layers(
                tracks=all_tracks, total_frames=1, shape=shape,
                margin=mask_margin, carry_frames=0, base_strength=strength,
            )
            top_kernel = max(k for _m, k in layers)
            if top_kernel > strength:
                print(f"Adaptive blur: large regions use up to kernel {top_kernel} so they stay unreadable")
            redacted = base_frame
            for layer_masks, kernel in layers:
                redacted = apply_redaction_to_frame(redacted, layer_masks[0], mode="blur", strength=kernel)
        else:
            redacted = apply_redaction_to_frame(base_frame, masks[0], mode=mode if mode in ("blur", "pixelate") else "blur", strength=strength)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_photo(output_path, redacted)
    if _detect_tmp is not None:
        _detect_tmp.cleanup()
    print(f"Wrote: {output_path}")

    # Metadata: decoded frames carry no source tags, so the output is
    # clean by construction. Inspect anyway, report the findings, and
    # only carry tags over when the user asked to keep them.
    from .metadata import inspect_metadata, labels as metadata_labels

    meta_findings = inspect_metadata(input_path)
    found_labels = metadata_labels(meta_findings)
    metadata_info = {"found": found_labels, "stripped": bool(meta_findings) and not keep_metadata}
    if meta_findings:
        if keep_metadata:
            _reattach_exif(input_path, output_path)
            print(f"Metadata: {', '.join(found_labels)} found in the source and kept in the output by request")
        else:
            survivors = [l for l in metadata_labels(inspect_metadata(output_path)) if l in found_labels]
            if survivors:
                print(f"Metadata warning: {', '.join(survivors)} still present in the output")
            else:
                print(f"Metadata: {', '.join(found_labels)} found in the source, stripped from the output")
    if not sam_sources:
        sam_mask_source = "none"
    elif all(s == "pixel" for s in sam_sources):
        sam_mask_source = "pixel"
    elif all(s == "box" for s in sam_sources):
        sam_mask_source = "box"
    else:
        sam_mask_source = "mixed"
    return {"input": str(input_path), "output": str(output_path), "objects": total_objects, "photo": True, "kept_visible": left_visible, "sam_mask_source": sam_mask_source, "metadata": metadata_info}
