"""Photo orientation: EXIF rotation is applied on read for every format.

JPEGs from phones and cameras can store pixels sideways with an
orientation tag saying how to display them. OpenCV builds disagree on
whether imread applies the tag, so the pipeline pins the stored pixels
and corrects in one place. These tests pin the behavior against the
Pillow reference implementation for all eight orientation values.
"""

from pathlib import Path

import numpy as np
import piexif
import pytest
from PIL import Image, ImageOps

from open_redactor.image_pipeline import read_photo, run_image_pipeline


def pattern_image() -> Image.Image:
    img = Image.new("RGB", (48, 32), (0, 0, 0))
    px = img.load()
    for y in range(32):
        for x in range(48):
            if y < 16 and x < 24:
                px[x, y] = (220, 30, 30)
            elif y < 16:
                px[x, y] = (30, 220, 30)
            elif x < 24:
                px[x, y] = (30, 30, 220)
            else:
                px[x, y] = (240, 240, 240)
    return img


def dominant_quadrants(arr_rgb: np.ndarray):
    h, w = arr_rgb.shape[:2]
    out = []
    for sy, sx in [
        (slice(0, h // 2), slice(0, w // 2)),
        (slice(0, h // 2), slice(w // 2, w)),
        (slice(h // 2, h), slice(0, w // 2)),
        (slice(h // 2, h), slice(w // 2, w)),
    ]:
        m = arr_rgb[sy, sx].reshape(-1, 3).mean(axis=0)
        s = np.sort(m)
        out.append(int(np.argmax(m)) if s[2] - s[1] > 40 else 3)
    return tuple(out)


def tagged(path: Path, orientation: int) -> Path:
    pattern_image().save(path, quality=95)
    piexif.insert(piexif.dump({"0th": {piexif.ImageIFD.Orientation: orientation}}), str(path))
    return path


@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
def test_read_photo_matches_pillow_for_every_orientation(tmp_path: Path, orientation: int):
    photo = tagged(tmp_path / f"o{orientation}.jpg", orientation)
    ref = np.asarray(ImageOps.exif_transpose(Image.open(photo).convert("RGB")))
    got = read_photo(photo)[:, :, ::-1]
    assert got.shape == ref.shape
    assert dominant_quadrants(got) == dominant_quadrants(ref)


def test_untagged_photo_reads_unchanged(tmp_path: Path):
    photo = tmp_path / "plain.jpg"
    pattern_image().save(photo, quality=95)
    got = read_photo(photo)
    assert got.shape == (32, 48, 3)
    assert dominant_quadrants(got[:, :, ::-1]) == (0, 1, 2, 3)


def test_rotated_photo_pipeline_output_is_upright(tmp_path: Path):
    # Stored landscape with orientation 6 displays as portrait
    photo = tagged(tmp_path / "sideways.jpg", 6)
    out = tmp_path / "out.jpg"
    run_image_pipeline(photo, out, targets=[], local=True)
    with Image.open(out) as img:
        assert img.size == (32, 48)
