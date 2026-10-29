"""Text PII detection for frames.

Two layers. Pattern matching on OCR text finds card numbers, US Social
Security numbers, phone numbers, emails, and GPS coordinate pairs of
the kind dashcams and camera apps burn into a frame as an overlay.
Card numbers are verified with the Luhn check so random long numbers
do not get flagged. Coordinate pairs must carry at least four decimal
places on both numbers and pass range checks, so prices, dimensions,
and version strings do not get flagged. Pairs may use a dot or a comma
as the decimal mark, carry hemisphere letters, degree signs, and label
tokens (LAT, LON, GPS) between the numbers, and use whitespace, comma,
semicolon, slash, or pipe separators. DMS forms are not matched. OCR
is optional and lazy through pytesseract plus the tesseract binary.
When OCR is not installed the scanner returns nothing with a clear
message and never invents detections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Box = Tuple[int, int, int, int]


@dataclass
class PiiHit:
    kind: str
    text: str
    start: int
    end: int


CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
SSN_RE = re.compile(r"(?<!\d)(\d{3})-(\d{2})-(\d{4})(?!\d)")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Decimal coordinate pairs as printed by dashcam and camera overlays:
# "-30.920099 146.96307E", "37.774929, -122.419418", "N 51.50722 W 0.12750",
# "LAT 37.774929 LON -122.419418", and decimal-comma forms such as
# "52,520008 13,404954". Both numbers need at least four decimal places;
# _valid_coordinate_pair then enforces latitude and longitude ranges in
# either order. The separator may be whitespace (spaces, tabs, newlines),
# a comma, semicolon, slash, or pipe, with an optional LAT/LON/GPS label
# token in the middle.
_SIGN = r"[+\-−]?"
_HEMI = r"[NSEW]"
_SEPARATOR = (
    r"(?:(?:\s*[,;/|:\-]\s*|\s+)"
    r"(?:(?:LAT(?:ITUDE)?|LON(?:GITUDE)?|GPS)\b\s*[:.\-]?\s*)?)"
)


def _pair_pattern(decimal_mark: str) -> re.Pattern:
    number = rf"{_SIGN}\d{{1,3}}{decimal_mark}\d{{4,}}"
    # A suffix hemisphere letter must not be the first letter of the
    # next word ("-122.419418 now" is not "...418 N"), hence the
    # letter lookahead after each suffix group.
    suffix = rf"({_HEMI}(?![A-Za-z]))?"
    return re.compile(
        rf"(?<![\d.,])"
        rf"({_HEMI})?\s*({number})\s*°?\s*{suffix}"
        rf"{_SEPARATOR}"
        rf"({_HEMI})?\s*({number})\s*°?\s*{suffix}"
        rf"(?![\d.,])",
        re.IGNORECASE,
    )


GPS_PAIR_RE = _pair_pattern(r"\.")
GPS_COMMA_PAIR_RE = _pair_pattern(r",")


def _coordinate_value(number: str, hemi_before: str, hemi_after: str) -> float:
    value = float(number.replace("−", "-").replace(",", "."))
    hemi = (hemi_after or hemi_before or "").upper()
    if hemi in ("S", "W"):
        value = -abs(value)
    return value


def _valid_coordinate_pair(m: re.Match) -> bool:
    """Range-check a candidate pair in lat,lon or lon,lat order.

    A pair of exact zeros is the no-fix placeholder overlays print
    before satellite lock, not a location, so it is rejected.
    """
    try:
        first = _coordinate_value(m.group(2), m.group(1), m.group(3))
        second = _coordinate_value(m.group(5), m.group(4), m.group(6))
    except (TypeError, ValueError):
        return False
    if first == 0.0 and second == 0.0:
        return False
    if abs(first) <= 90 and abs(second) <= 180:
        return True
    return abs(first) <= 180 and abs(second) <= 90


def _gps_candidates(text: str) -> List[Tuple[int, int]]:
    """All valid coordinate-pair spans in the text, overlaps included.

    The scan restarts one character past each candidate's start, valid
    or not. A plain finditer walk consumes a rejected candidate whole,
    so a neighbouring decimal (an altitude readout, a decimal
    timestamp) could swallow a real pair's latitude and hide the pair,
    or a valid-but-incidental pair could claim the latitude and leave
    the longitude exposed. Collecting every overlapping candidate and
    merging the overlaps into one span keeps the whole run covered.
    """
    spans: List[Tuple[int, int]] = []
    for pattern in (GPS_PAIR_RE, GPS_COMMA_PAIR_RE):
        pos = 0
        while pos < len(text):
            m = pattern.search(text, pos)
            if m is None:
                break
            # A prefix "hemisphere" glued to a word ("in 37.77...") is
            # the word's last letter, not a hemisphere marker; the clean
            # candidate starting at the number itself is found on the
            # next pass.
            glued = (
                m.group(1)
                and m.start() > 0
                and text[m.start() - 1].isalpha()
            )
            if not glued and _valid_coordinate_pair(m):
                spans.append((m.start(), m.end()))
            pos = m.start() + 1
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    # Hit spans carry no surrounding whitespace, so OCR word mapping
    # covers exactly the words the coordinates occupy.
    trimmed: List[Tuple[int, int]] = []
    for start, end in merged:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            trimmed.append((start, end))
    return trimmed


def luhn_ok(digits: str) -> bool:
    """Standard Luhn check used by payment card numbers."""
    nums = [int(c) for c in digits if c.isdigit()]
    if len(nums) < 13 or len(nums) > 19:
        return False
    total = 0
    alt = False
    for n in reversed(nums):
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0


def find_pii_in_text(text: str) -> List[PiiHit]:
    """Find verified PII in one block of OCR text."""
    hits: List[PiiHit] = []
    taken: List[Tuple[int, int]] = []

    def free(start: int, end: int) -> bool:
        return all(end <= s or start >= e for s, e in taken)

    for m in EMAIL_RE.finditer(text):
        if free(m.start(), m.end()):
            hits.append(PiiHit("email", m.group(0), m.start(), m.end()))
            taken.append((m.start(), m.end()))
    for m in SSN_RE.finditer(text):
        area, group, serial = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # SSA rules: area 0, 666, and 900 plus are never issued, group and serial are never 0
        if area == 0 or area == 666 or area >= 900 or group == 0 or serial == 0:
            continue
        if free(m.start(), m.end()):
            hits.append(PiiHit("ssn", m.group(0), m.start(), m.end()))
            taken.append((m.start(), m.end()))
    for start, end in _gps_candidates(text):
        # Runs before the card pass: a coordinate pair is a run of digits
        # and spaces that a Luhn check could claim by chance, and the pair
        # is a location either way.
        if free(start, end):
            hits.append(PiiHit("gps coordinates", text[start:end], start, end))
            taken.append((start, end))
    for m in CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if luhn_ok(digits) and free(m.start(), m.end()):
            hits.append(PiiHit("credit card number", m.group(0), m.start(), m.end()))
            taken.append((m.start(), m.end()))
    for m in PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        # Skip anything already claimed as a card and skip short local numbers
        if len(digits) >= 10 and free(m.start(), m.end()):
            # A phone span inside a Luhn valid card was already taken above
            hits.append(PiiHit("phone number", m.group(0), m.start(), m.end()))
            taken.append((m.start(), m.end()))
    return sorted(hits, key=lambda h: h.start)


def ocr_engine() -> str | None:
    """Return the available OCR engine name: tesseract or rapidocr."""
    try:
        import pytesseract  # noqa: F401
        import shutil

        if shutil.which("tesseract") is not None:
            return "tesseract"
    except Exception:
        pass
    try:
        import rapidocr_onnxruntime  # noqa: F401

        return "rapidocr"
    except Exception:
        return None


def ocr_available() -> bool:
    return ocr_engine() is not None


@dataclass
class PiiScanner:
    """Scan sampled frames with OCR and return box tracks per kind."""

    sample_every: int = 10
    tracks: Dict[str, Dict[int, List[Box]]] = field(default_factory=dict)

    def scan_frames(self, frames) -> Dict[str, Dict[int, List[Box]]]:
        """Frames are BGR numpy arrays. Returns {kind-index: {frame: [boxes]}}.

        OCR runs on sampled frames only and each hit box is held until the
        next sample, which matches how documents sit still on screen. The
        pipeline carry logic handles the final smoothing.
        """
        engine = ocr_engine()
        if engine is None:
            print(
                "Text PII scan needs an OCR engine: pytesseract plus the "
                "tesseract binary, or rapidocr-onnxruntime. Skipping text "
                "scan, object targets still run."
            )
            return {}
        if engine == "rapidocr":
            return self._scan_with_rapidocr(frames)
        import pytesseract
        from PIL import Image

        result: Dict[str, Dict[int, List[Box]]] = {}
        counters: Dict[str, int] = {}
        total = len(frames)
        for idx in range(0, total, max(1, self.sample_every)):
            frame = frames[idx]
            rgb = frame[:, :, ::-1]
            data = pytesseract.image_to_data(Image.fromarray(rgb), output_type=pytesseract.Output.DATAFRAME)
            if data is None or data.empty:
                continue
            # Group words into lines by block, paragraph, and line numbers
            lines: Dict[Tuple[int, int, int], List[Tuple[str, Box]]] = {}
            for _, row in data.iterrows():
                word = str(row.get("text", "") or "").strip()
                if not word or float(row.get("conf", -1) or -1) < 0:
                    continue
                key = (int(row.get("block_num", 0)), int(row.get("par_num", 0)), int(row.get("line_num", 0)))
                box = (
                    int(row["left"]),
                    int(row["top"]),
                    int(row["left"] + row["width"]),
                    int(row["top"] + row["height"]),
                )
                lines.setdefault(key, []).append((word, box))
            for words in lines.values():
                line_text = " ".join(w for w, _ in words)
                hits = find_pii_in_text(line_text)
                if not hits:
                    continue
                # Map character spans back to word boxes, simple span cover
                char_boxes: List[Tuple[int, int, Box]] = []
                pos = 0
                for word, box in words:
                    char_boxes.append((pos, pos + len(word), box))
                    pos += len(word) + 1
                for hit in hits:
                    covered = [b for s, e, b in char_boxes if e > hit.start and s < hit.end]
                    if not covered:
                        continue
                    x1 = min(b[0] for b in covered)
                    y1 = min(b[1] for b in covered)
                    x2 = max(b[2] for b in covered)
                    y2 = max(b[3] for b in covered)
                    counters[hit.kind] = counters.get(hit.kind, 0) + 1
                    track_key = f"{hit.kind}:{counters[hit.kind]}"
                    hold_until = min(total, idx + max(1, self.sample_every))
                    for f in range(idx, hold_until):
                        result.setdefault(track_key, {}).setdefault(f, []).append((x1, y1, x2, y2))
                    print(f"Text PII: {hit.kind} found near frame {idx}")
        return result


    def _scan_with_rapidocr(self, frames) -> Dict[str, Dict[int, List[Box]]]:
        """RapidOCR returns line boxes with text, which maps straight to hits."""
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR()
        result: Dict[str, Dict[int, List[Box]]] = {}
        counters: Dict[str, int] = {}
        total = len(frames)
        for idx in range(0, total, max(1, self.sample_every)):
            frame = frames[idx]
            ocr_result, _elapsed = engine(frame)
            if not ocr_result:
                continue
            for points, text, _score in ocr_result:
                hits = find_pii_in_text(str(text))
                if not hits:
                    continue
                xs = [pt[0] for pt in points]
                ys = [pt[1] for pt in points]
                box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                for hit in hits:
                    counters[hit.kind] = counters.get(hit.kind, 0) + 1
                    track_key = f"{hit.kind}:{counters[hit.kind]}"
                    hold_until = min(total, idx + max(1, self.sample_every))
                    for f in range(idx, hold_until):
                        result.setdefault(track_key, {}).setdefault(f, []).append(box)
                    print(f"Text PII: {hit.kind} found near frame {idx}")
        return result
