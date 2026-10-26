"""Location audit: estimate how locatable a processed image still is.

Redaction removes the objects a user names. It says nothing about the
pixel layer as a whole: signage, architecture, vegetation, terrain,
and road furniture can still place a photo. This module asks a
geolocation model to place the processed output (or, in a shadow
audit, the source) and reports the estimate with the cues the model
used, so the summary can state the residual exposure instead of
implying the output is unlocatable.

The estimator is the standard tier Muse Spark model on the Meta
Model API. The contributor tier is never used here: it trades
training use of the traffic, which is the wrong trade for a privacy
feature holding a user's own photos. The estimate is one model's
best guess from one image or one sampled frame, and the summary
wording says so. A failed estimate is recorded as failed in the
summary, never silently omitted and never invented.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

from .sam_client import DEFAULT_ENDPOINT

ESTIMATOR_MODEL = "muse-spark-1.3"

AUDIT_NOTE = (
    "Redaction reduces locatability, it does not remove it. This estimate is "
    "one model's guess from the pixels of one image or sampled frame."
)

PROMPT = (
    "You are an expert geolocation analyst. Estimate where this photo was taken, "
    "from the pixels alone. Use every visible cue: signage and text, architecture, "
    "vegetation, terrain, road markings and furniture, driving side, vehicles, light.\n\n"
    "Reply with ONLY a JSON object, no other text:\n"
    '{"lat": <decimal degrees>, "lon": <decimal degrees>, "country": "<country name>", '
    '"city": "<nearest city name>", "confidence_25km": <0-100, your probability that the '
    'true location is within 25 km of your guess>, "cues": ["<cue>", "..."]}'
)


@dataclass
class LocationEstimate:
    latitude: float
    longitude: float
    country: str
    city: str
    confidence_25km: float  # 0..1, the model's stated probability
    cues: List[str] = field(default_factory=list)
    model: str = ESTIMATOR_MODEL


def build_estimate_body(image_b64: str, media_type: str = "image/jpeg") -> dict:
    """Build the Responses API body for one estimator call."""
    return {
        "model": ESTIMATOR_MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{image_b64}",
                    },
                ],
            }
        ],
        "stream": False,
    }


def parse_estimate(payload: dict) -> LocationEstimate:
    """Parse an estimator response payload into a LocationEstimate."""
    text = ""
    if isinstance(payload.get("output_text"), str):
        text = payload["output_text"]
    else:
        parts: List[str] = []
        for item in payload.get("output", []) or []:
            for content in item.get("content", []) or []:
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    parts.append(content["text"])
        text = "\n".join(parts)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("the estimator replied without a JSON verdict")
    try:
        verdict = json.loads(text[start:end + 1])
        confidence = float(verdict.get("confidence_25km", 0)) / 100.0
        return LocationEstimate(
            latitude=float(verdict["lat"]),
            longitude=float(verdict["lon"]),
            country=str(verdict.get("country", "")),
            city=str(verdict.get("city", "")),
            confidence_25km=min(1.0, max(0.0, confidence)),
            cues=[str(c) for c in verdict.get("cues", [])],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"the estimator verdict could not be parsed: {exc}") from exc


def estimate_location(
    image_path: Path,
    api_key: Optional[str],
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = 120.0,
) -> LocationEstimate:
    """Place one image file with the estimator. Raises RuntimeError."""
    if httpx is None:
        raise RuntimeError("httpx is required for the location audit. Install with pip install httpx")
    if not api_key:
        raise RuntimeError("no API key found for the location audit. Set the env var named by --api-key-env")
    suffix = image_path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(endpoint, json=build_estimate_body(image_b64, media_type), headers=headers)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        raise RuntimeError(f"the estimator call failed: {exc}") from exc
    return parse_estimate(payload)


def summary_block(
    estimate: Optional[LocationEstimate],
    basis: str,
    error: Optional[str] = None,
) -> dict:
    """The location_audit block stored in the summary JSON."""
    if estimate is None:
        return {
            "status": "failed",
            "estimator": ESTIMATOR_MODEL,
            "basis": basis,
            "error": error or "unknown error",
            "note": AUDIT_NOTE,
        }
    return {
        "status": "ok",
        "estimator": estimate.model,
        "basis": basis,
        "estimated_country": estimate.country,
        "estimated_city": estimate.city,
        "latitude": round(estimate.latitude, 4),
        "longitude": round(estimate.longitude, 4),
        "confidence_within_25km": round(estimate.confidence_25km, 2),
        "cues": list(estimate.cues),
        "note": AUDIT_NOTE,
    }


def text_lines(block: dict) -> List[str]:
    """Human-readable summary lines for a location_audit block."""
    if block.get("status") != "ok":
        return [
            f"Location audit: requested, but the estimate failed ({block.get('error', 'unknown error')}). "
            "No location estimate was produced, so do not treat this output as location-checked."
        ]
    place = block.get("estimated_city") or "unknown city"
    country = block.get("estimated_country") or "unknown country"
    lines = [
        f"Location audit: a geolocation model ({block.get('estimator')}) placed the {block.get('basis')} "
        f"near {place}, {country} ({block.get('latitude')}, {block.get('longitude')}); "
        f"estimated chance that placement is within 25 km: {block.get('confidence_within_25km', 0):.0%}"
    ]
    cues = [str(c) for c in block.get("cues", [])]
    if cues:
        joined = "; ".join(cues[:6])
        if len(joined) > 320:
            joined = joined[:317].rstrip() + "..."
        lines.append(f"Location cues the model used: {joined}")
    lines.append(f"Location note: {AUDIT_NOTE}")
    return lines


def run_location_audit(
    image_path: Optional[Path] = None,
    frame=None,
    *,
    api_key_env: str = "MODEL_API_KEY",
    endpoint: Optional[str] = None,
    basis: str = "processed output",
) -> dict:
    """Run the audit and return the summary block. Never raises.

    Pass either an image file path or a BGR frame array (written to a
    temporary JPEG). Failures come back as a failed block so the run
    that produced the media still completes and the summary records
    that no estimate exists.
    """
    try:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"no API key found. Set the env var named by --api-key-env ({api_key_env})")
        resolved_endpoint = endpoint or os.environ.get("OPEN_REDACTOR_ENDPOINT") or DEFAULT_ENDPOINT
        if image_path is not None:
            estimate = estimate_location(Path(image_path), api_key, endpoint=resolved_endpoint)
        elif frame is not None:
            import tempfile

            import cv2

            with tempfile.TemporaryDirectory(prefix="open-redactor-location-") as tmp:
                sample = Path(tmp) / "frame.jpg"
                if not cv2.imwrite(str(sample), frame):
                    raise RuntimeError("could not write the sampled frame for the location audit")
                estimate = estimate_location(sample, api_key, endpoint=resolved_endpoint)
        else:
            raise RuntimeError("the location audit needs an image or a frame")
        return summary_block(estimate, basis)
    except Exception as exc:
        print(f"Location audit failed: {exc}")
        return summary_block(None, basis, error=str(exc))
