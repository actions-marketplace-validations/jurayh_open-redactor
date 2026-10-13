"""API polish: conflict validation and the quiet option."""

import shutil
from pathlib import Path

import pytest

from open_redactor import RedactionOptions, audit_media, redact_media

QR = Path(__file__).parent.parent / "eval" / "data" / "qr.mp4"


@pytest.fixture()
def clip(tmp_path: Path) -> Path:
    target = tmp_path / "qr.mp4"
    shutil.copy(QR, target)
    return target


@pytest.mark.parametrize(
    "options, fragment",
    [
        ({"provider": "owlv2", "backend": "api"}, "runs locally"),
        ({"provider": "yunet", "backend": "hosted"}, "runs locally"),
        ({"provider": "grounding-sam", "backend": "api"}, "runs locally"),
        ({"provider": "sam", "backend": "local"}, "remote SAM"),
        ({"local": True, "backend": "api"}, "Conflicting options"),
    ],
)
def test_conflicting_backend_provider_pairs_raise(clip: Path, options: dict, fragment: str):
    with pytest.raises(ValueError, match=fragment):
        redact_media(clip, options=options)


@pytest.mark.parametrize(
    "options",
    [
        {"provider": "owlv2"},
        {"provider": "owlv2", "backend": "local"},
        {"provider": "grounding-sam"},
        {"provider": "sam", "backend": "api"},
        {"provider": "sam", "backend": "hosted"},
        {"local": True},
        {"local": True, "backend": "local"},
        {},
    ],
)
def test_compatible_backend_provider_pairs_validate(options: dict):
    RedactionOptions(**options).validate()


def test_quiet_redact_writes_result_without_stdout(clip: Path, capsys):
    result = redact_media(clip, options={"sensitive": True, "quiet": True})
    captured = capsys.readouterr()
    assert "Rendering" not in captured.out
    assert "Input:" not in captured.out
    assert result.output_path and Path(result.output_path).exists()
    assert result.summary is not None
    assert result.summary["elements_found"] == 2


def test_default_redact_still_reports_progress(clip: Path, capsys):
    redact_media(clip, options={"sensitive": True})
    captured = capsys.readouterr()
    assert "Input:" in captured.out


def test_quiet_audit_writes_result_without_stdout(clip: Path, capsys):
    result = audit_media(clip, options={"sensitive": True, "quiet": True})
    captured = capsys.readouterr()
    assert "Input:" not in captured.out
    assert result.audit_path and Path(result.audit_path).exists()
    assert result.summary is not None
    assert result.summary["elements_found"] == 2
