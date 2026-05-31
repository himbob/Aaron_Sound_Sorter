from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from locked_smoke_acceptance import (  # noqa: E402
    FAIL_MISSING_MANIFEST,
    FAIL_MISSING_SAMPLE,
    FAIL_TIMEOUT,
    FAIL_WRONG_FOLDER,
    FAIL_WRONG_TOP,
    PASS,
    PASS_REVIEW,
    AcceptanceCase,
    AcceptanceRules,
    CaseResult,
    SorterResult,
    evaluate_case,
    evaluate_manifest_case,
    exit_code_for_results,
)


def protected_voice_case() -> AcceptanceCase:
    return AcceptanceCase(
        id="voice",
        filename="voice.wav",
        expected_top="Instruments",
        accepted_folder_prefixes=("Instruments/Voice",),
        accepted_review_prefixes=("_TO_REVIEW",),
        protected=True,
        notes="voice case",
    )


def fake_row(path: str, filename: str = "voice.wav") -> dict[str, str]:
    return {
        "source_path": f"/tmp/input/{filename}",
        "folder_path": path,
        "final_label": path,
        "final_top": path.split("/", 1)[0],
    }


def ok_sorter() -> SorterResult:
    return SorterResult(returncode=0, timed_out=False)


def sample_file(tmp_path: Path, name: str = "voice.wav") -> Path:
    path = tmp_path / name
    path.write_bytes(b"not real audio")
    return path


def test_accepted_folder_prefix_passes(tmp_path: Path) -> None:
    case = protected_voice_case()
    result = evaluate_manifest_case(
        case,
        fake_row("Instruments/Voice/Voice Loops"),
        sample_file(tmp_path),
        tmp_path / "out",
        tmp_path / "manifest.csv",
        ok_sorter(),
        AcceptanceRules(),
    )

    assert result.status == PASS
    assert result.matched_prefix == "Instruments/Voice"


def test_accepted_review_prefix_passes(tmp_path: Path) -> None:
    case = protected_voice_case()
    result = evaluate_manifest_case(
        case,
        fake_row("_TO_REVIEW/Measured Role Conflict"),
        sample_file(tmp_path),
        tmp_path / "out",
        tmp_path / "manifest.csv",
        ok_sorter(),
        AcceptanceRules(),
    )

    assert result.status == PASS_REVIEW
    assert result.matched_prefix == "_TO_REVIEW"


def test_wrong_top_fails(tmp_path: Path) -> None:
    case = protected_voice_case()
    result = evaluate_manifest_case(
        case,
        fake_row("FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX"),
        sample_file(tmp_path),
        tmp_path / "out",
        tmp_path / "manifest.csv",
        ok_sorter(),
        AcceptanceRules(),
    )

    assert result.status == FAIL_WRONG_TOP


def test_wrong_folder_fails_when_top_is_correct(tmp_path: Path) -> None:
    case = protected_voice_case()
    result = evaluate_manifest_case(
        case,
        fake_row("Instruments/Synth/Synth Loops/Loops"),
        sample_file(tmp_path),
        tmp_path / "out",
        tmp_path / "manifest.csv",
        ok_sorter(),
        AcceptanceRules(),
    )

    assert result.status == FAIL_WRONG_FOLDER


def test_missing_sample_fails(tmp_path: Path) -> None:
    case = protected_voice_case()
    result = evaluate_case(
        case,
        tmp_path / "missing.wav",
        tmp_path / "out",
        None,
        SorterResult(returncode=None, timed_out=False),
        AcceptanceRules(),
    )

    assert result.status == FAIL_MISSING_SAMPLE


def test_missing_manifest_fails(tmp_path: Path) -> None:
    case = protected_voice_case()
    result = evaluate_case(
        case,
        sample_file(tmp_path),
        tmp_path / "out",
        None,
        ok_sorter(),
        AcceptanceRules(),
    )

    assert result.status == FAIL_MISSING_MANIFEST


def test_timeout_status_fails(tmp_path: Path) -> None:
    case = protected_voice_case()
    result = evaluate_case(
        case,
        sample_file(tmp_path),
        tmp_path / "out",
        None,
        SorterResult(returncode=None, timed_out=True, error="Timed out"),
        AcceptanceRules(),
    )

    assert result.status == FAIL_TIMEOUT


def test_unprotected_failure_is_reported_but_does_not_fail_command() -> None:
    unprotected = CaseResult(
        id="diagnostic",
        filename="diagnostic.wav",
        protected=False,
        status=FAIL_WRONG_TOP,
        actual_path="FX/Risers",
        actual_top="FX",
        expected_top="Instruments",
        matched_prefix="",
        sorter_returncode=0,
        sample_path="/tmp/diagnostic.wav",
        manifest_path="/tmp/manifest.csv",
        output_dir="/tmp/out",
        notes="diagnostic only",
    )
    protected_pass = CaseResult(
        id="protected",
        filename="protected.wav",
        protected=True,
        status=PASS,
        actual_path="Instruments/Voice",
        actual_top="Instruments",
        expected_top="Instruments",
        matched_prefix="Instruments/Voice",
        sorter_returncode=0,
        sample_path="/tmp/protected.wav",
        manifest_path="/tmp/manifest.csv",
        output_dir="/tmp/out",
        notes="protected",
    )

    assert exit_code_for_results([unprotected, protected_pass]) == 0
