from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import category_stability_gate as gate
import locked_smoke_acceptance as smoke


def stability_case() -> gate.StabilityCase:
    return gate.StabilityCase(
        id="voice_firewall",
        filename="voice.wav",
        anchor_pack="voice",
        target_category="Instruments/Voice",
        scenario="category_positive",
        expected_top="Instruments",
        accepted_folder_prefixes=("Instruments/Voice",),
        accepted_review_prefixes=(),
        forbidden_folder_prefixes=("FX/Structural and Transitional FX/Risers and Builds", "Drums"),
        protected=True,
    )


def case_result(path: str, status: str = smoke.PASS) -> smoke.CaseResult:
    return smoke.CaseResult(
        id="voice_firewall",
        filename="voice.wav",
        protected=True,
        status=status,
        actual_path=path,
        actual_top=smoke.top_folder(path),
        expected_top="Instruments",
        matched_prefix="Instruments/Voice" if path.startswith("Instruments/Voice") else "",
        sorter_returncode=0,
        sample_path="/tmp/voice.wav",
        manifest_path="/tmp/manifest.csv",
        output_dir="/tmp/out",
        notes="",
    )


def test_forbidden_prefix_becomes_catastrophic_steal() -> None:
    result = gate.apply_firewall(
        stability_case(),
        case_result("FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX", smoke.FAIL_WRONG_TOP),
    )

    assert result.status == gate.FAIL_CATASTROPHIC_STEAL
    assert result.matched_prefix == "FX/Structural and Transitional FX/Risers and Builds"
    assert smoke.result_blocks_exit(result)


def test_accepted_folder_is_not_blocked_by_firewall() -> None:
    result = gate.apply_firewall(
        stability_case(),
        case_result("Instruments/Voice/Phrase/One Shots", smoke.PASS),
    )

    assert result.status == smoke.PASS
    assert result.actual_path == "Instruments/Voice/Phrase/One Shots"
