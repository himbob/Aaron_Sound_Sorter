"""Regression tests for compact brain ensemble audit logging."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aaron_sound_sorter.domain.models import (
    AudioPhysics,
    CategoryGuess,
    ConsensusDecision,
    SharedAudioFacts,
    SortFileResult,
    VoterResult,
)
from aaron_sound_sorter.infrastructure.brain_lane_validation import (
    brain_lane_group_winner_rows,
    brain_lane_validation_rows,
    competence_summary_rows,
    write_brain_lane_validation_reports,
)
from aaron_sound_sorter.infrastructure.report_writer import manifest_row, write_brain_ensemble_audit


def _guess(label: str, *, rank: int = 1, lanes: list[str] | None = None) -> CategoryGuess:
    evidence = {}
    if lanes is not None:
        evidence = {
            "brain_candidate_lanes": lanes,
            "brain_ensemble_support": 1.234,
            "brain_ensemble_score": 0.81,
            "brain_ensemble_weight_profile": "generic_pitched_full_guarded",
            "brain_ensemble_lane_trust_reason": "test profile reason",
        }
    return CategoryGuess(
        label=label,
        folder_path=label,
        top_family=label.split("/", 1)[0],
        score=float(rank),
        confidence=0.5,
        rank=rank,
        reason="test",
        evidence=evidence,
    )


def _digest(name: str, guesses: list[CategoryGuess], *, enabled: bool = True) -> dict:
    return {
        "voter_name": name,
        "diagnostics": {"enabled": enabled, "lane_name": name.replace("brain_", "")},
        "top_guesses": [
            {
                "label": guess.label,
                "folder_path": guess.folder_path,
                "rank": guess.rank,
                "score": guess.score,
                "confidence": guess.confidence,
                "evidence": guess.evidence,
            }
            for guess in guesses
        ],
    }


def _result(source_path: Path = Path("Loop/Brass_Woodwind/sample_sax.wav")) -> SortFileResult:
    final_label = "Instruments/Woodwinds/Saxophone/One Shots"
    final_folder = "Instruments/Brass and Woodwinds/Loops"
    ensemble_guess = _guess(final_label, lanes=["core_baby", "spread_baby", "outlier_baby"])
    full_guess = _guess("FX/Designed Noise FX/Siren/Long FX")
    core_guess = _guess(final_label)
    spread_guess = _guess(final_label)
    outlier_guess = _guess(final_label)
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "brain_ensemble_vote_result": {
                "voter_name": "brain",
                "diagnostics": {
                    "architecture": "weighted_brain_ensemble",
                    "ensemble_version": "v31.86_vocal_guarded_role_profile_rank_fusion",
                },
                "top_guesses": _digest("brain", [ensemble_guess])["top_guesses"],
            },
            "full_brain_vote_result": _digest("brain_full", [full_guess]),
            "core_baby_vote_result": _digest("brain_core_baby", [core_guess]),
            "spread_baby_vote_result": _digest("brain_spread_baby", [spread_guess]),
            "outlier_baby_vote_result": _digest("brain_outlier_baby", [outlier_guess]),
            "shape_vote": {"primary_shape": "pitched_phrase", "confidence": 0.91},
        },
    )
    return SortFileResult(
        source_path=source_path,
        placed_path=Path("out/sample.wav"),
        physics=AudioPhysics(
            source_path=Path("sample.wav"),
            fingerprint=np.zeros(4, dtype=np.float32),
            duration_sec=1.0,
            read_status="ok",
        ),
        facts=facts,
        brain_votes=VoterResult("brain", [ensemble_guess]),
        physics_votes=VoterResult("physics", [_guess("FX/Designed Noise FX/Siren/Long FX")]),
        decision=ConsensusDecision(
            final_label=final_label,
            final_top="Instruments",
            folder_path=final_folder,
            consensus_status="test_status",
            reason="test reason",
        ),
    )


def test_manifest_row_exposes_compact_brain_ensemble_fields() -> None:
    row = manifest_row(_result())

    assert row["brain_ensemble_mode"] == "weighted_brain_ensemble"
    assert row["brain_ensemble_version"] == "v31.86_vocal_guarded_role_profile_rank_fusion"
    assert row["brain_ensemble_vote_1"] == "Instruments/Woodwinds/Saxophone/One Shots"
    assert row["brain_ensemble_weight_profile"] == "generic_pitched_full_guarded"
    assert row["brain_ensemble_lane_trust_reason"] == "test profile reason"
    assert "core_baby" in row["brain_ensemble_vote_1_lanes"]
    assert row["final_agrees_with_brain_ensemble"] == "True"
    assert row["final_agrees_with_full_brain"] == "False"
    assert row["final_agrees_with_core_baby"] == "True"


def test_brain_ensemble_audit_csv_is_compact(tmp_path: Path) -> None:
    path = tmp_path / "Aaron_Brain_Ensemble_Audit.csv"

    write_brain_ensemble_audit(path, [_result()])
    text = path.read_text(encoding="utf-8")

    assert "brain_ensemble_vote_1" in text
    assert "full_brain_vote_1" in text
    assert "brain_ensemble_weight_profile" in text
    assert "brain_ensemble_lane_trust_reason" in text
    assert "shared_facts_json" not in text


def test_brain_lane_validation_matrix_reports_lane_competence(tmp_path: Path) -> None:
    result = _result(Path("FX_Aaron2/Loop/Brass_Woodwind/sample_sax.wav"))

    rows = brain_lane_validation_rows([result])
    summary = competence_summary_rows(rows)

    assert rows[0]["expected_group"] == "sax_reed"
    assert rows[0]["full_brain_top1_group"] == "true_transition_fx"
    assert rows[0]["full_brain_broad_safe"] == "False"
    assert rows[0]["core_baby_strict_correct"] == "True"
    assert rows[0]["outlier_baby_strict_correct"] == "True"
    assert rows[0]["expected_source_quality"] in {"weak_expected_label", "dirty_or_ambiguous_expected_label"}
    assert any(
        row["lane_name"] == "full_brain" and row["expected_group"] == "sax_reed" and row["fx_overcall_count"] == "1"
        for row in summary
    )
    winners = brain_lane_group_winner_rows(summary)
    assert any(
        row["expected_group"] == "sax_reed"
        and row["best_brain_strict_lane"] in {"brain_ensemble", "core_baby", "spread_baby", "outlier_baby"}
        for row in winners
    )

    write_brain_lane_validation_reports(tmp_path, [result])

    matrix_text = (tmp_path / "Aaron_Brain_Lane_Validation_Matrix.csv").read_text(encoding="utf-8")
    summary_text = (tmp_path / "Aaron_Brain_Lane_Competence_Summary.csv").read_text(encoding="utf-8")
    winners_text = (tmp_path / "Aaron_Brain_Lane_Group_Winners.csv").read_text(encoding="utf-8")
    assert "expected_group" in matrix_text
    assert "expected_source_quality" in matrix_text
    assert "core_baby_top5_json" in matrix_text
    assert "common_wrong_group" in summary_text
    assert "recommendation" in summary_text
    assert "best_brain_strict_lane" in winners_text
