from __future__ import annotations

from pathlib import Path

from aaron_sound_sorter.gui.models import PreviewRow
from aaron_sound_sorter.gui.neural_explanations import neural_evidence_lines


def _row() -> PreviewRow:
    return PreviewRow(
        row_id="00001",
        source_path=Path("/audio/by-hash.wav"),
        display_name="by-hash.wav",
        proposed_folder="_TO_REVIEW/Measured Role Conflict",
        approved_folder="_TO_REVIEW/Measured Role Conflict",
        final_top="_TO_REVIEW",
        consensus_status="neural_semantic_family_conflict_review",
        confidence=0.7,
        duration_sec=1.0,
        read_status="ok",
        decision_reason="independent disagreement",
        diagnostic_summary="neural evidence",
        neural_folder="Instruments/Voice/Vocal One Shots/One Shots",
        neural_known_distribution=True,
        neural_ownership_ready=False,
        neural_label_example_count=4,
        neural_exact_training_match=True,
        neural_semantic_family="brass",
        neural_semantic_score=0.31,
        neural_semantic_margin=0.08,
        neural_prompt_status="advisory_only",
        neural_prompt_suggestions=[
            {
                "path": "Instruments/Woodwinds/Saxophone/Alto/One Shots",
                "top_positive_prompt": "an alto saxophone one shot",
            }
        ],
        panns_status="advisory_only",
        panns_model_id="panns/test",
        panns_events=[{"label": "Saxophone", "score": 0.88}, {"label": "Music", "score": 0.74}],
    )


def test_neural_explanation_distinguishes_memory_clap_and_authority() -> None:
    lines = neural_evidence_lines(_row())

    assert lines[0].startswith("Your trained memory: exact human-approved audio match")
    assert lines[1] == "CLAP broad hearing: brass instrument (clear signal; this is not a probability)."
    assert "experimental; advice only" in lines[2]
    assert "alto saxophone one shot" in lines[3]
    assert lines[4] == "PANNs broad events (raw scores; advice only): Saxophone 0.88; Music 0.74."
    assert lines[-1].startswith("Neural result: sent to Review")


def test_neural_explanation_reports_unavailable_runtime() -> None:
    row = _row()
    row.neural_known_distribution = None

    assert neural_evidence_lines(row) == ["Neural audio: unavailable for this preview."]
