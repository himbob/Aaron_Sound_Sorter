from __future__ import annotations

from aaron_sound_sorter.voters.physics_category_layer import PhysicsCategoryLayer
from aaron_sound_sorter.voters.physics_layer_types import PhysicsLayerDecision


def make_decision(branch: str, flat: dict[str, float]) -> PhysicsLayerDecision:
    return PhysicsLayerDecision(
        top_family="Drums",
        top_confidence=0.86,
        branch=branch,
        branch_confidence=0.82,
        leaf_strategy="profile_leaf_with_drum_branch_and_category_panel_safeguard",
        evidence={"physics_category_panel_flat": flat},
    )


def test_category_layer_prefers_exact_high_scoring_leaf() -> None:
    layer = PhysicsCategoryLayer()
    decision = make_decision(
        "Kick",
        {
            "drums_kick_drums_sub_kick_one_shots_score": 0.90,
            "drums_kick_drums_generic_kick_one_shots_score": 0.22,
        },
    )

    score, reasons, evidence = layer.apply(
        "Drums/Kick Drums/Sub Kick/One Shots",
        0.80,
        decision,
    )

    assert score < 0.30
    assert any(reason.startswith("category_leaf_exact_target") for reason in reasons)
    assert evidence["physics_category_candidate_score_key"] == "drums_kick_drums_sub_kick_one_shots_score"
    assert evidence["physics_category_branch_selected_key"] == "drums_kick_drums_sub_kick_one_shots_score"


def test_category_layer_penalizes_weak_exact_leaf_when_top_is_decisive() -> None:
    layer = PhysicsCategoryLayer()
    decision = make_decision(
        "Kick",
        {
            "drums_kick_drums_sub_kick_one_shots_score": 0.90,
            "drums_kick_drums_generic_kick_one_shots_score": 0.12,
        },
    )

    score, reasons, evidence = layer.apply(
        "Drums/Kick Drums/Generic Kick/One Shots",
        0.50,
        decision,
    )

    assert score > 0.50
    assert "category_leaf_weak_exact:+0.12" in reasons
    assert evidence["physics_category_candidate_score_key"] == "drums_kick_drums_generic_kick_one_shots_score"


def test_category_layer_reports_no_panel_for_unknown_candidate() -> None:
    layer = PhysicsCategoryLayer()
    decision = make_decision("Kick", {})

    score, reasons, evidence = layer.apply("Drums/Unknown Bucket/One Shots", 0.42, decision)

    assert score == 0.42
    assert reasons == []
    assert evidence["physics_category_match_status"] == "no_matching_category_panel"
