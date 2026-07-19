from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from fx_taxonomy_audit import (  # noqa: E402
    FxAuditSnapshot,
    FxRoleDefinition,
    build_label_coverage,
    build_role_coverage,
    role_for_label,
    training_slot_for_label,
)


def test_training_slot_for_public_fx_label() -> None:
    training_root = Path("training/locked_curated_v1")

    assert training_slot_for_label(
        training_root,
        "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX",
    ) == Path("training/locked_curated_v1/FX/Structural and Transitional FX/Risers and Builds/Generic Riser/_LONG_FX")
    assert training_slot_for_label(
        training_root,
        "FX/Impacts and Hits/Generic Impact/One Shots",
    ) == Path("training/locked_curated_v1/FX/Impacts and Hits/Generic Impact/_ONE_SHOTS")


def test_role_for_label_uses_explicit_ontology_prefix() -> None:
    roles = (
        FxRoleDefinition(
            role_id="fx_transition_motion",
            display_name="Transition Motion",
            label_prefixes=("FX/Structural and Transitional FX",),
            target_memory_roles=(),
            target_physics_branches=(),
            training_goal_per_role=10,
        ),
    )

    assert (
        role_for_label("FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX", roles)
        == "fx_transition_motion"
    )
    assert role_for_label("FX/Unknown/Leaf/One Shots", roles) == "fx_unmapped"


def test_role_coverage_recommends_trusted_seed_before_runtime_tuning() -> None:
    roles = (
        FxRoleDefinition(
            role_id="fx_impact_punctuation",
            display_name="Impact and Punctuation",
            label_prefixes=("FX/Impacts and Hits",),
            target_memory_roles=("fx_impact",),
            target_physics_branches=("FX/ImpactHit",),
            training_goal_per_role=12,
        ),
    )
    snapshot = FxAuditSnapshot(
        roles=roles,
        brain_counts={
            "FX/Impacts and Hits/Generic Impact/Long FX": 96,
            "FX/Impacts and Hits/Generic Impact/One Shots": 96,
        },
        gui_labels={"FX/Impacts and Hits/Generic Impact/Long FX"},
        locked_training_counts={
            "FX/Impacts and Hits/Generic Impact/Long FX": 1,
            "FX/Impacts and Hits/Generic Impact/One Shots": 0,
        },
        user_memory_labels=set(),
        voter_memory_ids={"fx_impact"},
        physics_memory_ids={"FX/ImpactHit"},
    )

    label_rows = build_label_coverage(snapshot)
    role_rows = build_role_coverage(snapshot, label_rows)

    assert label_rows[1].recommendation == "seed_locked_training"
    assert role_rows[0].recommendation == "seed_trusted_fx_role_panel"


def test_ontology_json_keeps_fx_roles_broad_and_structured() -> None:
    ontology_path = ROOT / "config" / "fx_role_ontology_v1.json"
    payload = json.loads(ontology_path.read_text(encoding="utf-8"))
    role_ids = {role["role_id"] for role in payload["roles"]}

    assert "fx_transition_motion" in role_ids
    assert "fx_impact_punctuation" in role_ids
    assert "fx_texture_soundscape" in role_ids
