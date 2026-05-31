from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import category_stability_gate as gate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STABILITY_DIR = PROJECT_ROOT / "tests" / "stability"
SAMPLES_DIR = PROJECT_ROOT / "tests" / "acceptance" / "locked_smoke_v1" / "samples"


def test_anchor_packs_have_required_neighbor_trap_shapes() -> None:
    packs = gate.load_stability_packs(STABILITY_DIR)
    required = {
        "category_positive",
        "category_near_miss",
        "category_must_not_steal",
        "category_should_review",
    }

    assert packs
    for pack in packs:
        scenarios = {case.scenario for case in pack.cases}
        assert required.issubset(scenarios), pack.anchor_pack


def test_anchor_case_ids_are_unique_and_protected_samples_exist() -> None:
    packs = gate.load_stability_packs(STABILITY_DIR)
    cases = gate.flatten_cases(packs)
    ids = [case.id for case in cases]

    assert len(ids) == len(set(ids))
    for case in cases:
        if case.protected:
            assert (SAMPLES_DIR / case.filename).is_file(), case.id


def test_every_protected_case_has_a_firewall_or_review_escape() -> None:
    cases = gate.flatten_cases(gate.load_stability_packs(STABILITY_DIR))

    for case in cases:
        if not case.protected:
            continue
        assert case.forbidden_folder_prefixes or case.accepted_review_prefixes, case.id
