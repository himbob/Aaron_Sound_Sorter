#!/usr/bin/env python3
# SOURCE-NAME BLINDNESS INVARIANT:
# This audit uses numeric fingerprints and internal trained targets only.  It
# must never use source filenames, source folders, ZIP member names, or sample
# pack names as classification evidence.
"""Audit category-specific tolerance learned from GUI correction examples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aaron_audio_intelligence.adaptive_teacher_metric import adaptive_teacher_metric
from aaron_audio_intelligence.learned_memory_features import (
    PITCH_REGISTER_SENSITIVE_MEMORY_FEATURES,
    weighted_normalized_signature_vector,
)
from aaron_sound_sorter.voters.human_override_recall import (
    human_override_identity_neighbor_threshold,
    nearest_identity_neighbor_distance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("brain", type=Path, help="GUI user-memory brain JSON")
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument(
        "--extrapolation",
        type=float,
        default=0.30,
        help="Fraction beyond each teacher away from its nearest same-target neighbor (default: 0.30)",
    )
    return parser.parse_args()


def safe_int(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except Exception:
        return 0


def top_family(label: str) -> str:
    return str(label).split("/", 1)[0] if label else "Unknown"


def gui_signatures(brain: dict[str, Any], label: str) -> list[np.ndarray]:
    examples_by_label = brain.get("training_examples_detailed_by_label", {})
    examples = examples_by_label.get(label, []) if isinstance(examples_by_label, dict) else []
    include_names = set(str(name) for name in brain.get("feature_names", []))
    signatures: list[np.ndarray] = []
    for example in examples if isinstance(examples, list) else []:
        if not isinstance(example, dict) or not bool(example.get("incremental_gui_correction")):
            continue
        try:
            fingerprint = np.asarray(example.get("fingerprint", []), dtype=np.float32).reshape(-1)
        except Exception:
            continue
        signature = weighted_normalized_signature_vector(
            brain,
            fingerprint,
            include_feature_names=include_names,
            exclude_feature_names=PITCH_REGISTER_SENSITIVE_MEMORY_FEATURES,
            minimum_feature_count=24,
        )
        if signature.size:
            signatures.append(signature)
    return signatures


def label_weight(brain: dict[str, Any], label: str) -> int:
    reliability = brain.get("label_reliability_by_label", {})
    row = reliability.get(label, {}) if isinstance(reliability, dict) else {}
    if not isinstance(row, dict):
        return 1
    return max(
        1,
        safe_int(row.get("human_override_effective_weight")),
        safe_int(row.get("training_count")),
    )


def audit_label(
    brain: dict[str, Any],
    label: str,
    signatures: list[np.ndarray],
    extrapolation: float,
) -> dict[str, Any]:
    weight = label_weight(brain, label)
    count = len(signatures)
    row: dict[str, Any] = {
        "top_family": top_family(label),
        "label": label,
        "teacher_example_count": count,
        "effective_weight": weight,
        "mode": "adaptive_teacher_cloud" if count >= 2 else "single_teacher_prototype",
        "variable_feature_count": "",
        "stable_feature_count": "",
        "adaptive_threshold": "",
        "extrapolation_cases": 0,
        "extrapolation_matches": 0,
        "adaptive_cloud_matches": 0,
        "teacher_neighbor_rescues": 0,
        "extrapolation_match_rate": "",
        "status": "prototype_only_needs_more_examples" if count < 2 else "",
    }
    if count < 2:
        return row

    baseline = adaptive_teacher_metric(signatures[0], signatures, effective_weight=weight)
    row["variable_feature_count"] = baseline.variable_feature_count
    row["stable_feature_count"] = baseline.stable_feature_count
    row["adaptive_threshold"] = round(float(baseline.threshold), 6)

    case_count = min(8, count)
    matches = 0
    adaptive_matches = 0
    neighbor_rescues = 0
    neighbor_threshold = human_override_identity_neighbor_threshold(weight)
    for index in range(case_count):
        first = signatures[index]
        neighbor_candidates = [
            (nearest_identity_neighbor_distance(first, [candidate]), candidate)
            for other_index, candidate in enumerate(signatures)
            if other_index != index
        ]
        neighbor_candidates.sort(key=lambda item: item[0])
        second = neighbor_candidates[0][1]
        query = first + float(extrapolation) * (first - second)
        result = adaptive_teacher_metric(query, signatures, effective_weight=weight)
        neighbor_distance = nearest_identity_neighbor_distance(query, signatures)
        neighbor_guardrail_pass = bool(
            not result.available or (result.stable_violation_fraction <= 0.05 and result.stable_upper_rms <= 1.85)
        )
        neighbor_match = bool(neighbor_distance <= neighbor_threshold and neighbor_guardrail_pass)
        combined_match = bool(result.matched or neighbor_match)
        matches += int(combined_match)
        adaptive_matches += int(result.matched)
        neighbor_rescues += int(neighbor_match and not result.matched)

    rate = matches / max(1, case_count)
    row["extrapolation_cases"] = case_count
    row["extrapolation_matches"] = matches
    row["adaptive_cloud_matches"] = adaptive_matches
    row["teacher_neighbor_rescues"] = neighbor_rescues
    row["extrapolation_match_rate"] = round(rate, 6)
    if rate >= 0.75:
        row["status"] = "adaptive_tolerance_active"
    elif rate >= 0.50:
        row["status"] = "adaptive_tolerance_partial_review_examples"
    else:
        row["status"] = "teacher_cloud_too_sparse_or_multimodal"
    return row


def main() -> int:
    args = parse_args()
    brain = json.loads(args.brain.expanduser().read_text(encoding="utf-8"))
    labels = [str(label) for label in brain.get("labels", []) if str(label)]
    rows: list[dict[str, Any]] = []
    for label in sorted(labels):
        signatures = gui_signatures(brain, label)
        if signatures:
            rows.append(audit_label(brain, label, signatures, args.extrapolation))

    output_folder = args.output_folder.expanduser()
    output_folder.mkdir(parents=True, exist_ok=True)
    csv_path = output_folder / "adaptive_teacher_tolerance_audit.csv"
    fields = list(rows[0].keys()) if rows else ["label"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    adaptive = [row for row in rows if row["mode"] == "adaptive_teacher_cloud"]
    prototype = [row for row in rows if row["mode"] == "single_teacher_prototype"]
    active = [row for row in adaptive if row["status"] == "adaptive_tolerance_active"]
    partial = [row for row in adaptive if row["status"] == "adaptive_tolerance_partial_review_examples"]
    sparse = [row for row in adaptive if row["status"] == "teacher_cloud_too_sparse_or_multimodal"]
    total_cases = sum(int(row["extrapolation_cases"]) for row in adaptive)
    total_matches = sum(int(row["extrapolation_matches"]) for row in adaptive)
    family_counts = Counter(row["top_family"] for row in rows)
    family_active = Counter(row["top_family"] for row in active)

    summary_lines = [
        "Adaptive Teacher Tolerance Audit",
        "================================",
        f"Brain: {args.brain.expanduser().resolve()}",
        f"Trained targets audited: {len(rows)}",
        f"Adaptive targets (2+ examples): {len(adaptive)}",
        f"Prototype-only targets (1 example): {len(prototype)}",
        f"Adaptive targets active at >=75% extrapolation match: {len(active)}",
        f"Adaptive targets partial at 50-74%: {len(partial)}",
        f"Adaptive targets needing broader/more coherent examples: {len(sparse)}",
        f"All adaptive extrapolation cases matched: {total_matches}/{total_cases}",
        "",
        "By top family",
        "-------------",
    ]
    for family in sorted(family_counts):
        summary_lines.append(
            f"{family}: {family_counts[family]} trained targets; {family_active[family]} adaptive targets active"
        )
    summary_lines.extend(
        [
            "",
            "Interpretation",
            "--------------",
            "One example can teach an exact correction and a conservative nearby prototype,",
            "but it cannot reveal which dimensions legitimately vary across a category.",
            "Two or more examples activate learned per-feature tolerance. Stable dimensions",
            "remain guardrails; observed variable dimensions receive bounded flexibility.",
            "A bounded register-invariant neighborhood around each approved teacher also",
            "supports multimodal targets without turning the whole category into one giant cloud.",
            "The extrapolation test is a calibration audit, not proof that every unseen sound",
            "belongs to the target. Final placement still requires normal voter/physics agreement.",
        ]
    )
    summary_path = output_folder / "adaptive_teacher_tolerance_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"), end="")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
