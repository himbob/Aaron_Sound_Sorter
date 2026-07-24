#!/usr/bin/env python3
"""Summarize held-out vocal shadow evidence without rerunning inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.legacy_manifest import read_legacy_folder_map_by_hash


def read_csv_by_hash(path: Path, *, split: str | None = None) -> dict[str, dict[str, str]]:
    """Read a CSV into a file-hash map, optionally selecting one split."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        digest = row.get("file_sha256", "")
        if len(digest) != 64:
            raise ValueError(f"missing or invalid file_sha256 in {path}")
        if digest in selected:
            raise ValueError(f"duplicate file_sha256 in {path}: {digest}")
        selected[digest] = row
    return selected


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(rows: Sequence[dict[str, object]], field: str) -> float:
    return float(fmean(float(row[field]) for row in rows)) if rows else 0.0


def build_report_rows(
    manifest_by_hash: Mapping[str, Mapping[str, str]],
    predictions_by_hash: Mapping[str, Mapping[str, str]],
    legacy_folders_by_hash: Mapping[str, str],
    *,
    voice_label: str,
) -> list[dict[str, object]]:
    """Join held-out truth, neural predictions, and optional legacy placements."""
    missing = set(manifest_by_hash) - set(predictions_by_hash)
    extra = set(predictions_by_hash) - set(manifest_by_hash)
    if missing or extra:
        raise ValueError(f"manifest/prediction hash mismatch: missing={len(missing)} extra={len(extra)}")

    report_rows: list[dict[str, object]] = []
    for digest, manifest in sorted(manifest_by_hash.items()):
        prediction = predictions_by_hash[digest]
        expected_label = manifest["label"]
        if prediction.get("expected_label") != expected_label:
            raise ValueError(f"expected-label mismatch for {digest}")
        predicted_label = prediction["predicted_label"]
        legacy_folder = legacy_folders_by_hash.get(digest, "")
        report_rows.append(
            {
                "file_sha256": digest,
                "display_name": prediction.get("display_name", ""),
                "subgroup": manifest.get("subgroup", ""),
                "provenance": manifest.get("provenance", ""),
                "expected_label": expected_label,
                "predicted_label": predicted_label,
                "second_label": prediction.get("second_label", ""),
                "top_similarity": float(prediction["top_similarity"]),
                "second_similarity": float(prediction["second_similarity"]),
                "margin": float(prediction["margin"]),
                "known_distribution": prediction.get("known_distribution") == "1",
                "radius_ratio": float(prediction["radius_ratio"]),
                "legacy_folder": legacy_folder,
                "clap_voice_correct": expected_label == voice_label and predicted_label == voice_label,
                "clap_voice_false_positive": expected_label != voice_label and predicted_label == voice_label,
                "legacy_voice_family_correct": expected_label == voice_label
                and legacy_folder.startswith("Instruments/Voice/"),
                "human_verdict": "",
            }
        )
    return report_rows


def summarize_rows(rows: Sequence[dict[str, object]], *, voice_label: str) -> dict[str, object]:
    """Calculate broad-family, OOD, subgroup, and legacy comparison metrics."""
    vocal_rows = [row for row in rows if row["expected_label"] == voice_label]
    negative_rows = [row for row in rows if row["expected_label"] != voice_label]
    legacy_vocal_rows = [row for row in vocal_rows if row["legacy_folder"]]

    clap_vocal_correct = sum(bool(row["clap_voice_correct"]) for row in vocal_rows)
    negative_voice_false_positives = sum(bool(row["clap_voice_false_positive"]) for row in negative_rows)
    negative_known_voice_false_positives = sum(
        bool(row["clap_voice_false_positive"]) and bool(row["known_distribution"]) for row in negative_rows
    )
    known_count = sum(bool(row["known_distribution"]) for row in rows)
    vocal_known_count = sum(bool(row["known_distribution"]) for row in vocal_rows)
    top1_correct = sum(row["expected_label"] == row["predicted_label"] for row in rows)
    legacy_vocal_correct = sum(bool(row["legacy_voice_family_correct"]) for row in legacy_vocal_rows)

    comparison = Counter()
    for row in legacy_vocal_rows:
        clap_correct = bool(row["clap_voice_correct"])
        legacy_correct = bool(row["legacy_voice_family_correct"])
        if clap_correct and legacy_correct:
            comparison["both"] += 1
        elif clap_correct:
            comparison["clap_only"] += 1
        elif legacy_correct:
            comparison["legacy_only"] += 1
        else:
            comparison["neither"] += 1

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["expected_label"]), str(row["subgroup"]))].append(row)
    subgroup_metrics: list[dict[str, object]] = []
    for (expected_label, subgroup), group_rows in sorted(grouped.items()):
        correct_count = sum(row["predicted_label"] == expected_label for row in group_rows)
        predicted_voice_count = sum(row["predicted_label"] == voice_label for row in group_rows)
        group_known_count = sum(bool(row["known_distribution"]) for row in group_rows)
        subgroup_metrics.append(
            {
                "expected_label": expected_label,
                "subgroup": subgroup,
                "count": len(group_rows),
                "correct_count": correct_count,
                "top1_accuracy": _rate(correct_count, len(group_rows)),
                "predicted_voice_count": predicted_voice_count,
                "predicted_voice_rate": _rate(predicted_voice_count, len(group_rows)),
                "known_distribution_count": group_known_count,
                "review_rate": 1.0 - _rate(group_known_count, len(group_rows)),
                "mean_top_similarity": _mean(group_rows, "top_similarity"),
                "mean_margin": _mean(group_rows, "margin"),
                "mean_radius_ratio": _mean(group_rows, "radius_ratio"),
            }
        )

    return {
        "schema_version": 1,
        "definitions": {
            "voice_label": voice_label,
            "legacy_voice_family": "final folder begins with Instruments/Voice/",
            "review_rate": "fraction whose neural prediction is outside the learned prototype radius",
            "human_correct_win": "broad family comparison against the explicitly labeled vocal pack; no per-file human adjudication",
        },
        "heldout_count": len(rows),
        "vocal_count": len(vocal_rows),
        "protected_negative_count": len(negative_rows),
        "overall_top1_correct": top1_correct,
        "overall_top1_accuracy": _rate(top1_correct, len(rows)),
        "overall_known_distribution_count": known_count,
        "overall_review_rate": 1.0 - _rate(known_count, len(rows)),
        "heldout_vocal_correct": clap_vocal_correct,
        "heldout_vocal_recall": _rate(clap_vocal_correct, len(vocal_rows)),
        "heldout_vocal_known_distribution_count": vocal_known_count,
        "heldout_vocal_review_rate": 1.0 - _rate(vocal_known_count, len(vocal_rows)),
        "confusing_negative_voice_false_positives": negative_voice_false_positives,
        "confusing_negative_false_positive_rate": _rate(negative_voice_false_positives, len(negative_rows)),
        "confusing_negative_known_voice_false_positives": negative_known_voice_false_positives,
        "confusing_negative_known_false_positive_rate": _rate(
            negative_known_voice_false_positives,
            len(negative_rows),
        ),
        "legacy_pack_count": len(legacy_vocal_rows),
        "legacy_voice_family_correct": legacy_vocal_correct,
        "legacy_voice_family_recall": _rate(legacy_vocal_correct, len(legacy_vocal_rows)),
        "legacy_vs_clap_broad_family": dict(sorted(comparison.items())),
        "subgroups": subgroup_metrics,
        "limitations": [
            "The prototype index has only five approved musical-vocal trainers.",
            "OOD calibration is not yet learned; prototype-radius rejection remains conservative.",
            "Speech, breath, altered/formant voice FX, texture, and mixed-loop trainers are audited separately because legacy labels are disputed.",
            "No second music-focused encoder was available locally for a per-label leaderboard.",
        ],
    }


def summarize_supplement(
    manifest_by_hash: Mapping[str, Mapping[str, str]],
    shadow_by_hash: Mapping[str, Mapping[str, str]],
    *,
    voice_label: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Summarize disputed legacy-labeled confusers separately from clean truth."""
    missing = set(manifest_by_hash) - set(shadow_by_hash)
    extra = set(shadow_by_hash) - set(manifest_by_hash)
    if missing or extra:
        raise ValueError(f"supplement/shadow hash mismatch: missing={len(missing)} extra={len(extra)}")
    joined: list[dict[str, object]] = []
    for digest, manifest in sorted(manifest_by_hash.items()):
        prediction = shadow_by_hash[digest]
        predicted_voice = prediction["neural_label"] == voice_label
        known = prediction["neural_known_distribution"] == "1"
        joined.append(
            {
                "file_sha256": digest,
                "group": manifest["group"],
                "legacy_label": manifest["legacy_label"],
                "neural_label": prediction["neural_label"],
                "neural_second_label": prediction["neural_second_label"],
                "neural_similarity": float(prediction["neural_similarity"]),
                "neural_margin": float(prediction["neural_margin"]),
                "neural_radius_ratio": float(prediction["neural_radius_ratio"]),
                "neural_known_distribution": known,
                "predicted_voice": predicted_voice,
                "known_voice_claim": predicted_voice and known,
                "human_verdict": "",
            }
        )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in joined:
        grouped[str(row["group"])].append(row)
    groups: list[dict[str, object]] = []
    for group, group_rows in sorted(grouped.items()):
        voice_count = sum(bool(row["predicted_voice"]) for row in group_rows)
        known_voice_count = sum(bool(row["known_voice_claim"]) for row in group_rows)
        review_count = sum(not bool(row["neural_known_distribution"]) for row in group_rows)
        groups.append(
            {
                "group": group,
                "count": len(group_rows),
                "voice_claim_count": voice_count,
                "voice_claim_rate": _rate(voice_count, len(group_rows)),
                "known_voice_claim_count": known_voice_count,
                "known_voice_claim_rate": _rate(known_voice_count, len(group_rows)),
                "review_count": review_count,
                "review_rate": _rate(review_count, len(group_rows)),
                "mean_similarity": _mean(group_rows, "neural_similarity"),
                "mean_margin": _mean(group_rows, "neural_margin"),
            }
        )
    voice_count = sum(bool(row["predicted_voice"]) for row in joined)
    known_voice_count = sum(bool(row["known_voice_claim"]) for row in joined)
    summary = {
        "status": "diagnostic_only_disputed_legacy_labels",
        "count": len(joined),
        "voice_claim_count": voice_count,
        "voice_claim_rate": _rate(voice_count, len(joined)),
        "known_voice_claim_count": known_voice_count,
        "known_voice_claim_rate": _rate(known_voice_count, len(joined)),
        "groups": groups,
        "warning": "Do not report this set as clean held-out accuracy; its legacy labels are under contamination review.",
    }
    return joined, summary


def write_reports(
    rows: Sequence[dict[str, object]],
    summary: Mapping[str, object],
    output_dir: Path,
    *,
    supplement_rows: Sequence[dict[str, object]] = (),
) -> None:
    """Write joined per-file evidence, subgroup metrics, and JSON summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "verified_file_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    subgroup_path = output_dir / "verified_subgroup_metrics.csv"
    subgroups = list(summary.get("subgroups", []))
    with subgroup_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(subgroups[0]) if subgroups else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subgroups)

    if supplement_rows:
        supplement_path = output_dir / "verified_supplemental_confusers.csv"
        with supplement_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = list(supplement_rows[0])
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(supplement_rows)

    (output_dir / "verified_experiment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voice-label", default="Voice/Musical Vocal")
    parser.add_argument("--supplement-manifest", type=Path)
    parser.add_argument("--supplement-shadow", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build verified experiment reports from existing source-blind inference."""
    args = parse_args(argv)
    manifest = read_csv_by_hash(args.manifest, split="heldout_eval")
    predictions = read_csv_by_hash(args.predictions)
    legacy = read_legacy_folder_map_by_hash(args.legacy_manifest)
    rows = build_report_rows(manifest, predictions, legacy, voice_label=args.voice_label)
    summary = summarize_rows(rows, voice_label=args.voice_label)
    supplement_rows: list[dict[str, object]] = []
    if bool(args.supplement_manifest) != bool(args.supplement_shadow):
        raise ValueError("--supplement-manifest and --supplement-shadow must be supplied together")
    if args.supplement_manifest and args.supplement_shadow:
        supplement_manifest = read_csv_by_hash(args.supplement_manifest)
        supplement_shadow = read_csv_by_hash(args.supplement_shadow)
        supplement_rows, supplement_summary = summarize_supplement(
            supplement_manifest,
            supplement_shadow,
            voice_label=args.voice_label,
        )
        summary["supplemental_disputed_confusers"] = supplement_summary
    write_reports(rows, summary, args.output_dir, supplement_rows=supplement_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
