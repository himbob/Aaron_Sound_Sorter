"""Diagnostic brain-lane validation reports.

These reports are intentionally post-decision diagnostics. They may inspect
source paths to infer an expected audit group, but the results never feed back
into routing, voting, eligibility, or final placement.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aaron_sound_sorter.domain.models import SortFileResult

LANE_SPECS = (
    ("brain_ensemble", "brain_ensemble_vote_result"),
    ("full_brain", "full_brain_vote_result"),
    ("core_baby", "core_baby_vote_result"),
    ("spread_baby", "spread_baby_vote_result"),
    ("outlier_baby", "outlier_baby_vote_result"),
    ("harmonic_core_baby", "harmonic_core_baby_vote_result"),
    ("harmonic_spread_baby", "harmonic_spread_baby_vote_result"),
    ("harmonic_outlier_baby", "harmonic_outlier_baby_vote_result"),
)

BRAIN_LANES = tuple(lane for lane, _key in LANE_SPECS)
SUMMARY_LANES = BRAIN_LANES + ("physics", "final")

STRONG_EXPECTED = "strong_expected_label"
WEAK_EXPECTED = "weak_expected_label"
DIRTY_EXPECTED = "dirty_or_ambiguous_expected_label"
UNKNOWN_EXPECTED = "unknown_expected_label"

REPORT_VERSION = "v31_95_expected_quality_and_fx_subgroups"


def write_brain_lane_validation_reports(output_dir: Path, file_results: list[SortFileResult]) -> None:
    """Write diagnostic matrix, summary, and best-lane CSVs."""
    matrix_rows = brain_lane_validation_rows(file_results)
    summary_rows = competence_summary_rows(matrix_rows)
    write_validation_matrix(output_dir / "Aaron_Brain_Lane_Validation_Matrix.csv", matrix_rows)
    write_competence_summary_rows(output_dir / "Aaron_Brain_Lane_Competence_Summary.csv", summary_rows)
    write_group_winners(output_dir / "Aaron_Brain_Lane_Group_Winners.csv", summary_rows)


def brain_lane_validation_rows(file_results: Iterable[SortFileResult]) -> list[dict[str, str]]:
    """Return one diagnostic validation row per sorted file."""
    rows: list[dict[str, str]] = []
    for result in file_results:
        profile = expected_profile_from_source_path(result.source_path)
        expected_group = profile["expected_group"]
        expected_tops = compatible_tops_for_expected_group(expected_group)
        row: dict[str, str] = {
            "source_path": str(result.source_path),
            "expected_group": expected_group,
            "expected_source_quality": profile["expected_source_quality"],
            "expected_group_reason": profile["expected_group_reason"],
            "expected_profile_version": REPORT_VERSION,
            "expected_compatible_tops": json.dumps(expected_tops),
            "final_folder": result.decision.folder_path,
            "final_top": result.decision.final_top,
            "final_group": label_group(result.decision.folder_path or result.decision.final_label),
            "final_strict_correct": bool_text(
                groups_are_strict_match(
                    label_group(result.decision.folder_path),
                    expected_group,
                )
            ),
            "final_broad_safe": bool_text(is_broad_safe(result.decision.folder_path, expected_group)),
            "consensus_status": result.decision.consensus_status,
            "shape_vote": shape_value(result, "primary_shape"),
            "shape_confidence": shape_value(result, "confidence"),
        }
        add_lane_columns(row, "physics", [guess.label for guess in result.physics_votes.guesses[:5]], expected_group)
        add_lane_columns(row, "final", [result.decision.folder_path], expected_group)
        evidence = result.facts.evidence if isinstance(result.facts.evidence, dict) else {}
        for lane_name, digest_key in LANE_SPECS:
            labels = digest_top_labels(evidence.get(digest_key, {}), limit=5)
            add_lane_columns(row, lane_name, labels, expected_group)
        rows.append(row)
    return rows


def write_validation_matrix(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the per-file diagnostic validation matrix."""
    fields = validation_matrix_fields()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_competence_summary(path: Path, matrix_rows: list[dict[str, str]]) -> None:
    """Write lane-level competence totals by expected source group."""
    write_competence_summary_rows(path, competence_summary_rows(matrix_rows))


def write_competence_summary_rows(path: Path, summary_rows: list[dict[str, str]]) -> None:
    """Write precomputed lane-level competence rows."""
    fields = competence_summary_fields()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def competence_summary_fields() -> list[str]:
    """Return stable fields for the lane competence summary."""
    return [
        "lane_name",
        "expected_group",
        "expected_source_quality",
        "total",
        "strict_correct_count",
        "broad_safe_count",
        "wrong_count",
        "strict_accuracy",
        "broad_safe_accuracy",
        "common_wrong_group",
        "common_wrong_top_family",
        "voice_overcall_count",
        "guitar_overcall_count",
        "fx_overcall_count",
        "drum_overcall_count",
        "dead_lane_count",
        "recommendation",
    ]


def write_group_winners(path: Path, summary_rows: list[dict[str, str]]) -> None:
    """Write one row per expected group showing which lane is best for what."""
    rows = brain_lane_group_winner_rows(summary_rows)
    fields = group_winner_fields()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def group_winner_fields() -> list[str]:
    """Return stable fields for the group winner report."""
    return [
        "expected_group",
        "expected_source_quality",
        "total_examples",
        "best_brain_strict_lane",
        "best_brain_strict_accuracy",
        "best_brain_broad_lane",
        "best_brain_broad_safe_accuracy",
        "best_nonfinal_strict_lane",
        "best_nonfinal_strict_accuracy",
        "best_nonfinal_broad_lane",
        "best_nonfinal_broad_safe_accuracy",
        "final_strict_accuracy",
        "final_broad_safe_accuracy",
        "physics_strict_accuracy",
        "physics_broad_safe_accuracy",
        "dead_or_missing_brain_lanes",
        "watch_lanes",
        "brain_lane_rankings_json",
    ]


def brain_lane_group_winner_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return per-group best-lane rows for diagnostic brain study.

    This is intentionally report-only. It ranks existing diagnostic lane
    results and never feeds back into routing or voter scores.
    """
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        expected = row.get("expected_group", "unknown") or "unknown"
        quality = row.get("expected_source_quality", "") or UNKNOWN_EXPECTED
        if int_value(row.get("total", "0")) > 0:
            grouped[(expected, quality)].append(row)

    rows: list[dict[str, str]] = []
    for expected, quality in sorted(grouped):
        bucket = grouped[(expected, quality)]
        brain_rows = [row for row in bucket if row.get("lane_name") in BRAIN_LANES]
        nonfinal_rows = [row for row in bucket if row.get("lane_name") != "final"]
        best_brain_strict = best_lane_row(brain_rows, primary="strict_accuracy")
        best_brain_broad = best_lane_row(brain_rows, primary="broad_safe_accuracy")
        best_nonfinal_strict = best_lane_row(nonfinal_rows, primary="strict_accuracy")
        best_nonfinal_broad = best_lane_row(nonfinal_rows, primary="broad_safe_accuracy")
        final_row = first_lane_row(bucket, "final")
        physics_row = first_lane_row(bucket, "physics")
        total_examples = max((int_value(row.get("total", "0")) for row in bucket), default=0)
        watch_lanes = [
            row.get("lane_name", "") for row in bucket if str(row.get("recommendation", "")).startswith("watch_")
        ]
        dead_lanes = [
            row.get("lane_name", "")
            for row in brain_rows
            if row.get("recommendation") == "dead_or_missing_lane_diagnostic_only"
        ]
        rankings = [
            {
                "lane_name": row.get("lane_name", ""),
                "strict_accuracy": row.get("strict_accuracy", ""),
                "broad_safe_accuracy": row.get("broad_safe_accuracy", ""),
                "strict_correct_count": row.get("strict_correct_count", ""),
                "broad_safe_count": row.get("broad_safe_count", ""),
                "wrong_count": row.get("wrong_count", ""),
                "common_wrong_group": row.get("common_wrong_group", ""),
                "dead_lane_count": row.get("dead_lane_count", ""),
                "recommendation": row.get("recommendation", ""),
            }
            for row in sorted(brain_rows, key=lambda item: lane_sort_key(item, "strict_accuracy"), reverse=True)
        ]
        rows.append(
            {
                "expected_group": expected,
                "expected_source_quality": quality,
                "total_examples": str(total_examples),
                "best_brain_strict_lane": lane_name(best_brain_strict),
                "best_brain_strict_accuracy": metric_text(best_brain_strict, "strict_accuracy"),
                "best_brain_broad_lane": lane_name(best_brain_broad),
                "best_brain_broad_safe_accuracy": metric_text(best_brain_broad, "broad_safe_accuracy"),
                "best_nonfinal_strict_lane": lane_name(best_nonfinal_strict),
                "best_nonfinal_strict_accuracy": metric_text(best_nonfinal_strict, "strict_accuracy"),
                "best_nonfinal_broad_lane": lane_name(best_nonfinal_broad),
                "best_nonfinal_broad_safe_accuracy": metric_text(best_nonfinal_broad, "broad_safe_accuracy"),
                "final_strict_accuracy": metric_text(final_row, "strict_accuracy"),
                "final_broad_safe_accuracy": metric_text(final_row, "broad_safe_accuracy"),
                "physics_strict_accuracy": metric_text(physics_row, "strict_accuracy"),
                "physics_broad_safe_accuracy": metric_text(physics_row, "broad_safe_accuracy"),
                "dead_or_missing_brain_lanes": ";".join(lane for lane in dead_lanes if lane),
                "watch_lanes": ";".join(lane for lane in watch_lanes if lane),
                "brain_lane_rankings_json": json.dumps(rankings, sort_keys=True),
            }
        )
    return rows


def best_lane_row(rows: list[dict[str, str]], *, primary: str) -> dict[str, str]:
    """Return the best lane row according to one metric and safe tie-breakers."""
    if not rows:
        return {}
    return max(rows, key=lambda row: lane_sort_key(row, primary))


def first_lane_row(rows: list[dict[str, str]], lane: str) -> dict[str, str]:
    """Return the first row for a named lane."""
    for row in rows:
        if row.get("lane_name") == lane:
            return row
    return {}


def lane_sort_key(row: dict[str, str], primary: str) -> tuple[float, float, int, int, int, int, int]:
    """Sort lanes by accuracy first and overcalls last."""
    secondary = "broad_safe_accuracy" if primary == "strict_accuracy" else "strict_accuracy"
    overcalls = (
        int_value(row.get("voice_overcall_count", "0"))
        + int_value(row.get("guitar_overcall_count", "0"))
        + int_value(row.get("fx_overcall_count", "0"))
        + int_value(row.get("drum_overcall_count", "0"))
    )
    dead_count = int_value(row.get("dead_lane_count", "0"))
    return (
        float_value(row.get(primary, "0")),
        float_value(row.get(secondary, "0")),
        int_value(row.get("strict_correct_count", "0")),
        int_value(row.get("broad_safe_count", "0")),
        int_value(row.get("total", "0")),
        -overcalls,
        -dead_count,
    )


def lane_name(row: dict[str, str]) -> str:
    """Return the lane name from a possibly empty row."""
    return row.get("lane_name", "") if row else ""


def metric_text(row: dict[str, str], field: str) -> str:
    """Return a metric value from a possibly empty row."""
    return row.get(field, "") if row else ""


def float_value(value: str) -> float:
    """Parse a CSV numeric field robustly."""
    text = str(value or "0").strip()
    if text.endswith("%"):
        text = text[:-1].strip()
        try:
            return float(text) / 100.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def int_value(value: str) -> int:
    """Parse a CSV integer field robustly."""
    try:
        return int(float(str(value or "0").strip()))
    except ValueError:
        return 0


def competence_summary_rows(matrix_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Summarize strict and broad-safe lane behavior by expected group and source quality."""
    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in matrix_rows:
        expected = row.get("expected_group", "unknown") or "unknown"
        quality = row.get("expected_source_quality", "") or UNKNOWN_EXPECTED
        for lane in SUMMARY_LANES:
            if row.get(f"{lane}_top1") or lane in BRAIN_LANES:
                buckets[(lane, expected, quality)].append(row)

    rows: list[dict[str, str]] = []
    for (lane, expected, quality), bucket in sorted(buckets.items()):
        total = len(bucket)
        strict = sum(1 for row in bucket if row.get(f"{lane}_strict_correct") == "True")
        broad = sum(1 for row in bucket if row.get(f"{lane}_broad_safe") == "True")
        wrong_groups = [
            row.get(f"{lane}_top1_group", "unknown") or "unknown"
            for row in bucket
            if row.get(f"{lane}_strict_correct") != "True"
        ]
        wrong_tops = [
            row.get(f"{lane}_top1_top_family", "unknown") or "unknown"
            for row in bucket
            if row.get(f"{lane}_strict_correct") != "True"
        ]
        voice_overcalls = count_overcall(bucket, lane, "voice", expected)
        guitar_overcalls = count_overcall(bucket, lane, "guitar", expected)
        fx_overcalls = count_overcall(bucket, lane, "fx", expected) + count_overcall(
            bucket, lane, "true_transition_fx", expected
        )
        drum_overcalls = count_overcall(bucket, lane, "drums", expected)
        dead_lane_count = sum(1 for row in bucket if lane_is_dead(row, lane))
        rows.append(
            {
                "lane_name": lane,
                "expected_group": expected,
                "expected_source_quality": quality,
                "total": str(total),
                "strict_correct_count": str(strict),
                "broad_safe_count": str(broad),
                "wrong_count": str(total - strict),
                "strict_accuracy": ratio_text(strict, total),
                "broad_safe_accuracy": ratio_text(broad, total),
                "common_wrong_group": most_common(wrong_groups),
                "common_wrong_top_family": most_common(wrong_tops),
                "voice_overcall_count": str(voice_overcalls),
                "guitar_overcall_count": str(guitar_overcalls),
                "fx_overcall_count": str(fx_overcalls),
                "drum_overcall_count": str(drum_overcalls),
                "dead_lane_count": str(dead_lane_count),
                "recommendation": lane_recommendation(
                    total=total,
                    strict=strict,
                    broad=broad,
                    voice_overcalls=voice_overcalls,
                    guitar_overcalls=guitar_overcalls,
                    fx_overcalls=fx_overcalls,
                    drum_overcalls=drum_overcalls,
                    dead_lane_count=dead_lane_count,
                ),
            }
        )
    return rows


def add_lane_columns(row: dict[str, str], lane: str, labels: list[str], expected_group: str) -> None:
    """Add top labels and correctness flags for one diagnostic lane."""
    top1 = labels[0] if labels else ""
    top_group = label_group(top1)
    row[f"{lane}_top1"] = top1
    row[f"{lane}_top1_group"] = top_group
    row[f"{lane}_top1_top_family"] = top_family(top1)
    row[f"{lane}_top5_json"] = json.dumps(labels, sort_keys=True)
    row[f"{lane}_strict_correct"] = bool_text(groups_are_strict_match(top_group, expected_group))
    row[f"{lane}_broad_safe"] = bool_text(is_broad_safe(top1, expected_group))
    row[f"{lane}_overcall"] = overcall_name(top_group, expected_group)


def validation_matrix_fields() -> list[str]:
    """Return stable CSV fields for the validation matrix."""
    fields = [
        "source_path",
        "expected_group",
        "expected_source_quality",
        "expected_group_reason",
        "expected_profile_version",
        "expected_compatible_tops",
        "final_folder",
        "final_top",
        "final_group",
        "final_strict_correct",
        "final_broad_safe",
        "consensus_status",
        "shape_vote",
        "shape_confidence",
    ]
    for lane in ("physics", "final") + tuple(lane for lane, _key in LANE_SPECS):
        fields.extend(
            [
                f"{lane}_top1",
                f"{lane}_top1_group",
                f"{lane}_top1_top_family",
                f"{lane}_top5_json",
                f"{lane}_strict_correct",
                f"{lane}_broad_safe",
                f"{lane}_overcall",
            ]
        )
    return fields


def digest_top_labels(digest: Any, *, limit: int) -> list[str]:
    """Return top labels from a serialized voter digest."""
    if not isinstance(digest, dict):
        return []
    guesses = digest.get("top_guesses", [])
    if not isinstance(guesses, list):
        return []
    labels: list[str] = []
    for guess in guesses[:limit]:
        if isinstance(guess, dict):
            labels.append(str(guess.get("label") or guess.get("folder_path") or ""))
    return [label for label in labels if label]


def expected_profile_from_source_path(path: Path) -> dict[str, str]:
    """Infer expected diagnostic group plus label quality from path/name.

    This is a report-only profile. It is deliberately independent from routing.
    """
    text = normalize_text("/".join(path.parts[-10:]))
    explicit = explicit_expected_group(text)
    if explicit:
        quality = explicit_quality(text, default=STRONG_EXPECTED)
        return {
            "expected_group": explicit,
            "expected_source_quality": quality,
            "expected_group_reason": "explicit_expected_group_folder",
        }

    quality = explicit_quality(text, default=WEAK_EXPECTED)
    group = expected_group_from_normalized_text(text)
    reason = "path_token_inference"

    if group == "fx":
        split_group = split_fx_pack_expected_group(text)
        if split_group != "fx":
            group = split_group
            reason = "fx_pack_subgroup_split"

    if group == "unknown":
        quality = UNKNOWN_EXPECTED
        reason = "no_reliable_path_tokens"
    elif quality == WEAK_EXPECTED and is_dirty_or_ambiguous_source(text, group):
        quality = DIRTY_EXPECTED
        reason = "ambiguous_or_dirty_source_tokens"

    return {
        "expected_group": group,
        "expected_source_quality": quality,
        "expected_group_reason": reason,
    }


def explicit_expected_group(text: str) -> str:
    """Read a group from generated expected-group folder names."""
    candidates = [
        "true_transition_fx",
        "melody_loop_inside_fx_pack",
        "multi_instrument_loop",
        "string_or_synth_loop",
        "sax_reed",
        "drums",
        "voice",
        "bass",
        "guitar",
        "keys",
        "synth",
        "strings",
        "texture",
        "fx",
    ]
    for group in candidates:
        group_text = group.replace("_", " ")
        if f"expected {group_text}" in text or f"expected group {group_text}" in text:
            return group
        if f"strong expected {group_text}" in text:
            return group
        if f"weak expected {group_text}" in text:
            return group
        if f"dirty expected {group_text}" in text:
            return group
        if f"dirty ambiguous expected {group_text}" in text:
            return group
    return ""


def explicit_quality(text: str, *, default: str) -> str:
    """Read expected-label quality markers from source folders."""
    if any_token(text, "strong expected", "golden expected", "locked curated", "trusted expected"):
        return STRONG_EXPECTED
    if any_token(text, "dirty expected", "dirty ambiguous expected", "ambiguous expected", "unclean expected"):
        return DIRTY_EXPECTED
    if any_token(text, "weak expected", "filename expected", "source hint expected"):
        return WEAK_EXPECTED
    return default


def expected_group_from_source_path(path: Path) -> str:
    """Infer expected diagnostic group from source-pack folders and filenames."""
    return expected_profile_from_source_path(path)["expected_group"]


def expected_group_from_normalized_text(text: str) -> str:
    """Infer a diagnostic group from normalized source path text."""
    if any_token(text, "sax", "saxophone", "woodwind", "woodwinds", "flute", "clarinet", "bassoon", "reed"):
        return "sax_reed"
    if any_token(text, "vocal", "vocals", "vox", "voice", "choir", "spoken", "breath"):
        return "voice"
    if any_token(text, "bass drum"):
        return "drums"
    if any_token(
        text,
        "kick",
        "snare",
        "clap",
        "hat",
        "hats",
        "drum",
        "drums",
        "drumloop",
        "percussion",
        "perc",
        "tom",
        "cymbal",
        "shaker",
        "tambourine",
        "bongo",
        "conga",
        "rim",
        "sidestick",
    ):
        return "drums"
    if any_token(text, "bass", "808", "sub"):
        return "bass"
    if any_token(text, "guitar", "gtr"):
        return "guitar"
    if any_token(text, "piano", "rhodes", "keys", "keyboards", "organ", "wurlitzer", "electric piano"):
        return "keys"
    if any_token(text, "string", "strings", "violin", "viola", "cello"):
        return "strings"
    if any_token(text, "synth", "pad", "lead", "pluck", "arp"):
        return "synth"
    if any_token(
        text,
        "drone",
        "texture",
        "atmosphere",
        "ambience",
        "ambient",
        "field recording",
        "noise",
        "static",
        "wind",
        "ocean",
    ):
        return "texture"
    if any_token(
        text,
        "riser",
        "downlifter",
        "drop",
        "impact",
        "whoosh",
        "sweep",
        "fx",
        "glitch",
        "stutter",
        "alarm",
        "siren",
        "reverse",
        "build",
    ):
        return "fx"
    return "unknown"


def split_fx_pack_expected_group(text: str) -> str:
    """Split dirty FX-pack labels into more useful diagnostic groups."""
    instrument_count = sum(
        1
        for family_tokens in (
            ("sax", "woodwind", "brass", "horn", "flute", "clarinet"),
            ("guitar", "gtr"),
            ("piano", "keys", "rhodes", "organ"),
            ("synth", "pad", "lead", "pluck", "arp"),
            ("string", "strings", "violin", "cello"),
            ("bass", "808", "sub"),
            ("vocal", "voice", "vox", "choir"),
        )
        if any_token(text, *family_tokens)
    )
    loopish = any_token(
        text, "loop", "loops", "bpm", "melody", "melodic", "chord", "progression", "phrase", "full mix", "fullmix"
    )
    concrete_fx = any_token(
        text,
        "riser",
        "downlifter",
        "uplifter",
        "whoosh",
        "sweep",
        "impact",
        "glitch",
        "stutter",
        "alarm",
        "siren",
        "reverse",
        "build",
        "drop",
    )
    string_or_synth = any_token(text, "synth", "pad", "lead", "arp", "string", "strings", "violin", "viola", "cello")
    melody_loop = any_token(text, "melody", "melodic", "chord", "progression", "music loop", "musical loop")
    multi = instrument_count >= 2 or any_token(
        text, "multi", "full mix", "fullmix", "construction", "songstarter", "song starter"
    )

    if concrete_fx and not instrument_count and not melody_loop:
        return "true_transition_fx"
    if multi and loopish:
        return "multi_instrument_loop"
    if string_or_synth and loopish:
        return "string_or_synth_loop"
    if melody_loop and loopish:
        return "melody_loop_inside_fx_pack"
    return "fx"


def is_dirty_or_ambiguous_source(text: str, group: str) -> bool:
    """Return True when path tokens are weak enough to avoid tuning from them."""
    if group == "unknown":
        return True
    if any_token(text, "dirty", "ambiguous", "mixed", "maybe", "unknown", "review"):
        return True
    if any_token(text, "fx aaron", "fx_aaron", "fx pack") and group not in {"true_transition_fx", "fx"}:
        return True
    return bool(
        group == "fx" and any_token(text, "loop", "bpm", "melody", "chord", "synth", "guitar", "voice", "vocal", "bass")
    )


def label_group(label: str) -> str:
    """Map an internal label/folder to a diagnostic group."""
    text = normalize_text(label)
    if not text:
        return "unknown"
    if any_token(text, "human and voice", "vocal", "voice", "vox", "spoken", "choir", "breath"):
        return "voice"
    if any_token(
        text,
        "sax",
        "saxophone",
        "woodwind",
        "woodwinds",
        "brass",
        "trumpet",
        "trombone",
        "horn",
        "flute",
        "clarinet",
        "reed",
        "bassoon",
    ):
        return "sax_reed"
    if any_token(
        text,
        "drum",
        "drums",
        "kick",
        "snare",
        "clap",
        "hat",
        "hi hat",
        "percussion",
        "tom",
        "cymbal",
        "shaker",
        "tambourine",
        "bongo",
        "conga",
    ):
        return "drums"
    if any_token(text, "bass", "808", "sub bass"):
        return "bass"
    if any_token(text, "guitar"):
        return "guitar"
    if any_token(text, "piano", "rhodes", "keys", "organ", "wurlitzer"):
        return "keys"
    if any_token(text, "string", "strings", "violin", "viola", "cello"):
        return "strings"
    if any_token(text, "synth", "pad", "lead", "chord", "pluck", "arp"):
        return "synth"
    if any_token(text, "instrument loops", "mixed musical loops"):
        return "instrument_loop_broad"
    if (
        any_token(
            text,
            "riser",
            "downlifter",
            "uplifter",
            "whoosh",
            "sweep",
            "impact",
            "glitch",
            "stutter",
            "alarm",
            "siren",
            "reverse",
            "build",
            "drop",
        )
        and top_family(label) == "FX"
    ):
        return "true_transition_fx"
    if top_family(label) == "FX":
        return "fx"
    if any_token(text, "texture", "drone", "atmosphere", "ambient", "noise", "static"):
        return "texture"
    return normalize_text(top_family(label)) or "unknown"


def compatible_tops_for_expected_group(group: str) -> list[str]:
    """Return broad top families that are safe for a diagnostic expected group."""
    if group in {
        "sax_reed",
        "bass",
        "guitar",
        "keys",
        "synth",
        "strings",
        "melody_loop_inside_fx_pack",
        "multi_instrument_loop",
        "string_or_synth_loop",
    }:
        return ["Instruments"]
    if group == "voice":
        return ["FX", "Instruments"]
    if group == "drums":
        return ["Drums"]
    if group in {"fx", "true_transition_fx"}:
        return ["FX"]
    if group == "texture":
        return ["Textures", "FX"]
    return []


def groups_are_strict_match(predicted_group: str, expected_group: str) -> bool:
    """Return True when the prediction satisfies the diagnostic identity group."""
    if predicted_group == expected_group and expected_group != "unknown":
        return True
    if expected_group == "true_transition_fx" and predicted_group == "fx":
        return True
    if expected_group == "melody_loop_inside_fx_pack":
        return predicted_group in {
            "instrument_loop_broad",
            "synth",
            "keys",
            "guitar",
            "sax_reed",
            "bass",
            "strings",
            "voice",
        }
    if expected_group == "multi_instrument_loop":
        return predicted_group in {
            "instrument_loop_broad",
            "synth",
            "keys",
            "guitar",
            "sax_reed",
            "bass",
            "strings",
            "voice",
        }
    if expected_group == "string_or_synth_loop":
        return predicted_group in {"synth", "strings", "instrument_loop_broad"}
    return False


def is_broad_safe(label: str, expected_group: str) -> bool:
    """Return True when a label is at least in a compatible broad family."""
    if not label or expected_group == "unknown":
        return False
    prediction = label_group(label)
    if groups_are_strict_match(prediction, expected_group):
        return True
    if prediction == "instrument_loop_broad" and expected_group in {
        "sax_reed",
        "bass",
        "guitar",
        "keys",
        "synth",
        "strings",
        "voice",
    }:
        return True
    return top_family(label) in set(compatible_tops_for_expected_group(expected_group))


def overcall_name(predicted_group: str, expected_group: str) -> str:
    """Return a compact overcall label for important false-positive families."""
    if predicted_group in {"unknown", expected_group}:
        return ""
    if groups_are_strict_match(predicted_group, expected_group):
        return ""
    if predicted_group == "instrument_loop_broad" and expected_group in {
        "sax_reed",
        "bass",
        "guitar",
        "keys",
        "synth",
        "strings",
        "voice",
    }:
        return "broad_instrument"
    if predicted_group in {"voice", "guitar", "fx", "true_transition_fx", "drums"}:
        return f"{predicted_group}_overcall"
    return "other_overcall"


def count_overcall(rows: list[dict[str, str]], lane: str, predicted_group: str, expected_group: str) -> int:
    """Count one type of group overcall."""
    if groups_are_strict_match(predicted_group, expected_group):
        return 0
    return sum(1 for row in rows if row.get(f"{lane}_top1_group") == predicted_group)


def lane_is_dead(row: dict[str, str], lane: str) -> bool:
    """Return True when a diagnostic lane appears empty or unavailable."""
    if lane not in BRAIN_LANES:
        return False
    top1 = row.get(f"{lane}_top1", "")
    top5 = row.get(f"{lane}_top5_json", "")
    return not top1 or top5 in {"[]", ""}


def lane_recommendation(
    *,
    total: int,
    strict: int,
    broad: int,
    voice_overcalls: int,
    guitar_overcalls: int,
    fx_overcalls: int,
    drum_overcalls: int,
    dead_lane_count: int,
) -> str:
    """Return a diagnostic-only lane competence recommendation."""
    if total <= 0:
        return "no_data"
    if dead_lane_count >= total:
        return "dead_or_missing_lane_diagnostic_only"
    strict_rate = strict / total
    broad_rate = broad / total
    if strict_rate >= 0.75:
        return "good_for_deep_identity_evidence"
    if broad_rate >= 0.75:
        return "good_for_broad_family_diagnostic_only_for_deep_identity"
    if voice_overcalls / total >= 0.25:
        return "watch_voice_overcall_diagnostic_only"
    if guitar_overcalls / total >= 0.25:
        return "watch_guitar_overcall_diagnostic_only"
    if fx_overcalls / total >= 0.25:
        return "watch_fx_overcall_diagnostic_only"
    if drum_overcalls / total >= 0.25:
        return "watch_drum_overcall_diagnostic_only"
    return "needs_more_validation_data"


def shape_value(result: SortFileResult, key: str) -> str:
    """Return one shape-vote value as text."""
    evidence = result.facts.evidence if isinstance(result.facts.evidence, dict) else {}
    shape = evidence.get("shape_vote", {})
    if not isinstance(shape, dict):
        return ""
    return str(shape.get(key, ""))


def top_family(label: str) -> str:
    """Return the top folder for an internal label/folder."""
    parts = [part for part in str(label or "").replace("\\", "/").split("/") if part]
    return parts[0] if parts else ""


def normalize_text(value: Any) -> str:
    """Normalize text for diagnostic grouping."""
    return str(value or "").replace("\\", "/").replace("_", " ").replace("-", " ").lower()


def any_token(text: str, *needles: str) -> bool:
    """Return True when any normalized fragment appears in text."""
    return any(needle.lower() in text for needle in needles)


def bool_text(value: bool) -> str:
    """Return stable CSV boolean text."""
    return "True" if value else "False"


def ratio_text(count: int, total: int) -> str:
    """Return a rounded ratio string."""
    if total <= 0:
        return "0.000"
    return f"{count / total:.3f}"


def most_common(values: list[str]) -> str:
    """Return the most common non-empty value."""
    counter = Counter(value for value in values if value)
    if not counter:
        return ""
    return counter.most_common(1)[0][0]
