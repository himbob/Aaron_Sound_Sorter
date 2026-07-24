#!/usr/bin/env python3
"""Build a human-only audio review queue for neural category gaps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.dataset import AUDIO_SUFFIXES  # noqa: E402
from aaron_sound_sorter.neural_audio.gui_training import configured_clap_trainer  # noqa: E402
from aaron_sound_sorter.neural_audio.hashing import sha256_file  # noqa: E402
from aaron_sound_sorter.neural_audio.prototype_index import PrototypeIndex  # noqa: E402

MINIMUM_GENERALIZATION_EXAMPLES = 3
SAFE_CURATION_STATUSES = frozenset(
    {
        "supported",
        "limited_support",
        "prototype_only_singleton",
        "possible_label_outlier",
    }
)
BOUNDARY_LABELS = frozenset(
    {
        "Drums/Snares/Acoustic Snare/One Shots",
        "FX/Human and Voice FX/Altered Voice/Long FX",
        "Instruments/Mixed Musical Loops/Multi Instrument/Loops",
        "Instruments/Synths/Synth Pad/Loops",
        "Instruments/Voice/Vocal Loops/Loops",
        "Instruments/Woodwinds/Saxophone/Loops",
    }
)
DISCOVERY_TERMS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "Drums/Claps Snaps Slaps/Generic Clap/One Shots": (("clap", "snap", "slap"), ("one shot",)),
    "Drums/Drum Loops/Full Drum Loops/Loops": (("drum", "beat"), ("loop",)),
    "Drums/Drum Loops/Loops": (("drum", "beat", "percussion"), ("loop",)),
    "Drums/Kick Drums/Generic Kick/One Shots": (("kick",), ("one shot",)),
    "Drums/Rims and Sticks/Generic Rim or Stick/One Shots": (
        ("rim", "stick", "cross"),
        ("one shot",),
    ),
    "Drums/Rims and Sticks/Rimshot/One Shots": (("rimshot", "rim shot"), ("one shot",)),
    "Drums/Rims and Sticks/Sidestick/One Shots": (
        ("sidestick", "side stick", "cross stick", "cross"),
        ("one shot",),
    ),
    "Drums/Snares/Acoustic Snare/One Shots": (("snare",), ("one shot",)),
    "FX/Designed Noise FX/Siren/Long FX": (("siren", "police"), ()),
    "FX/Human and Voice FX/Altered Voice/Long FX": (
        ("vocal", "voice", "vox", "acapella"),
        ("fx", "wet", "processed", "glitch", "altered"),
    ),
    "FX/Human and Voice FX/Altered Voice/One Shots": (
        ("vocal", "voice", "vox"),
        ("one shot", "fx", "processed", "glitch"),
    ),
    "FX/Human and Voice FX/Spoken Voice/Long FX": (
        ("spoken", "speech", "rap", "vocal", "voice", "acapella"),
        ("loop", "long", "phrase"),
    ),
    "FX/Human and Voice FX/Spoken Voice/One Shots": (
        ("spoken", "speech", "word", "shout", "hey", "vocal", "voice"),
        ("one shot", "phrase"),
    ),
    "FX/Structural and Transitional FX/Drops and Downlifters/Generic Downlifter/Long FX": (
        ("downlifter", "downer", "fall", "drop"),
        ("fx",),
    ),
    "FX/Structural and Transitional FX/Reverses and Tails/Generic Reverse/One Shots": (
        ("reverse", "reversed", "rvrs"),
        ("one shot", "fx"),
    ),
    "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX": (
        ("riser", "rise", "build", "uplifter"),
        ("fx",),
    ),
    "FX/Structural and Transitional FX/Risers and Builds/Long Riser/Long FX": (
        ("riser", "rise", "build", "uplifter"),
        ("long", "loop", "fx"),
    ),
    "Instruments/Bass/808 Bass/Loops": (("808",), ("bass", "loop")),
    "Instruments/Bass/Electric Bass/Loops": (
        ("electric bass", "bass guitar", "bass"),
        ("loop",),
    ),
    "Instruments/Bass/Sub Bass/Loops": (("sub bass", "subbass", "sub"), ("bass", "loop")),
    "Instruments/Guitar/Acoustic Guitar/Loops": (("acoustic guitar", "guitar"), ("loop",)),
    "Instruments/Guitar/Guitar Chords/One Shots": (("guitar", "gtr"), ("chord", "one shot")),
    "Instruments/Guitar/Guitar Loops/Loops": (("guitar", "gtr"), ("loop",)),
    "Instruments/Keys/Electric Piano/Loops": (
        ("electric piano", "rhodes", "wurli", "wurlitzer"),
        ("loop",),
    ),
    "Instruments/Keys/Keys Loops/Loops": (("keys", "keyboard", "piano"), ("loop",)),
    "Instruments/Keys/Piano/Loops": (("piano",), ("loop",)),
    "Instruments/Mallets and Bells/Bells and Mallets/Loops": (
        ("bell", "mallet", "vibraphone", "marimba", "glock"),
        ("loop",),
    ),
    "Instruments/Mixed Musical Loops/Multi Instrument/Loops": (
        ("songstarter", "multi instrument", "full melody", "melody"),
        ("loop",),
    ),
    "Instruments/Plucked Strings/Koto/Loops": (("koto",), ("loop",)),
    "Instruments/Strings/String Loops/Loops": (
        ("strings", "violin", "viola", "cello"),
        ("loop",),
    ),
    "Instruments/Synths/Synth Arp/Loops": (("arp", "arpeggio"), ("synth", "loop")),
    "Instruments/Synths/Synth Lead/Loops": (("synth lead", "lead"), ("synth", "loop")),
    "Instruments/Synths/Synth Loops/Loops": (("synth",), ("loop",)),
    "Instruments/Synths/Synth Pad/Loops": (("pad",), ("synth", "loop")),
    "Instruments/Voice/Phrase/One Shots": (
        ("vocal", "voice", "vox", "shout", "phrase"),
        ("one shot",),
    ),
    "Instruments/Voice/Vocal Loops/Loops": (
        ("vocal", "voice", "vox", "acapella", "rap"),
        ("loop",),
    ),
    "Instruments/Woodwinds/Saxophone/Loops": (("sax", "saxophone"), ("loop",)),
}


@dataclass(frozen=True)
class CandidateSource:
    """One name-discovered audio candidate awaiting content inspection."""

    path: Path
    review_hint: str
    discovery_source: str
    discovery_score: float
    discovery_reason: str


@dataclass(frozen=True)
class ScoredCandidate:
    """One candidate ranked by waveform and neural evidence."""

    path: Path
    file_sha256: str
    review_hint: str
    discovery_source: str
    discovery_score: float
    discovery_reason: str
    duration_sec: float
    target_similarity: float
    predicted_label: str
    second_label: str
    margin: float
    known_distribution: bool
    rank_score: float


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for the review-set workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sample-library-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--candidates-per-gap", type=int, default=4)
    parser.add_argument("--boundary-candidates", type=int, default=4)
    parser.add_argument("--name-pool-per-label", type=int, default=14)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Select and optionally copy a content-ranked human review queue."""
    args = build_parser().parse_args(argv)
    if args.candidates_per_gap < 1 or args.boundary_candidates < 1:
        raise ValueError("candidate counts must be positive")
    if not args.dry_run and args.destination is None:
        raise ValueError("--destination is required unless --dry-run is used")

    project_root = args.project_root.expanduser().resolve()
    sample_library_root = args.sample_library_root.expanduser().resolve()
    trainer = configured_clap_trainer(project_root)
    index_path = _active_index_path(project_root, trainer.pointer_path)
    index = PrototypeIndex.load(index_path)
    active_counts = dict(index.metadata.label_example_counts)
    target_counts = _target_review_counts(
        active_counts,
        candidates_per_gap=args.candidates_per_gap,
        boundary_candidates=args.boundary_candidates,
    )
    active_hashes = set(index.metadata.training_hashes)
    source_pool = _candidate_source_pool(
        project_root=project_root,
        sample_library_root=sample_library_root,
        target_labels=set(target_counts),
        active_hashes=active_hashes,
        maximum_per_label=args.name_pool_per_label,
    )
    scored = _score_candidates(
        source_pool,
        trainer=trainer,
        index=index,
        active_hashes=active_hashes,
    )
    selected = _select_candidates(scored, target_counts)

    copied_rows: list[dict[str, Any]] = []
    destination = args.destination.expanduser().resolve() if args.destination else None
    if not args.dry_run:
        assert destination is not None
        copied_rows = _copy_review_set(
            destination,
            selected,
            active_counts=active_counts,
        )
    summary = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "built",
        "destination": str(destination) if destination else "",
        "active_index_path": str(index_path),
        "selected_count": len(selected),
        "project_candidate_count": sum(row.discovery_source == "project_curated_review" for row in selected),
        "sample_library_candidate_count": sum(row.discovery_source == "sample_library_name_search" for row in selected),
        "target_counts": target_counts,
        "selected_by_hint": {label: sum(row.review_hint == label for row in selected) for label in target_counts},
        "unfilled_hints": {
            label: count - sum(row.review_hint == label for row in selected)
            for label, count in target_counts.items()
            if sum(row.review_hint == label for row in selected) < count
        },
        "source_name_policy": (
            "names and paths discover review candidates only; they are never approved labels or runtime model evidence"
        ),
        "training_policy": "no candidate is trained until a human approves its GUI category",
        "candidates": copied_rows if copied_rows else [_candidate_payload(row) for row in selected],
    }
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "candidates"}, indent=2))
    return 0


def _active_index_path(project_root: Path, pointer_path: Path) -> Path:
    value = pointer_path.read_text(encoding="utf-8").strip()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _target_review_counts(
    active_counts: dict[str, int],
    *,
    candidates_per_gap: int,
    boundary_candidates: int,
) -> dict[str, int]:
    targets: dict[str, int] = {}
    for label, count in active_counts.items():
        if count < MINIMUM_GENERALIZATION_EXAMPLES:
            deficit = MINIMUM_GENERALIZATION_EXAMPLES - count
            targets[label] = max(candidates_per_gap, deficit * 2)
        elif label in BOUNDARY_LABELS:
            targets[label] = boundary_candidates
    return dict(sorted(targets.items()))


def _candidate_source_pool(
    *,
    project_root: Path,
    sample_library_root: Path,
    target_labels: set[str],
    active_hashes: set[str],
    maximum_per_label: int,
) -> list[CandidateSource]:
    candidates_by_label: dict[str, list[CandidateSource]] = defaultdict(list)
    provenance_path = project_root / "neural_artifacts/20260724_real_curation/corpus_provenance.csv"
    if provenance_path.is_file():
        with provenance_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                label = str(row.get("intended_label", "")).strip()
                if label not in target_labels:
                    continue
                if str(row.get("contamination_status", "")) not in SAFE_CURATION_STATUSES:
                    continue
                digest = str(row.get("file_sha256", "")).strip()
                if digest in active_hashes:
                    continue
                path = _project_audio_path(project_root, str(row.get("audio_path", "")))
                if path is None or not path.is_file():
                    continue
                candidates_by_label[label].append(
                    CandidateSource(
                        path=path,
                        review_hint=label,
                        discovery_source="project_curated_review",
                        discovery_score=8.0,
                        discovery_reason=("historical curated slot with no known cross-label or duplicate conflict"),
                    )
                )

    destination_name = "aaron_review_me"
    for path in sample_library_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        if destination_name in {part.lower() for part in path.parts}:
            continue
        relative_text = _normalized_discovery_text(path.relative_to(sample_library_root))
        for label in target_labels:
            score = _discovery_score(label, relative_text)
            if score <= 0.0:
                continue
            candidates_by_label[label].append(
                CandidateSource(
                    path=path.resolve(),
                    review_hint=label,
                    discovery_source="sample_library_name_search",
                    discovery_score=score,
                    discovery_reason="filename/folder metadata matched a missing-label search",
                )
            )

    pooled: list[CandidateSource] = []
    for label in sorted(target_labels):
        ordered = sorted(
            candidates_by_label[label],
            key=lambda row: (
                -row.discovery_score,
                row.discovery_source != "project_curated_review",
                str(row.path).lower(),
            ),
        )
        pooled.extend(_diverse_name_pool(ordered, maximum_per_label))
    return pooled


def _diverse_name_pool(
    candidates: list[CandidateSource],
    maximum: int,
) -> list[CandidateSource]:
    selected: list[CandidateSource] = []
    seen_parent_groups: set[str] = set()
    for candidate in candidates:
        parent_group = str(candidate.path.parent.parent).lower()
        if parent_group in seen_parent_groups and len(selected) < max(2, maximum // 2):
            continue
        selected.append(candidate)
        seen_parent_groups.add(parent_group)
        if len(selected) >= maximum:
            break
    if len(selected) < maximum:
        selected_paths = {row.path for row in selected}
        selected.extend(row for row in candidates if row.path not in selected_paths)
    return selected[:maximum]


def _normalized_discovery_text(path: Path) -> str:
    text = str(path).lower()
    for character in "_-/\\.":
        text = text.replace(character, " ")
    return " ".join(text.split())


def _discovery_score(label: str, normalized_path_text: str) -> float:
    primary_terms, context_terms = DISCOVERY_TERMS.get(label, ((), ()))
    primary_hits = sum(term in normalized_path_text for term in primary_terms)
    if not primary_hits:
        return 0.0
    context_hits = sum(term in normalized_path_text for term in context_terms)
    score = 4.0 + min(primary_hits, 3) + 0.75 * min(context_hits, 3)
    if label.endswith("/One Shots") and "loop" in normalized_path_text:
        score -= 2.0
    if label.endswith("/Loops") and "one shot" in normalized_path_text:
        score -= 2.0
    return max(score, 0.0)


def _project_audio_path(project_root: Path, value: str) -> Path | None:
    if not value.startswith("project://"):
        return None
    candidate = (project_root / value.removeprefix("project://")).resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError:
        return None
    return candidate


def _score_candidates(
    candidates: list[CandidateSource],
    *,
    trainer: Any,
    index: PrototypeIndex,
    active_hashes: set[str],
) -> list[ScoredCandidate]:
    scored: list[ScoredCandidate] = []
    seen_pairs: set[tuple[str, str]] = set()
    for position, candidate in enumerate(candidates, start=1):
        try:
            digest = sha256_file(candidate.path)
            if digest in active_hashes or (candidate.review_hint, digest) in seen_pairs:
                continue
            info = sf.info(candidate.path)
            duration_sec = float(info.duration)
            if not math.isfinite(duration_sec) or not 0.05 <= duration_sec <= 180.0:
                continue
            record = trainer.cache.get_or_compute(candidate.path, trainer.provider)
            prediction = index.predict(record)
            target_similarity = index.label_similarities(record).get(candidate.review_hint, -1.0)
        except (OSError, RuntimeError, ValueError):
            continue
        seen_pairs.add((candidate.review_hint, digest))
        prediction_bonus = 0.12 if prediction.predicted_label == candidate.review_hint else 0.0
        second_bonus = 0.04 if prediction.second_label == candidate.review_hint else 0.0
        duration_bonus = _duration_role_bonus(candidate.review_hint, duration_sec)
        rank_score = (
            target_similarity + prediction_bonus + second_bonus + duration_bonus + 0.015 * candidate.discovery_score
        )
        scored.append(
            ScoredCandidate(
                path=candidate.path,
                file_sha256=digest,
                review_hint=candidate.review_hint,
                discovery_source=candidate.discovery_source,
                discovery_score=candidate.discovery_score,
                discovery_reason=candidate.discovery_reason,
                duration_sec=duration_sec,
                target_similarity=target_similarity,
                predicted_label=prediction.predicted_label,
                second_label=prediction.second_label,
                margin=prediction.margin,
                known_distribution=prediction.known_distribution,
                rank_score=rank_score,
            )
        )
        if position % 25 == 0:
            print(f"scored {position}/{len(candidates)} candidate paths", flush=True)
    return scored


def _duration_role_bonus(label: str, duration_sec: float) -> float:
    if label.endswith("/One Shots"):
        return 0.05 if duration_sec <= 5.0 else -0.08
    if label.endswith("/Loops"):
        return 0.05 if duration_sec >= 1.0 else -0.08
    if label.endswith("/Long FX"):
        return 0.05 if duration_sec >= 1.0 else -0.08
    return 0.0


def _select_candidates(
    scored: list[ScoredCandidate],
    target_counts: dict[str, int],
) -> list[ScoredCandidate]:
    by_label: dict[str, list[ScoredCandidate]] = defaultdict(list)
    for candidate in scored:
        by_label[candidate.review_hint].append(candidate)
    selected: list[ScoredCandidate] = []
    selected_hashes: set[str] = set()
    for label in sorted(target_counts, key=lambda value: (-target_counts[value], value)):
        ordered = sorted(
            by_label[label],
            key=lambda row: (
                -row.rank_score,
                -row.target_similarity,
                row.file_sha256,
            ),
        )
        for candidate in ordered:
            if candidate.file_sha256 in selected_hashes:
                continue
            selected.append(candidate)
            selected_hashes.add(candidate.file_sha256)
            if sum(row.review_hint == label for row in selected) >= target_counts[label]:
                break
    return selected


def _copy_review_set(
    destination: Path,
    selected: list[ScoredCandidate],
    *,
    active_counts: dict[str, int],
) -> list[dict[str, Any]]:
    if destination.exists():
        raise FileExistsError(f"review destination already exists: {destination}")
    destination.mkdir(parents=True)
    payloads: list[dict[str, Any]] = []
    for position, candidate in enumerate(selected, start=1):
        safe_name = _safe_filename(candidate.path.stem)
        copied_name = f"candidate_{position:03d}_{candidate.file_sha256[:8]}_{safe_name}{candidate.path.suffix.lower()}"
        copied_path = destination / copied_name
        shutil.copy2(candidate.path, copied_path)
        payload = _candidate_payload(candidate)
        payload.update(
            {
                "candidate_file": copied_name,
                "active_label_examples_before_review": active_counts.get(candidate.review_hint, 0),
                "examples_needed_for_generalization": max(
                    0,
                    MINIMUM_GENERALIZATION_EXAMPLES - active_counts.get(candidate.review_hint, 0),
                ),
            }
        )
        payloads.append(payload)
    _write_manifest(destination / "REVIEW_MANIFEST.csv", payloads)
    (destination / "README.txt").write_text(
        "Aaron_Review_Me\n\n"
        "1. Listen to every audio file.\n"
        "2. Use review_hint only as a question, never as truth.\n"
        "3. Correct the category in the GUI.\n"
        "4. Press Train only after the category is right.\n\n"
        "Filenames and folders found candidates. The model never uses them as audio evidence.\n",
        encoding="utf-8",
    )
    return payloads


def _candidate_payload(candidate: ScoredCandidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["path"] = str(candidate.path)
    return payload


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _safe_filename(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in " ._-" else "_" for character in value)
    return "_".join(cleaned.split())[:120] or "audio"


if __name__ == "__main__":
    raise SystemExit(main())
