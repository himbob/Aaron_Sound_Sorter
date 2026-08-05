#!/usr/bin/env python3
"""Build and audit broad named-sample coverage panels.

This is an offline QA/training tool, not production classifier logic. It may
inspect source filenames and source folder names to select samples that appear
human-named and category-specific. Those names are used only as post-sort test
identifiers and expected-category hints. Runtime sorter code must remain
source-name blind.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

csv.field_size_limit(sys.maxsize)

AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac", ".ogg"}
DEFAULT_SAMPLE_ROOT = Path("/path/to/sample-library")
DEFAULT_PROJECT_ROOT = Path("/path/to/Aaron_Sound_Sorter")
DEFAULT_PANEL_SIZE = 5000
DEFAULT_PER_CATEGORY_LIMIT = 120
DEFAULT_SEED = 20260721

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    ".venv_phase4",
    "__MACOSX",
    "__pycache__",
    "_reports",
    "_TO_REVIEW",
    "Aaron_Sorted_Sounds",
    "node_modules",
}

SKIP_PATH_PHRASES = {
    "aaron_sound_sorter",
    "backup",
    "backups",
    "combined_training",
    "regression_audio",
    "report",
    "reports",
    "sorted samples",
    "test_runner",
    "trash",
    "upload_back",
}

GLOBAL_REJECT_TERMS = {
    "._",
    "ableton project",
    "demo",
    "evaluation",
    "eval",
    "preview",
    "render test",
    "train",
    "test tone",
    "val",
    "validation",
}


@dataclass(frozen=True)
class PanelCategory:
    """Offline category selector for a broad coverage panel.

    Attributes:
        category_id: Stable identifier used in reports.
        accepted_prefixes: Sorter output prefixes considered acceptable.
        include_terms: Strong name or folder phrases that imply this category.
        reject_terms: Phrases that make the match too ambiguous.
        minimum_score: Minimum offline name-confidence score to accept.
        per_category_limit: Maximum selected examples for this category.
        per_folder_limit: Diversity cap for direct parent folders.
    """

    category_id: str
    accepted_prefixes: tuple[str, ...]
    include_terms: tuple[str, ...]
    reject_terms: tuple[str, ...] = ()
    minimum_score: int = 5
    per_category_limit: int = DEFAULT_PER_CATEGORY_LIMIT
    per_folder_limit: int = 8


@dataclass(frozen=True)
class CandidateSample:
    """A named audio file that matched one offline panel category."""

    source_path: Path
    category_id: str
    accepted_prefixes: tuple[str, ...]
    score: int
    direct_folder: str
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class SelectedPanelSample:
    """A final panel entry with a stable panel-local filename."""

    panel_filename: str
    source_path: Path
    category_id: str
    accepted_prefixes: tuple[str, ...]
    score: int
    direct_folder: str
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class AuditCaseResult:
    """One expected named-panel case compared against sorter output."""

    panel_filename: str
    category_id: str
    expected_prefixes: tuple[str, ...]
    actual_path: str
    actual_top: str
    status: str
    source_path: str
    consensus_status: str
    final_claim_source: str
    brain_vote: str
    physics_vote: str
    shape_vote: str


@lru_cache(maxsize=131_072)
def normalize_text(value: str) -> str:
    """Return lowercase searchable text with camel-case and separators split."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_text(value: str) -> str:
    """Return normalized text without spaces for joined-token matching."""
    return normalize_text(value).replace(" ", "")


@lru_cache(maxsize=2048)
def normalized_search_term(term: str) -> str:
    """Return a cached normalized category/search term."""
    return normalize_text(term)


@lru_cache(maxsize=2048)
def word_boundary_pattern(term: str) -> re.Pattern[str]:
    """Return a cached word-boundary regex for one normalized term."""
    return re.compile(rf"(?:^| ){re.escape(term)}(?: |$)")


def contains_term(search_text: str, compact_search_text: str, term: str) -> bool:
    """Return whether a normalized text contains a term or phrase."""
    normalized_term = normalized_search_term(term)
    if not normalized_term:
        return False
    if " " in normalized_term:
        return normalized_term in search_text or normalized_term.replace(" ", "") in compact_search_text
    return bool(word_boundary_pattern(normalized_term).search(search_text))


def term_matches_for_path(path: Path, terms: Sequence[str]) -> tuple[str, ...]:
    """Return terms found in a source path's normalized stem and parent text."""
    text = normalize_text(f"{path.stem} {' '.join(path.parts[-5:-1])}")
    compact = text.replace(" ", "")
    return tuple(term for term in terms if contains_term(text, compact, term))


def path_has_any_term(path: Path, terms: Iterable[str]) -> bool:
    """Return true when any normalized term appears in the source path."""
    text = normalize_text(str(path))
    compact = text.replace(" ", "")
    return any(contains_term(text, compact, term) for term in terms)


def score_path_for_category(path: Path, category: PanelCategory) -> tuple[int, tuple[str, ...]]:
    """Score a source path against one offline category definition."""
    if path_has_any_term(path, GLOBAL_REJECT_TERMS) or path_has_any_term(path, category.reject_terms):
        return 0, ()

    stem_text = normalize_text(path.stem)
    parent_text = normalize_text(" ".join(path.parts[-5:-1]))
    combined_text = f"{stem_text} {parent_text}"
    stem_compact = stem_text.replace(" ", "")
    parent_compact = parent_text.replace(" ", "")
    combined_compact = combined_text.replace(" ", "")

    matched_terms: list[str] = []
    score = 0
    for term in category.include_terms:
        if contains_term(stem_text, stem_compact, term):
            matched_terms.append(term)
            score += 8
        elif contains_term(parent_text, parent_compact, term):
            matched_terms.append(term)
            score += 3
        elif contains_term(combined_text, combined_compact, term):
            matched_terms.append(term)
            score += 1

    if not matched_terms:
        return 0, ()
    if path.suffix.lower() in {".wav", ".aif", ".aiff"}:
        score += 1
    if contains_term(stem_text, stem_compact, "loop") or contains_term(stem_text, stem_compact, "bpm"):
        if any(prefix.endswith("/Loops") or "Loops" in prefix for prefix in category.accepted_prefixes):
            score += 2
    if contains_term(stem_text, stem_compact, "one shot") or contains_term(stem_text, stem_compact, "oneshot"):
        if any(prefix.endswith("/One Shots") or "One Shots" in prefix for prefix in category.accepted_prefixes):
            score += 2
    return score, tuple(sorted(set(matched_terms)))


def panel_categories(per_category_limit: int) -> tuple[PanelCategory, ...]:
    """Return broad offline selectors covering drums, instruments, and FX."""
    limit = max(1, per_category_limit)
    return tuple(
        PanelCategory(category_id, accepted, include, reject, per_category_limit=limit)
        for category_id, accepted, include, reject in [
            (
                "drums_kick_one_shots",
                ("Drums/Kick Drums",),
                ("kick", "kik"),
                ("bass loop", "bassline", "drum loop", "drumloop", "breakbeat"),
            ),
            ("drums_snare_one_shots", ("Drums/Snares",), ("snare",), ("drum loop", "loop")),
            ("drums_clap_snap_one_shots", ("Drums/Claps Snaps Slaps",), ("clap", "snap"), ("loop",)),
            (
                "drums_hat_one_shots",
                ("Drums/Hi Hats",),
                ("hat", "hihat", "hi hat", "closed hat", "open hat"),
                ("loop", "synth hat"),
            ),
            ("drums_cymbal_one_shots", ("Drums/Cymbals",), ("cymbal", "crash", "ride", "splash"), ("loop",)),
            ("drums_tom_one_shots", ("Drums/Toms",), ("tom", "floor tom", "low tom"), ("loop",)),
            (
                "drums_rim_stick_one_shots",
                ("Drums/Rims and Sticks",),
                ("rimshot", "rim shot", "sidestick", "side stick", "clave", "wood block"),
                ("loop",),
            ),
            (
                "drums_shaker_tambourine",
                ("Drums/Percussion/Shakers and Tambourines", "Drums/Percussion"),
                ("shaker", "tambourine"),
                (),
            ),
            (
                "drums_world_hand_percussion",
                ("Drums/World Percussion", "Drums/Percussion", "Drums/Toms"),
                ("bongo", "conga", "tabla", "djembe", "timbale"),
                (),
            ),
            (
                "drums_percussion_loops",
                ("Drums/Drum Loops", "Drums/Percussion"),
                ("percussion loop", "perc loop", "drum loop", "drumloop", "breakbeat", "break"),
                ("bass break", "guitar break", "piano break"),
            ),
            (
                "instruments_bass",
                ("Instruments/Bass",),
                ("bass", "bassline", "sub bass", "808 bass", "electric bass", "upright bass"),
                ("kick", "boom", "sub hit", "impact", "drum"),
            ),
            (
                "instruments_guitar",
                ("Instruments/Guitar",),
                ("guitar", "gtr", "acoustic guitar", "electric guitar", "nylon guitar"),
                ("bass guitar",),
            ),
            (
                "instruments_plucked_strings",
                ("Instruments/Strings/Plucked", "Instruments/Guitar", "Instruments/Instrument Loops"),
                ("koto", "harp", "sitar", "banjo", "mandolin", "oud", "ukulele"),
                (),
            ),
            ("instruments_piano", ("Instruments/Keys/Piano", "Instruments/Keys"), ("piano",), ("electric piano",)),
            (
                "instruments_electric_piano",
                ("Instruments/Keys/Electric Piano", "Instruments/Keys/Rhodes", "Instruments/Keys"),
                ("electric piano", "rhodes", "wurli", "wurlitzer"),
                (),
            ),
            ("instruments_organ", ("Instruments/Keys/Organ", "Instruments/Keys"), ("organ", "hammond"), ()),
            (
                "instruments_synths",
                ("Instruments/Synths",),
                ("synth", "synth lead", "synth pad", "synth chord", "arp", "arpeggio", "sequence"),
                ("riser", "downlifter", "whoosh", "sweep", "fx"),
            ),
            (
                "instruments_strings",
                ("Instruments/Strings",),
                ("violin", "viola", "cello", "strings", "string section"),
                (),
            ),
            (
                "instruments_saxophone",
                ("Instruments/Woodwinds/Saxophone", "Instruments/Brass and Woodwinds", "Instruments/Woodwinds"),
                ("sax", "saxophone", "tenor sax", "alto sax"),
                (),
            ),
            (
                "instruments_other_woodwinds",
                ("Instruments/Woodwinds", "Instruments/Brass and Woodwinds"),
                ("flute", "clarinet", "bassoon", "oboe", "pan flute", "duduk"),
                (),
            ),
            (
                "instruments_brass",
                ("Instruments/Brass", "Instruments/Brass and Woodwinds"),
                ("trumpet", "trombone", "horn", "brass"),
                (),
            ),
            (
                "instruments_voice",
                ("Instruments/Voice",),
                ("vocal", "vocals", "vox", "voice", "rap", "spoken", "dialogue", "phrase"),
                ("applause", "crowd", "breath", "mouth"),
            ),
            (
                "instruments_mallets_bells",
                ("Instruments/Mallets and Bells", "Drums/Percussion/Bells and Metallic Percussion"),
                ("marimba", "xylophone", "vibraphone", "glockenspiel", "music box", "celesta", "bell"),
                ("cowbell", "cymbal"),
            ),
            (
                "fx_risers_builds",
                ("FX/Structural and Transitional FX/Risers and Builds",),
                ("riser", "build", "uplifter", "up lifter"),
                ("vocal riser", "sax riser"),
            ),
            (
                "fx_drops_downlifters",
                ("FX/Structural and Transitional FX/Drops and Downlifters",),
                ("drop", "downlifter", "down lifter", "falling"),
                (),
            ),
            (
                "fx_whoosh_sweep",
                ("FX/Structural and Transitional FX/Sweeps and Whooshes",),
                ("whoosh", "woosh", "sweep", "swoosh", "swish"),
                (),
            ),
            (
                "fx_reverse_tail",
                ("FX/Structural and Transitional FX/Reverses and Tails",),
                ("reverse", "tail", "swell"),
                ("vocal tail", "snare tail"),
            ),
            (
                "fx_impacts_hits",
                ("FX/Impacts and Hits",),
                ("impact", "boom", "slam", "sub hit", "hit fx"),
                ("kick", "snare", "clap"),
            ),
            (
                "fx_glitch_stutter",
                ("FX/Digital Mechanical Industrial Transport/Glitches and Stutters",),
                ("glitch", "stutter", "buffer", "chop", "chopped"),
                ("vocal chop",),
            ),
            ("fx_blip_beep_alarm", ("FX/Designed Noise FX",), ("blip", "beep", "alarm", "siren", "laser", "zap"), ()),
            (
                "fx_foley_doors_objects",
                ("FX/Everyday Foley",),
                (
                    "door",
                    "car keys",
                    "carkeys",
                    "key ring",
                    "key jingle",
                    "coins",
                    "coin",
                    "foley",
                    "latch",
                    "chain",
                    "zipper",
                ),
                ("bass", "chord", "guitar", "loop", "piano", "rhodes", "synth"),
            ),
            (
                "fx_machines",
                ("FX/Everyday Foley/Machines", "FX/Digital Mechanical Industrial Transport"),
                ("engine", "motor", "machine", "servo", "mechanical"),
                (),
            ),
            (
                "fx_water_rain_wind_fire",
                ("FX/Textures/Natural Ambience",),
                ("water", "ocean", "waves", "rain", "wind", "fire", "thunder"),
                (),
            ),
            (
                "fx_noise_static",
                ("FX/Textures/Noise and Static",),
                ("hiss", "static", "white noise", "vinyl noise", "noise"),
                ("riser",),
            ),
            (
                "fx_human_crowd_breath",
                ("FX/Human and Voice FX",),
                ("applause", "crowd", "breath", "mouth sound", "scream"),
                ("vocal loop", "rap vocal"),
            ),
            ("fx_animals", ("FX/Animals and Creatures",), ("bird", "cat", "dog", "cricket", "insect", "animal"), ()),
        ]
    )


def iter_audio_files(sample_root: Path) -> Iterable[Path]:
    """Yield candidate audio files beneath a sample root."""
    for root, dirnames, filenames in os.walk(sample_root, followlinks=False):
        root_path = Path(root)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not dirname.startswith(".")
            and dirname not in SKIP_DIR_NAMES
            and not path_has_any_term(root_path / dirname, SKIP_PATH_PHRASES)
        ]
        if path_has_any_term(root_path, SKIP_PATH_PHRASES):
            continue
        for filename in filenames:
            path = root_path / filename
            if filename.startswith(".") or filename.startswith("._"):
                continue
            if path.suffix.lower() in AUDIO_EXTENSIONS:
                yield path


def find_named_candidates(sample_root: Path, categories: Sequence[PanelCategory]) -> dict[str, list[CandidateSample]]:
    """Scan a sample library and collect high-confidence named candidates."""
    candidates_by_category: dict[str, list[CandidateSample]] = defaultdict(list)
    for source_path in iter_audio_files(sample_root):
        resolved_source = source_path.resolve()
        direct_folder = source_path.parent.name
        for category in categories:
            score, matched_terms = score_path_for_category(source_path, category)
            if score < category.minimum_score:
                continue
            candidates_by_category[category.category_id].append(
                CandidateSample(
                    source_path=resolved_source,
                    category_id=category.category_id,
                    accepted_prefixes=category.accepted_prefixes,
                    score=score,
                    direct_folder=direct_folder,
                    matched_terms=matched_terms,
                )
            )
    return dict(candidates_by_category)


def stable_random_number(seed: int, path: Path, category_id: str) -> float:
    """Return deterministic pseudo-random jitter for diverse candidate ordering."""
    digest = hashlib.sha256(f"{seed}:{category_id}:{path}".encode()).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def select_category_candidates(
    candidates: Sequence[CandidateSample],
    *,
    category: PanelCategory,
    seed: int,
) -> list[CandidateSample]:
    """Select high-scoring candidates while spreading them across folders."""
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            stable_random_number(seed, candidate.source_path, candidate.category_id),
            str(candidate.source_path).lower(),
        ),
    )
    selected: list[CandidateSample] = []
    folder_counts: Counter[str] = Counter()
    for candidate in ordered:
        if len(selected) >= category.per_category_limit:
            break
        if folder_counts[candidate.direct_folder] >= category.per_folder_limit:
            continue
        selected.append(candidate)
        folder_counts[candidate.direct_folder] += 1

    if len(selected) >= category.per_category_limit:
        return selected
    selected_paths = {candidate.source_path for candidate in selected}
    for candidate in ordered:
        if len(selected) >= category.per_category_limit:
            break
        if candidate.source_path in selected_paths:
            continue
        selected.append(candidate)
        selected_paths.add(candidate.source_path)
    return selected


def select_panel_samples(
    candidates_by_category: dict[str, list[CandidateSample]],
    categories: Sequence[PanelCategory],
    *,
    panel_size: int,
    seed: int,
) -> list[SelectedPanelSample]:
    """Round-robin category selections until the requested panel size is met."""
    category_lookup = {category.category_id: category for category in categories}
    category_pools = {
        category_id: select_category_candidates(
            candidates,
            category=category_lookup[category_id],
            seed=seed,
        )
        for category_id, candidates in candidates_by_category.items()
        if category_id in category_lookup
    }
    selected: list[SelectedPanelSample] = []
    seen_sources: set[Path] = set()
    category_offsets: Counter[str] = Counter()
    category_ids = [category.category_id for category in categories if category.category_id in category_pools]

    while len(selected) < panel_size and category_ids:
        added_this_round = False
        for category_id in category_ids:
            pool = category_pools[category_id]
            while category_offsets[category_id] < len(pool):
                candidate = pool[category_offsets[category_id]]
                category_offsets[category_id] += 1
                if candidate.source_path in seen_sources:
                    continue
                selected.append(make_selected_sample(candidate, len(selected) + 1))
                seen_sources.add(candidate.source_path)
                added_this_round = True
                break
            if len(selected) >= panel_size:
                break
        if not added_this_round:
            break
    return selected


def safe_filename_part(value: str, max_length: int = 58) -> str:
    """Return a filesystem-safe short filename component."""
    text = re.sub(r"[^A-Za-z0-9#]+", "_", value).strip("_")
    text = text or "sample"
    return text[:max_length].strip("_") or "sample"


def make_selected_sample(candidate: CandidateSample, index: int) -> SelectedPanelSample:
    """Create a panel-local filename for one selected source file."""
    digest = hashlib.sha1(str(candidate.source_path).encode("utf-8")).hexdigest()[:10]
    stem = safe_filename_part(candidate.source_path.stem)
    category = safe_filename_part(candidate.category_id, max_length=34)
    panel_filename = f"{index:05d}_{category}_{stem}_{digest}{candidate.source_path.suffix.lower()}"
    return SelectedPanelSample(
        panel_filename=panel_filename,
        source_path=candidate.source_path,
        category_id=candidate.category_id,
        accepted_prefixes=candidate.accepted_prefixes,
        score=candidate.score,
        direct_folder=candidate.direct_folder,
        matched_terms=candidate.matched_terms,
    )


def materialize_panel(samples: Sequence[SelectedPanelSample], input_panel_dir: Path, link_mode: str) -> None:
    """Place panel audio in a run-local input folder as symlinks or copies."""
    input_panel_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        destination = input_panel_dir / sample.panel_filename
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if link_mode == "copy":
            shutil.copy2(sample.source_path, destination)
        else:
            destination.symlink_to(sample.source_path)


def write_expected_csv(samples: Sequence[SelectedPanelSample], path: Path) -> None:
    """Write the expected prefixes for the selected panel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "panel_filename",
                "category_id",
                "accepted_prefixes",
                "source_path",
                "score",
                "direct_folder",
                "matched_terms",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "panel_filename": sample.panel_filename,
                    "category_id": sample.category_id,
                    "accepted_prefixes": "|".join(sample.accepted_prefixes),
                    "source_path": str(sample.source_path),
                    "score": sample.score,
                    "direct_folder": sample.direct_folder,
                    "matched_terms": "|".join(sample.matched_terms),
                }
            )


def write_expected_json(samples: Sequence[SelectedPanelSample], path: Path) -> None:
    """Write a JSON version of the expected panel metadata."""
    payload = {
        "panel_name": "named_sample_coverage",
        "description": "Offline named-sample QA panel. Names are an oracle only, never runtime evidence.",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rules": {
            "fail_on_missing_manifest_row": True,
            "names_are_test_oracle_only": True,
            "sorter_must_remain_source_name_blind": True,
        },
        "cases": [
            {
                "filename": sample.panel_filename,
                "category_id": sample.category_id,
                "accepted_folder_prefixes": list(sample.accepted_prefixes),
                "source_path": str(sample.source_path),
                "score": sample.score,
                "matched_terms": list(sample.matched_terms),
            }
            for sample in samples
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_build_summary(
    *,
    output_dir: Path,
    sample_root: Path,
    samples: Sequence[SelectedPanelSample],
    candidates_by_category: dict[str, list[CandidateSample]],
) -> None:
    """Write a human-readable panel build summary."""
    counts = Counter(sample.category_id for sample in samples)
    lines = [
        "Named Sample Coverage Panel",
        "===========================",
        "",
        f"Sample root: {sample_root}",
        f"Selected samples: {len(samples)}",
        f"Report folder: {output_dir}",
        "",
        "Category coverage:",
    ]
    for category_id in sorted({*counts, *candidates_by_category}):
        lines.append(
            f"  {category_id}: selected={counts.get(category_id, 0)} "
            f"candidates={len(candidates_by_category.get(category_id, []))}"
        )
    lines.extend(
        [
            "",
            "Next commands:",
            f"  python3 Aaron_Sound_Sorter.py sort {output_dir / 'input_panel'} {output_dir / 'sort_output'} --no-zip",
            (
                "  python3 tools/named_sample_coverage_panel.py audit "
                f"--expected {output_dir / 'named_sample_panel_expected.csv'} "
                f"--manifest {output_dir / 'sort_output' / 'Aaron_Sorted_Sounds_manifest.csv'} "
                f"--output-dir {output_dir / 'audit'}"
            ),
            "",
            "Reminder: this panel uses names only as an offline oracle. The sorter must not read these names as evidence.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "build_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_panel(args: argparse.Namespace) -> int:
    """Build a named coverage panel from a sample library."""
    sample_root = Path(args.sample_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    categories = panel_categories(int(args.per_category_limit))
    candidates_by_category = find_named_candidates(sample_root, categories)
    selected_samples = select_panel_samples(
        candidates_by_category,
        categories,
        panel_size=int(args.panel_size),
        seed=int(args.seed),
    )
    materialize_panel(selected_samples, output_dir / "input_panel", str(args.link_mode))
    write_expected_csv(selected_samples, output_dir / "named_sample_panel_expected.csv")
    write_expected_json(selected_samples, output_dir / "expected_results.json")
    write_build_summary(
        output_dir=output_dir,
        sample_root=sample_root,
        samples=selected_samples,
        candidates_by_category=candidates_by_category,
    )
    print(f"Selected samples: {len(selected_samples)}")
    print(f"Input panel: {output_dir / 'input_panel'}")
    print(f"Expected CSV: {output_dir / 'named_sample_panel_expected.csv'}")
    return 0 if selected_samples else 1


def read_expected_cases(path: Path) -> dict[str, dict[str, str]]:
    """Read expected panel rows keyed by panel filename."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["panel_filename"]: row for row in csv.DictReader(handle)}


def row_audio_filename(row: dict[str, str]) -> str:
    """Return a manifest row's most stable audio filename."""
    for column in ("case_filename", "filename", "source_filename", "original_filename"):
        value = row.get(column, "").strip()
        if value:
            return Path(value).name
    for column in ("source_path", "placed_path", "new_path"):
        value = row.get(column, "").strip()
        if value:
            return Path(value).name
    return ""


def row_final_path(row: dict[str, str]) -> str:
    """Return the final sorter folder path from a manifest row."""
    for column in ("final_label", "folder_path", "new_relative_path", "placed_path"):
        value = row.get(column, "").strip()
        if value:
            if column == "placed_path":
                parts = Path(value).parts
                if "Aaron_Sorted_Sounds" in parts:
                    index = parts.index("Aaron_Sorted_Sounds")
                    return "/".join(parts[index + 1 : -1])
            return value
    return ""


def read_manifest_rows(path: Path) -> dict[str, dict[str, str]]:
    """Read sorter manifest rows keyed by panel filename."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row_audio_filename(row): row for row in rows if row_audio_filename(row)}


def audit_case(expected: dict[str, str], row: dict[str, str] | None) -> AuditCaseResult:
    """Compare one expected panel case against one sorter manifest row."""
    expected_prefixes = tuple(prefix for prefix in expected["accepted_prefixes"].split("|") if prefix)
    if row is None:
        return AuditCaseResult(
            panel_filename=expected["panel_filename"],
            category_id=expected["category_id"],
            expected_prefixes=expected_prefixes,
            actual_path="",
            actual_top="",
            status="FAIL_MISSING_MANIFEST",
            source_path=expected.get("source_path", ""),
            consensus_status="",
            final_claim_source="",
            brain_vote="",
            physics_vote="",
            shape_vote="",
        )
    actual_path = row_final_path(row)
    actual_top = actual_path.split("/", 1)[0] if actual_path else ""
    expected_tops = {prefix.split("/", 1)[0] for prefix in expected_prefixes}
    if actual_path.startswith("_TO_REVIEW"):
        status = "FAIL_REVIEW"
    elif any(actual_path.startswith(prefix) for prefix in expected_prefixes):
        status = "PASS"
    elif actual_top not in expected_tops:
        status = "FAIL_WRONG_TOP"
    else:
        status = "FAIL_WRONG_PREFIX"
    return AuditCaseResult(
        panel_filename=expected["panel_filename"],
        category_id=expected["category_id"],
        expected_prefixes=expected_prefixes,
        actual_path=actual_path,
        actual_top=actual_top,
        status=status,
        source_path=expected.get("source_path", ""),
        consensus_status=row.get("consensus_status", ""),
        final_claim_source=row.get("final_claim_source", ""),
        brain_vote=row.get("brain_ensemble_vote_1") or row.get("brain_vote_1", ""),
        physics_vote=row.get("physics_vote_1", ""),
        shape_vote=row.get("shape_vote", ""),
    )


def write_audit_outputs(results: Sequence[AuditCaseResult], output_dir: Path) -> None:
    """Write CSV and summary outputs for an audited named coverage panel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "named_sample_panel_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "status",
                "panel_filename",
                "category_id",
                "expected_prefixes",
                "actual_path",
                "actual_top",
                "consensus_status",
                "final_claim_source",
                "brain_vote",
                "physics_vote",
                "shape_vote",
                "source_path",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "status": result.status,
                    "panel_filename": result.panel_filename,
                    "category_id": result.category_id,
                    "expected_prefixes": "|".join(result.expected_prefixes),
                    "actual_path": result.actual_path,
                    "actual_top": result.actual_top,
                    "consensus_status": result.consensus_status,
                    "final_claim_source": result.final_claim_source,
                    "brain_vote": result.brain_vote,
                    "physics_vote": result.physics_vote,
                    "shape_vote": result.shape_vote,
                    "source_path": result.source_path,
                }
            )
    status_counts = Counter(result.status for result in results)
    category_failures = Counter(result.category_id for result in results if result.status != "PASS")
    top_failures = Counter(result.actual_top for result in results if result.status.startswith("FAIL"))
    lines = [
        "Named Sample Coverage Audit",
        "===========================",
        "",
        f"Cases: {len(results)}",
        f"Passed: {status_counts.get('PASS', 0)}",
        f"Failed: {sum(count for status, count in status_counts.items() if status != 'PASS')}",
        "",
        "Statuses:",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"  {status}: {count}")
    lines.append("")
    lines.append("Failure categories:")
    for category_id, count in category_failures.most_common(30):
        lines.append(f"  {category_id}: {count}")
    lines.append("")
    lines.append("Failure actual top folders:")
    for actual_top, count in top_failures.most_common(20):
        lines.append(f"  {actual_top or '(missing)'}: {count}")
    lines.append("")
    lines.append("First failures:")
    for result in [case for case in results if case.status != "PASS"][:40]:
        lines.append(
            f"  {result.status}: {result.panel_filename} expected={list(result.expected_prefixes)} "
            f"actual={result.actual_path} shape={result.shape_vote} physics={result.physics_vote}"
        )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_panel(args: argparse.Namespace) -> int:
    """Audit a sorter manifest against expected named-panel prefixes."""
    expected_cases = read_expected_cases(Path(args.expected).expanduser())
    manifest_rows = read_manifest_rows(Path(args.manifest).expanduser())
    results = [audit_case(expected, manifest_rows.get(filename)) for filename, expected in expected_cases.items()]
    write_audit_outputs(results, Path(args.output_dir).expanduser())
    failures = [result for result in results if result.status != "PASS"]
    print(f"Audited cases: {len(results)}")
    print(f"Failures: {len(failures)}")
    print(f"Audit output: {Path(args.output_dir).expanduser()}")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for panel build and audit subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser("build", help="Build a broad named-sample coverage panel.")
    build.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--panel-size", type=int, default=DEFAULT_PANEL_SIZE)
    build.add_argument("--per-category-limit", type=int, default=DEFAULT_PER_CATEGORY_LIMIT)
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    build.set_defaults(func=build_panel)

    audit = subcommands.add_parser("audit", help="Audit a sorter manifest against a named panel.")
    audit.add_argument("--expected", type=Path, required=True)
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.set_defaults(func=audit_panel)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the named sample coverage panel tool."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
