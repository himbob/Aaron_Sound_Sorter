#!/usr/bin/env python3
"""
recover_regression_audio_exact.py

Standalone recovery script for Aaron_Sound_Sorter regression-audio fixtures.

What it does:
  - Searches the sample library for exact audio filenames.
  - Restores only known regression fixture filenames.
  - Copies files into the correct tests/regression_audio* folders.
  - Does NOT scan random test strings, so it will not copy junk names like:
      input.wav
      missing.wav
      kick_{idx}.wav
      g_{i}.wav
      loop_{i}.wav

Default search roots:
  <project>/tests/acceptance/locked_smoke_v1/samples

Run:
  python3 tools/recovery/recover_regression_audio_exact.py

Wider search:
  python3 tools/recovery/recover_regression_audio_exact.py --search-root /path/to/samples

Overwrite existing recovered files:
  python3 tools/recovery/recover_regression_audio_exact.py --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_EXTS = {
    ".wav",
    ".aif",
    ".aiff",
    ".flac",
    ".ogg",
    ".mp3",
    ".m4a",
    ".aac",
    ".au",
}

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    ".venv_phase4",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "outputs",
    "reports",
    "_reports",
    "dist",
    "build",
    "node_modules",
}


@dataclass(frozen=True)
class RestoreTarget:
    """One fixture target to restore."""

    folder_name: str
    target_filename: str
    source_filename: str
    required: bool


# Known real/private regression fixture targets.
#
# If target_filename differs from source_filename, this mirrors the current
# tests/conftest.py behavior, where a real locked-smoke sample is copied under
# the historical/private fixture name expected by older tests.
#
# required=True means the source should exist as a real sample or locked-smoke
# sample. required=False means pytest can generate a synthetic fallback, but the
# script will still use a real sample if it finds one.
TARGETS: list[RestoreTarget] = [
    # tests/regression_audio
    RestoreTarget(
        "regression_audio", "04_Dmn_176bpm_bass.wav", "MS_TLV1_03_The Way It Is_Bass_Eminor_98bpm_Dry.wav", True
    ),
    RestoreTarget(
        "regression_audio", "04.bass_92bpm_Em.wav", "MS_TLV1_03_The Way It Is_Bass_Eminor_98bpm_Dry.wav", True
    ),
    RestoreTarget(
        "regression_audio", "DOJO_CGNB_Female_Vocal_Shot_01_D.wav", "DOJO_CGNB_Female_Vocal_Shot_01_D.wav", True
    ),
    RestoreTarget("regression_audio", "Drumloop_hats_Dark_Rap_140BPM.wav", "Drumloop_hats_Dark_Rap_140BPM.wav", False),
    RestoreTarget("regression_audio", "GangstaFunk_Cmaj_96bpm.wav", "GangstaFunk_Cmaj_96bpm.wav", False),
    RestoreTarget(
        "regression_audio", "SCY095_03_Drums_Top_Loop_90bpm_01.wav", "SCY095_03_Drums_Top_Loop_90bpm_01.wav", False
    ),
    RestoreTarget("regression_audio", "1.Saxophone_1_110bpm_Am.wav", "AA_JBL_78bpm_Cm_Sax_Loop_1.wav", True),
    RestoreTarget(
        "regression_audio", "HipHopTapes_28_Saxophone_D#m_90bpm.wav", "HipHopTapes_28_Saxophone_D#m_90bpm.wav", True
    ),
    RestoreTarget("regression_audio", "Vocal Phrase We Up 140bpm.wav", "DOJO_FBP_Female_Vocal_Shout.wav", True),
    RestoreTarget("regression_audio", "Compton_Fmin_100bpm.wav", "Compton_Fmin_100bpm.wav", False),
    RestoreTarget("regression_audio", "AV5_5_94bpm_Hit 2.wav", "AV5_5_94bpm_Hit 2.wav", True),
    # v23
    RestoreTarget(
        "regression_audio_v23_voice_short_hit_guard",
        "DOJO_CGNB_Female_Vocal_Shot_01_D.wav",
        "DOJO_CGNB_Female_Vocal_Shot_01_D.wav",
        True,
    ),
    RestoreTarget("regression_audio_v23_voice_short_hit_guard", "29793.wav", "29793.wav", False),
    RestoreTarget("regression_audio_v23_voice_short_hit_guard", "16791.wav", "16791.wav", False),
    # v24 drum-loop steal guard
    RestoreTarget(
        "regression_audio_drum_loop_steal_guard",
        "Full Drum Loop 03 - 78BPM.wav",
        "Full Drum Loop 03 - 78BPM.wav",
        False,
    ),
    RestoreTarget(
        "regression_audio_drum_loop_steal_guard",
        "JL_TDBL_Drum Full_act up_138bpm.wav",
        "JL_TDBL_Drum Full_act up_138bpm.wav",
        False,
    ),
    RestoreTarget(
        "regression_audio_drum_loop_steal_guard",
        "SCY097_03_Drums_Full_Loop_90bpm_03.wav",
        "SCY097_03_Drums_Full_Loop_90bpm_03.wav",
        False,
    ),
    RestoreTarget(
        "regression_audio_drum_loop_steal_guard",
        "US_JF_Drum_140_Dapocket_FULL.wav",
        "US_JF_Drum_140_Dapocket_FULL.wav",
        False,
    ),
    RestoreTarget(
        "regression_audio_drum_loop_steal_guard", "2.Drum Loop_1_100bpm.wav", "2.Drum Loop_1_100bpm.wav", False
    ),
    RestoreTarget(
        "regression_audio_drum_loop_steal_guard", "A1_Kick_Clap_Loop_99bpm.wav", "A1_Kick_Clap_Loop_99bpm.wav", False
    ),
    RestoreTarget("regression_audio_drum_loop_steal_guard", "MKS_98_Beat1.wav", "MKS_98_Beat1.wav", False),
    RestoreTarget(
        "regression_audio_drum_loop_steal_guard", "QUp_DzU_DrumLp_03_100bpm.wav", "QUp_DzU_DrumLp_03_100bpm.wav", False
    ),
    RestoreTarget(
        "regression_audio_drum_loop_steal_guard",
        "WS2_KIT_1_WestCoast_Drum_&_Perc_Loop_101BPM.wav",
        "WS2_KIT_1_WestCoast_Drum_&_Perc_Loop_101BPM.wav",
        False,
    ),
    RestoreTarget("regression_audio_drum_loop_steal_guard", "03_bass_Emn_178bpm.wav", "03_bass_Emn_178bpm.wav", False),
    RestoreTarget("regression_audio_drum_loop_steal_guard", "04.bass_92bpm_Em.wav", "04.bass_92bpm_Em.wav", False),
    RestoreTarget(
        "regression_audio_drum_loop_steal_guard",
        "FL_TR_Kit02_96_Bass_Loop_Synth_Gm.wav",
        "FL_TR_Kit02_96_Bass_Loop_Synth_Gm.wav",
        False,
    ),
    RestoreTarget(
        "regression_audio_drum_loop_steal_guard",
        "AA_JBL_78bpm_Cm_Sax_Loop_1.wav",
        "AA_JBL_78bpm_Cm_Sax_Loop_1.wav",
        True,
    ),
    RestoreTarget(
        "regression_audio_drum_loop_steal_guard",
        "DOJO_FBP_Female_Vocal_Shout.wav",
        "DOJO_FBP_Female_Vocal_Shout.wav",
        True,
    ),
    RestoreTarget(
        "regression_audio_drum_loop_steal_guard",
        "DOJO_CGNB_Female_Vocal_Shot_01_D.wav",
        "DOJO_CGNB_Female_Vocal_Shot_01_D.wav",
        True,
    ),
    # v25
    RestoreTarget(
        "regression_audio_v25_transition_reverb",
        "GrimyHipHop_Saxophone_26_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav",
        "GrimyHipHop_Saxophone_15_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav",
        True,
    ),
    RestoreTarget("regression_audio_v25_transition_reverb", "Riser Short Effect.wav", "Riser Short Effect.wav", True),
    RestoreTarget(
        "regression_audio_v25_transition_reverb",
        "US_CHV2_Vocal_female_shouts_processed_13.wav",
        "DOJO_FBP_Female_Vocal_Shout.wav",
        True,
    ),
    RestoreTarget(
        "regression_audio_v25_transition_reverb", "Piano 1 - 80 Bpm - Key C.wav", "Piano 1 - 80 Bpm - Key C.wav", True
    ),
    # v26
    RestoreTarget(
        "regression_audio_v26_fx_smoke_reverb",
        "HipHopTapes_29_Saxophone_D#m_90bpm.wav",
        "HipHopTapes_29_Saxophone_D#m_90bpm.wav",
        True,
    ),
    RestoreTarget(
        "regression_audio_v26_fx_smoke_reverb",
        "SCY093_02_Sax_Loop_KeyAbm_89bpm_01.wav",
        "SCY097_03_Sax_Loop_KeyEm_90bpm_01.wav",
        True,
    ),
    RestoreTarget("regression_audio_v26_fx_smoke_reverb", "ABOUTME_94_DRUMLOOP.wav", "ABOUTME_94_DRUMLOOP.wav", False),
    # v28 FX ZIP matrix
    RestoreTarget(
        "regression_audio_v28_fx_zip_matrix", "AA_JBL_78bpm_Cm_Sax_Loop_1.wav", "AA_JBL_78bpm_Cm_Sax_Loop_1.wav", True
    ),
    RestoreTarget(
        "regression_audio_v28_fx_zip_matrix",
        "AMV_VRNB1_102_brass_saxophone_loop_cranesinthesky_Am.wav",
        "Brass_Saxophone_RnB_Multi_Instrument_F_Minor_80BPM.wav",
        True,
    ),
    RestoreTarget(
        "regression_audio_v28_fx_zip_matrix",
        "EWS_Keys_resampled_HipHop_RnB_G_Major_88BPM.wav",
        "EWS_Keys_resampled_HipHop_RnB_G_Major_88BPM.wav",
        True,
    ),
    RestoreTarget(
        "regression_audio_v28_fx_zip_matrix",
        "WS2_KIT_1_Electric_Piano_Chords_Fm_101BPM.wav",
        "WS2_KIT_1_Electric_Piano_Chords_Fm_101BPM.wav",
        True,
    ),
    RestoreTarget(
        "regression_audio_v28_fx_zip_matrix",
        "GS_Synth_Gangsta_Lead_G#min_97bpm.wav",
        "GS_Synth_Gangsta_Lead_G#min_97bpm.wav",
        False,
    ),
    RestoreTarget(
        "regression_audio_v28_fx_zip_matrix",
        "MS_TLV1_03_The Way It Is_Synth Lead 1_Eminor_98bpm_Wet.wav",
        "MS_TLV1_03_The Way It Is_Synth Lead 1_Eminor_98bpm_Wet.wav",
        False,
    ),
    RestoreTarget(
        "regression_audio_v28_fx_zip_matrix",
        "Money_vocals_female_rap_110bpm.wav",
        "Money_vocals_female_rap_110bpm.wav",
        True,
    ),
    RestoreTarget(
        "regression_audio_v28_fx_zip_matrix",
        "Phonk_Rap_Vocals_26_keyCmin_151bpm.wav",
        "Phonk_Rap_Vocals_26_keyCmin_151bpm.wav",
        True,
    ),
    RestoreTarget(
        "regression_audio_v28_fx_zip_matrix", "Vocal Phrase We Up 140bpm.wav", "DOJO_FBP_Female_Vocal_Shout.wav", True
    ),
    RestoreTarget("regression_audio_v28_fx_zip_matrix", "Stab 3.wav", "DOJO_CGNB_Female_Vocal_Shot_01_D.wav", True),
    RestoreTarget(
        "regression_audio_v28_fx_zip_matrix",
        "MS_O_01_Outlaw_Fx Police_D#minor_91bpm_Wet.wav",
        "MS_O_01_Outlaw_Fx Police_D#minor_91bpm_Wet.wav",
        True,
    ),
    RestoreTarget("regression_audio_v28_fx_zip_matrix", "2.Drum Loop_1_100bpm.wav", "2.Drum Loop_1_100bpm.wav", False),
    RestoreTarget("regression_audio_v28_fx_zip_matrix", "ABOUTME_94_DRUMLOOP.wav", "ABOUTME_94_DRUMLOOP.wav", False),
    # full thread matrix
    RestoreTarget(
        "regression_audio_thread_matrix", "AA_JBL_74bpm_Am_Sax_Loop_18.wav", "AA_JBL_74bpm_Am_Sax_Loop_18.wav", False
    ),
    RestoreTarget(
        "regression_audio_thread_matrix", "AA_JBL_78bpm_Cm_Sax_Loop_1.wav", "AA_JBL_78bpm_Cm_Sax_Loop_1.wav", True
    ),
    RestoreTarget(
        "regression_audio_thread_matrix", "AA_JBL_86bpm_Cm_Sax_Loop_12.wav", "AA_JBL_86bpm_Cm_Sax_Loop_12.wav", False
    ),
    RestoreTarget(
        "regression_audio_thread_matrix", "AA_JBL_90bpm_Em_Sax_Loop_3.wav", "AA_JBL_90bpm_Em_Sax_Loop_3.wav", False
    ),
    RestoreTarget(
        "regression_audio_thread_matrix",
        "AMV_VRNB1_102_brass_saxophone_loop_cranesinthesky_Am.wav",
        "Brass_Saxophone_RnB_Multi_Instrument_F_Minor_80BPM.wav",
        True,
    ),
    RestoreTarget(
        "regression_audio_thread_matrix",
        "GrimyHipHop_Saxophone_26_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav",
        "GrimyHipHop_Saxophone_15_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav",
        True,
    ),
    RestoreTarget(
        "regression_audio_thread_matrix",
        "HipHopTapes_28_Saxophone_D#m_90bpm.wav",
        "HipHopTapes_28_Saxophone_D#m_90bpm.wav",
        True,
    ),
    RestoreTarget(
        "regression_audio_thread_matrix",
        "HipHopTapes_29_Saxophone_D#m_90bpm.wav",
        "HipHopTapes_29_Saxophone_D#m_90bpm.wav",
        True,
    ),
    RestoreTarget(
        "regression_audio_thread_matrix",
        "SCY093_02_Sax_Loop_KeyAbm_89bpm_01.wav",
        "SCY097_03_Sax_Loop_KeyEm_90bpm_01.wav",
        True,
    ),
    RestoreTarget("regression_audio_thread_matrix", "ABOUTME_94_DRUMLOOP.wav", "ABOUTME_94_DRUMLOOP.wav", False),
    RestoreTarget("regression_audio_thread_matrix", "DRUMS_LOOP_125BPM.wav", "DRUMS_LOOP_125BPM.wav", False),
    RestoreTarget(
        "regression_audio_thread_matrix", "Full Drum Loop 03 - 78BPM.wav", "Full Drum Loop 03 - 78BPM.wav", False
    ),
    RestoreTarget(
        "regression_audio_thread_matrix",
        "NDAEV4_FULL_DRUM_LOOP_01_AFTERLIFE_74BPM.wav",
        "NDAEV4_FULL_DRUM_LOOP_01_AFTERLIFE_74BPM.wav",
        False,
    ),
    RestoreTarget(
        "regression_audio_thread_matrix", "THO_ck4_drums top_130 bpm.wav", "THO_ck4_drums top_130 bpm.wav", False
    ),
    RestoreTarget(
        "regression_audio_thread_matrix",
        "DOJO_CGNB_Female_Vocal_Shot_01_D.wav",
        "DOJO_CGNB_Female_Vocal_Shot_01_D.wav",
        True,
    ),
    RestoreTarget(
        "regression_audio_thread_matrix", "DOJO_FBP_Female_Vocal_Shout.wav", "DOJO_FBP_Female_Vocal_Shout.wav", True
    ),
    RestoreTarget(
        "regression_audio_thread_matrix",
        "Money_vocals_female_rap_110bpm.wav",
        "Money_vocals_female_rap_110bpm.wav",
        True,
    ),
    RestoreTarget("regression_audio_thread_matrix", "Stab 3.wav", "DOJO_CGNB_Female_Vocal_Shot_01_D.wav", True),
    RestoreTarget("regression_audio_thread_matrix", "125591.wav", "125591.wav", False),
    RestoreTarget("regression_audio_thread_matrix", "13144.wav", "13144.wav", False),
    RestoreTarget("regression_audio_thread_matrix", "16791.wav", "16791.wav", False),
    RestoreTarget("regression_audio_thread_matrix", "33157.wav", "33157.wav", False),
    RestoreTarget("regression_audio_thread_matrix", "50728.wav", "50728.wav", False),
    RestoreTarget("regression_audio_thread_matrix", "13115.wav", "13115.wav", False),
    RestoreTarget("regression_audio_thread_matrix", "16291.wav", "16291.wav", False),
    RestoreTarget("regression_audio_thread_matrix", "AV5_5_94bpm_Hit 2.wav", "AV5_5_94bpm_Hit 2.wav", True),
    # parent eligibility percussion/FX guard real folder
    RestoreTarget("regression_audio_percussion_fx_guard", "125591.wav", "125591.wav", False),
    RestoreTarget("regression_audio_percussion_fx_guard", "13144.wav", "13144.wav", False),
    RestoreTarget("regression_audio_percussion_fx_guard", "33157.wav", "33157.wav", False),
    RestoreTarget("regression_audio_percussion_fx_guard", "50728.wav", "50728.wav", False),
    RestoreTarget("regression_audio_percussion_fx_guard", "16291.wav", "16291.wav", False),
]


def normalize_name(value: str) -> str:
    """Normalize filenames for case-insensitive Unicode-safe matching."""
    return unicodedata.normalize("NFC", value).casefold()


def find_project_root() -> Path:
    """Use cwd if it looks like the project, otherwise use the default root."""
    cwd = Path.cwd()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "Aaron_Sound_Sorter.py").exists() and (candidate / "tests").exists():
            return candidate
    return DEFAULT_PROJECT_ROOT


def default_search_roots(project_root: Path) -> list[Path]:
    """Return default search roots."""
    return [project_root / "tests" / "acceptance" / "locked_smoke_v1" / "samples"]


def ensure_real_folder(path: Path) -> None:
    """Ensure path is a real folder, replacing symlink/non-folder safely."""
    if path.is_symlink():
        old_target = path.readlink()
        path.unlink()
        print(f"REMOVED SYMLINK | {path} -> {old_target}")

    if path.exists() and not path.is_dir():
        backup = path.with_name(f"{path.name}_not_a_folder_backup")
        suffix = 1
        while backup.exists():
            backup = path.with_name(f"{path.name}_not_a_folder_backup_{suffix}")
            suffix += 1
        path.rename(backup)
        print(f"MOVED NON-FOLDER | {path} -> {backup}")

    path.mkdir(parents=True, exist_ok=True)


def iter_candidate_files(root: Path):
    """Yield files under a root, skipping generated/build junk."""
    if not root.exists():
        print(f"WARNING: search root missing: {root}")
        return

    if root.is_file():
        yield root
        return

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        current = Path(dirpath)
        for filename in filenames:
            yield current / filename


def build_loose_index(search_roots: list[Path], wanted_names: set[str]) -> tuple[dict[str, Path], list[Path]]:
    """Index exact wanted loose files and collect zip paths."""
    wanted_keys = {normalize_name(name) for name in wanted_names}
    loose: dict[str, Path] = {}
    zip_paths: list[Path] = []

    for root in search_roots:
        print(f"Searching loose files: {root}")
        for path in iter_candidate_files(root):
            suffix = path.suffix.lower()
            if suffix == ".zip":
                zip_paths.append(path)
                continue

            if suffix not in AUDIO_EXTS:
                continue

            key = normalize_name(path.name)
            if key in wanted_keys and key not in loose:
                loose[key] = path

    return loose, zip_paths


def build_zip_index(zip_paths: list[Path], wanted_names: set[str]) -> dict[str, tuple[Path, str]]:
    """Index wanted filenames inside zip files."""
    wanted_keys = {normalize_name(name) for name in wanted_names}
    found: dict[str, tuple[Path, str]] = {}

    if not zip_paths:
        return found

    print(f"Searching inside ZIP files: {len(zip_paths)}")

    for zip_path in zip_paths:
        if len(found) >= len(wanted_keys):
            break

        try:
            with zipfile.ZipFile(zip_path) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue

                    member_name = Path(member.filename).name
                    key = normalize_name(member_name)

                    if key in wanted_keys and key not in found:
                        found[key] = (zip_path, member.filename)

        except (OSError, zipfile.BadZipFile):
            continue

    return found


def copy_from_zip(zip_path: Path, member_name: str, destination: Path) -> None:
    """Extract one zip member to destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def restore_targets(
    project_root: Path,
    targets: list[RestoreTarget],
    loose_index: dict[str, Path],
    zip_index: dict[str, tuple[Path, str]],
    overwrite: bool,
) -> dict[str, int]:
    """Restore all target files."""
    counts = {
        "exists": 0,
        "copied": 0,
        "extracted": 0,
        "missing_required": 0,
        "missing_optional": 0,
    }

    folders = sorted({target.folder_name for target in targets})
    for folder_name in folders:
        ensure_real_folder(project_root / "tests" / folder_name)

    for target in targets:
        destination = project_root / "tests" / target.folder_name / target.target_filename

        if destination.exists() and not overwrite:
            counts["exists"] += 1
            print(f"EXISTS   | tests/{target.folder_name}/{target.target_filename}")
            continue

        source_key = normalize_name(target.source_filename)
        source_path = loose_index.get(source_key)

        if source_path is not None:
            shutil.copy2(source_path, destination)
            counts["copied"] += 1
            print(f"COPIED   | tests/{target.folder_name}/{target.target_filename}")
            print(f"           from: {source_path}")
            if target.target_filename != target.source_filename:
                print(f"           as:   {target.target_filename}")
            continue

        zip_hit = zip_index.get(source_key)
        if zip_hit is not None:
            zip_path, member_name = zip_hit
            copy_from_zip(zip_path, member_name, destination)
            counts["extracted"] += 1
            print(f"EXTRACT  | tests/{target.folder_name}/{target.target_filename}")
            print(f"           zip:    {zip_path}")
            print(f"           member: {member_name}")
            if target.target_filename != target.source_filename:
                print(f"           as:     {target.target_filename}")
            continue

        if target.required:
            counts["missing_required"] += 1
            print(f"MISSING REQUIRED | tests/{target.folder_name}/{target.target_filename}")
            print(f"                   searched source filename: {target.source_filename}")
        else:
            counts["missing_optional"] += 1
            print(f"MISSING OPTIONAL | tests/{target.folder_name}/{target.target_filename}")
            print(f"                   searched source filename: {target.source_filename}")

    return counts


def print_final_counts(project_root: Path, targets: list[RestoreTarget]) -> None:
    """Print final per-folder counts."""
    print()
    print("Final folder counts")
    print("-------------------")

    for folder_name in sorted({target.folder_name for target in targets}):
        folder = project_root / "tests" / folder_name
        count = 0
        if folder.exists() and folder.is_dir():
            count = sum(1 for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTS)
        print(f"{count:3} audio files | {folder}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Recover exact Aaron Sound Sorter regression audio fixtures from the sample library."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=f"Project root. Default: auto-detect, then {DEFAULT_PROJECT_ROOT}",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=None,
        help="Search root. Repeat as needed. Default: the locked-smoke sample folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist in tests/regression_audio* folders.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any required file is missing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """CLI entry point."""
    args = parse_args(argv)

    project_root = args.project_root.expanduser() if args.project_root else find_project_root()
    search_roots = args.search_root if args.search_root else default_search_roots(project_root)
    search_roots = [path.expanduser() for path in search_roots]

    wanted_source_names = {target.source_filename for target in TARGETS}

    print("Recover Aaron Sound Sorter regression audio")
    print("------------------------------------------")
    print(f"Project root: {project_root}")
    print(f"Targets:      {len(TARGETS)}")
    print(f"Source names: {len(wanted_source_names)}")
    print()
    print("This script restores only known regression fixture names.")
    print("It does not scan random placeholder strings from tests.")
    print()

    loose_index, zip_paths = build_loose_index(search_roots, wanted_source_names)
    print(f"Loose source matches: {len(loose_index)}")
    print(f"ZIP files found:      {len(zip_paths)}")
    print()

    missing_after_loose = {name for name in wanted_source_names if normalize_name(name) not in loose_index}
    zip_index = build_zip_index(zip_paths, missing_after_loose)
    print(f"ZIP source matches:   {len(zip_index)}")
    print()

    counts = restore_targets(
        project_root=project_root,
        targets=TARGETS,
        loose_index=loose_index,
        zip_index=zip_index,
        overwrite=args.overwrite,
    )

    print()
    print("Summary")
    print("-------")
    for key in ["exists", "copied", "extracted", "missing_required", "missing_optional"]:
        print(f"{key}: {counts[key]}")

    print_final_counts(project_root, TARGETS)

    if counts["missing_required"]:
        print()
        print("Some REQUIRED files were not found in the default search roots.")
        print("Try a wider search:")
        print("  python3 tools/recovery/recover_regression_audio_exact.py --search-root /path/to/samples --strict")
        return 1 if args.strict else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
