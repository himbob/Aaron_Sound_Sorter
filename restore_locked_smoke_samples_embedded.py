#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import unicodedata
import zipfile
from pathlib import Path


PROJECT_ROOT = Path("/Volumes/T9/testbed/Aaron_Sound_Sorter")
DEST_RELATIVE = Path("tests/acceptance/locked_smoke_v1/samples")

DEFAULT_SEARCH_ROOTS = [
    Path("/Volumes/T9/music_production/samples"),
    Path("/Volumes/T9/testbed"),
]

EXPECTED = [
    "CS_NE_Kick_OneShot_Monroe.wav",
    "JACKBABY_SNARE_11.wav",
    "Clap2.wav",
    "Drum_Full_06_95bpm.wav",
    "MS_TLV1_03_The Way It Is_Bass_Eminor_98bpm_Dry.wav",
    "Piano 1 - 80 Bpm - Key C.wav",
    "03.strings_77bpm_Ebm.wav",
    "01 E Pad 097 Ebm.wav",
    "01_WCS_No_Safety_BPM92_D#min__Bells.wav",
    "02_K_DOT_vol_4_Lead_Em_100bpm.wav",
    "Money_vocals_female_rap_110bpm.wav",
    "DOJO_FBP_Female_Vocal_Shout.wav",
    "AV5_5_94bpm_Hit 2.wav",
    "AA_JBL_74bpm_Am_Sax_Loop_11.wav",
    "SHE2_loop 14_full_123 bpm_D#.wav",
    "Brass_Saxophone_RnB_Multi_Instrument_F_Minor_80BPM.wav",
    "Riser Short Effect.wav",
    "AA_JBL_78bpm_Cm_Sax_Loop_1.wav",
    "JazzHipHop_15_Saxophone_Melody_D#m_84bpm.wav",
    "DHB Loop 1 Vintage Piano E minor 81 Bpm Piano chords.wav",
    "05_Emn_198bpm_synth 1.wav",
    "DOJO_CGNB_Female_Vocal_Shot_01_D.wav",
    "WS2_KIT_1_Electric_Piano_Chords_Fm_101BPM.wav",
    "Phonk_Rap_Vocals_26_keyCmin_151bpm.wav",
    "Intro Sax_FMinor_96BPM.wav",
    "SCY097_03_Sax_Loop_KeyEm_90bpm_01.wav",
    "HipHopTapes_29_Saxophone_D#m_90bpm.wav",
    "Chopart3_GodlikeLoops_90_Piano_Sax_Songstarter_Abm_Loop_Fine.wav",
    "GrimyHipHop_Saxophone_15_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav",
    "HipHopTapes_31_Saxophone_D#m_90bpm.wav",
    "EWS_Keys_resampled_HipHop_RnB_G_Major_88BPM.wav",
    "MS_O_01_Outlaw_Fx Police_D#minor_91bpm_Wet.wav",
]

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


def norm(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def is_inside(path: Path, possible_parent: Path) -> bool:
    try:
        path_resolved = path.resolve()
        parent_resolved = possible_parent.resolve()
    except OSError:
        return False

    return path_resolved == parent_resolved or parent_resolved in path_resolved.parents


def collect_sources(search_roots: list[Path], dest_dir: Path) -> tuple[dict[str, Path], list[Path]]:
    expected_keys = {norm(name) for name in EXPECTED}
    loose: dict[str, Path] = {}
    zip_paths: list[Path] = []

    for root in search_roots:
        if not root.exists():
            print(f"WARNING: search root does not exist: {root}")
            continue

        print(f"Searching loose files under: {root}")

        for dirpath, dirnames, filenames in os.walk(root):
            current_dir = Path(dirpath)

            if is_inside(current_dir, dest_dir):
                dirnames[:] = []
                continue

            dirnames[:] = [
                dirname
                for dirname in dirnames
                if dirname not in SKIP_DIR_NAMES
            ]

            for filename in filenames:
                path = current_dir / filename
                key = norm(filename)

                if filename.casefold().endswith(".zip"):
                    zip_paths.append(path)

                if key in expected_keys and key not in loose:
                    loose[key] = path

    return loose, zip_paths


def extract_from_zip(zip_path: Path, wanted_name: str, dest_path: Path) -> bool:
    wanted_key = norm(wanted_name)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue

                member_name = Path(member.filename).name
                if norm(member_name) != wanted_key:
                    continue

                print(f"EXTRACT | {wanted_name}")
                print(f"          from zip: {zip_path}")
                print(f"          member:   {member.filename}")
                print(f"          to:       {dest_path}")

                with archive.open(member) as source, dest_path.open("wb") as dest:
                    shutil.copyfileobj(source, dest)

                return True

    except (OSError, zipfile.BadZipFile):
        return False

    return False


def restore_samples(dest_dir: Path, search_roots: list[Path], overwrite: bool) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Destination: {dest_dir}")
    print(f"Expected files: {len(EXPECTED)}")
    print()

    loose_sources, zip_paths = collect_sources(search_roots, dest_dir)

    print()
    print(f"Candidate ZIP files found: {len(zip_paths)}")
    print()

    copied = 0
    extracted = 0
    existing = 0
    missing = 0

    for filename in EXPECTED:
        dest_path = dest_dir / filename

        if dest_path.exists() and not overwrite:
            existing += 1
            print(f"EXISTS  | {filename}")
            continue

        source_path = loose_sources.get(norm(filename))

        if source_path is not None:
            print(f"COPIED  | {filename}")
            print(f"          from: {source_path}")
            print(f"          to:   {dest_path}")
            shutil.copy2(source_path, dest_path)
            copied += 1
            continue

        found_in_zip = False
        for zip_path in zip_paths:
            if extract_from_zip(zip_path, filename, dest_path):
                found_in_zip = True
                extracted += 1
                break

        if found_in_zip:
            continue

        missing += 1
        print(f"MISSING | {filename}")

    print()
    print("Summary")
    print("-------")
    print(f"already present: {existing}")
    print(f"copied loose:    {copied}")
    print(f"extracted zip:   {extracted}")
    print(f"missing:         {missing}")
    print(f"destination:     {dest_dir}")

    return 1 if missing else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = args.project_root
    dest_dir = args.dest if args.dest else project_root / DEST_RELATIVE
    search_roots = args.search_root if args.search_root else DEFAULT_SEARCH_ROOTS

    return restore_samples(
        dest_dir=dest_dir,
        search_roots=search_roots,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
