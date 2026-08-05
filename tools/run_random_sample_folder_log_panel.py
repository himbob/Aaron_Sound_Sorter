#!/usr/bin/env python3
"""Run Aaron Sound Sorter against random real sample folders, logs only.

This tool samples folders under a sample-library root, picks a few audio files from
each selected folder, runs the sorter one file at a time, and keeps only manifests,
summaries, stdout, and CSV reports. It does not copy audio into the upload-back zip.

Default goal:
  - 20 different folders
  - 3 audio files per folder
  - no subprocess timeout unless --timeout-sec is set

This is intentionally a diagnostic runner. Folder names and filenames are used only
to flag suspicious results in the report. They do not affect sorter placement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

csv.field_size_limit(sys.maxsize)

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".au"}
DEFAULT_SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    ".venv_phase4",
    "__MACOSX",
    "_reports",
    "reports",
    "_backup",
    "_backups",
    "_TO_REVIEW",
    "Aaron_Sorted_Sounds",
    "node_modules",
    "__pycache__",
}
DEFAULT_SKIP_PATH_WORDS = {
    "aaron_sound_sorter",
    "upload_back",
    "report",
    "reports",
    "backup",
    "backups",
    "trash",
    "__macosx",
    # Default random sweeps should find fresh material, not the same curated/test
    # locations already used for regression work.
    "sorted samples",
    "fx_aaron",
    "one_shot_percussive_sounds",
    "loop 4",
    "combined_training",
    "regression_audio",
    "test_runner",
    # AKWF files are tiny/wavetable material.  The sorter is right to send them
    # to Broken Or Tiny, so skip those folders in broad random sweeps.
    "akwf",
    "adventure kid waveforms",
    "wavetable",
    "wavetables",
}


@dataclass(frozen=True)
class SelectedFile:
    source_path: Path
    folder_key: str
    folder_index: int
    file_index_in_folder: int


def is_audio_file(path: Path, *, skip_akwf: bool = True) -> bool:
    name = path.name
    if name.startswith(".") or name.startswith("._"):
        return False
    if skip_akwf and name.upper().startswith("AKWF"):
        return False
    if any(part.startswith(".") or part == "__MACOSX" for part in path.parts):
        return False
    if skip_akwf and any("AKWF" in part.upper() for part in path.parts):
        return False
    return path.suffix.lower() in AUDIO_EXTS


def clean_token_text(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def should_skip_dir(path: Path, sample_root: Path, skip_path_words: set[str]) -> bool:
    parts = [p for p in path.parts]
    if any(part in DEFAULT_SKIP_DIR_NAMES for part in parts):
        return True
    rel = ""
    try:
        rel = str(path.relative_to(sample_root)).lower()
    except ValueError:
        rel = str(path).lower()
    return any(word in rel for word in skip_path_words)


def iter_audio_folders(
    sample_root: Path,
    follow_symlinks: bool,
    *,
    skip_akwf: bool,
    skip_path_words: set[str],
) -> list[tuple[Path, list[Path]]]:
    """Return folders that directly contain audio files."""
    folders: list[tuple[Path, list[Path]]] = []
    for root, dirnames, filenames in os.walk(sample_root, followlinks=follow_symlinks):
        root_path = Path(root)
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and not should_skip_dir(root_path / d, sample_root, skip_path_words)
        ]
        if should_skip_dir(root_path, sample_root, skip_path_words):
            continue
        audio_files = []
        for filename in filenames:
            path = root_path / filename
            if is_audio_file(path, skip_akwf=skip_akwf):
                audio_files.append(path)
        if audio_files:
            folders.append((root_path, sorted(audio_files, key=lambda p: p.name.lower())))
    return sorted(folders, key=lambda item: str(item[0]).lower())


def relative_folder_key(folder: Path, sample_root: Path) -> str:
    try:
        return str(folder.relative_to(sample_root))
    except ValueError:
        return str(folder)


def select_random_files(
    *,
    sample_root: Path,
    folder_count: int,
    files_per_folder: int,
    max_total: int,
    seed: int,
    follow_symlinks: bool,
    contains: str = "",
    skip_akwf: bool = True,
    skip_path_words: set[str] | None = None,
) -> list[SelectedFile]:
    rng = random.Random(seed)
    folders = iter_audio_folders(
        sample_root,
        follow_symlinks,
        skip_akwf=skip_akwf,
        skip_path_words=skip_path_words or set(DEFAULT_SKIP_PATH_WORDS),
    )
    if contains:
        needle = contains.lower()
        filtered: list[tuple[Path, list[Path]]] = []
        for folder, files in folders:
            kept = [p for p in files if needle in str(p).lower()]
            if kept:
                filtered.append((folder, kept))
        folders = filtered
    rng.shuffle(folders)
    selected: list[SelectedFile] = []
    for folder_index, (folder, files) in enumerate(folders[: max(0, folder_count)], start=1):
        files = list(files)
        rng.shuffle(files)
        folder_key = relative_folder_key(folder, sample_root)
        for file_index, source in enumerate(files[: max(1, files_per_folder)], start=1):
            selected.append(SelectedFile(source.resolve(), folder_key, folder_index, file_index))
            if max_total > 0 and len(selected) >= max_total:
                return selected
    return selected


def read_first_manifest_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def parse_role(row: dict[str, str]) -> str:
    import json

    try:
        payload = json.loads(row.get("parent_role_audit_json", "") or "{}")
    except Exception:
        return ""
    return str(payload.get("detected_parent_role", ""))


def source_text(path: Path, folder_key: str) -> str:
    return clean_token_text(f"{folder_key} {path.name}")


def expected_top_from_source(path: Path, folder_key: str) -> str:
    text = source_text(path, folder_key)
    # Use only strong clues. This is diagnostic-only and should avoid false certainty.
    if re.search(
        r"\b(bass drum|kick|snare|clap|rimshot|rim shot|sidestick|side stick|hat|hihat|hi hat|tom|bongo|conga|tabla|shaker|tambourine|percussion|perc|drumloop|drum loop|drum loops|drums|beat)\b",
        text,
    ):
        return "Drums"
    if re.search(
        r"\b(keys jangling|key jangling|keys coins|coins|small objects|riser|rise hit|downlifter|whoosh|swoosh|impact|boom|glitch|stutter|alarm|siren|alert|foley|sfx|sound effect|explosion|shatter|reverse fx)\b",
        text,
    ):
        return "FX"
    if re.search(
        r"\b(sax|saxophone|flute|brass|woodwind|guitar|rhodes|piano|organ|synth|lead|pad|string|strings|violin|cello|808|sub bass|bass loop|bass guitar|vocal|vocals|vox|voice|choir|melody loop|melody)\b",
        text,
    ):
        return "Instruments"
    if re.search(r"\b(texture|atmosphere|ambience|ambient|drone|field recording|noise bed|room tone)\b", text):
        return "Textures"
    return ""


def allowed_tops_for_source(path: Path, folder_key: str) -> set[str]:
    text = source_text(path, folder_key)
    expected = expected_top_from_source(path, folder_key)
    if not expected:
        return set()
    if re.search(r"\b(vocal|vocals|vox|voice|shout|chant|rap|breath|mouth)\b", text):
        # Current product sometimes treats processed vocal as FX/Human and Voice FX.
        return {"Instruments", "FX", "_TO_REVIEW"}
    return {expected, "_TO_REVIEW"}


def suspicious_reason(row: dict[str, str], source_path: Path, folder_key: str) -> str:
    folder = row.get("folder_path", "") or row.get("placed_path", "")
    final_top = row.get("final_top", "") or (folder.split("/", 1)[0] if folder else "")
    text = source_text(source_path, folder_key)
    reasons: list[str] = []

    allowed = allowed_tops_for_source(source_path, folder_key)
    if allowed and final_top and final_top not in allowed:
        reasons.append(f"source_hint_expected_{sorted(allowed)}_got_{final_top}")

    if "Human and Voice FX" in folder and not re.search(
        r"\b(vocal|vocals|vox|voice|shout|chant|rap|breath|mouth|choir)\b", text
    ):
        reasons.append("non_vocal_source_to_human_voice_fx")

    if "Alarm" in folder and not re.search(r"\b(alarm|siren|alert|beep|bleep|warning)\b", text):
        reasons.append("non_alarm_source_to_alarm")

    if "Bass Loops" in folder and re.search(
        r"\b(rhodes|piano|keys|organ|synth lead|lead|pad|melody|guitar|string|strings|sax|brass|woodwind|vocal|vox|voice)\b",
        text,
    ):
        reasons.append("non_bass_instrument_source_to_bass_loop")

    if "Brass and Woodwinds" in folder and re.search(
        r"\b(melody loop|melody|mixed|kit|rhodes|piano|keys|organ|bell|bells|chime|chimes|vocal|vocals|vox|voice|guitar|string|strings|synth)\b",
        text,
    ):
        reasons.append("mixed_or_non_woodwind_source_to_brass_woodwinds")

    if final_top == "Drums" and re.search(
        r"\b(sax|saxophone|flute|brass|woodwind|guitar|rhodes|piano|organ|synth lead|pad|string|strings|vocal|vocals|vox|voice|choir)\b",
        text,
    ):
        if not re.search(
            r"\b(bass drum|drum and bass|drum loops|drum loop|snare|hat|hihat|hi hat|kick|tom|rim|cymbal|percussion|perc)\b",
            text,
        ):
            reasons.append("pitched_instrument_source_to_drums")

    if (
        final_top == "FX"
        and re.search(
            r"\b(rhodes|piano|keys|organ|synth lead|guitar|string|strings|sax|saxophone|brass|woodwind|melody loop)\b",
            text,
        )
        and "Human and Voice FX" not in folder
    ):
        reasons.append("instrument_source_to_fx")

    return ";".join(dict.fromkeys(reasons))


def safe_case_name(index: int, path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)[:50].strip("_") or "audio"
    return f"{index:04d}_{digest}_{stem}"


def copy_or_symlink_source(source: Path, dest: Path, copy_audio: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if copy_audio:
        shutil.copy2(source, dest)
        return
    try:
        os.symlink(str(source), str(dest))
    except Exception:
        shutil.copy2(source, dest)


def run_sorter_one_file(
    *,
    project_root: Path,
    source: Path,
    output_dir: Path,
    brain: str,
    timeout_sec: float,
    stdout_path: Path,
    copy_audio: bool,
) -> tuple[str, float]:
    with tempfile.TemporaryDirectory(prefix="aaron_random_folder_panel_") as tmp:
        tmp_path = Path(tmp)
        input_dir = tmp_path / "input"
        sort_output = tmp_path / "output"
        input_dir.mkdir()
        input_path = input_dir / source.name
        copy_or_symlink_source(source, input_path, copy_audio)
        cmd = [
            sys.executable,
            "Aaron_Sound_Sorter.py",
            "sort",
            str(input_dir),
            str(sort_output),
            "--brain",
            brain,
            "--no-zip",
        ]
        start = time.monotonic()
        timeout = None if timeout_sec <= 0 else float(timeout_sec)
        try:
            completed = subprocess.run(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
            seconds = time.monotonic() - start
            stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
            status = "pass" if completed.returncode == 0 else f"exit_{completed.returncode}"
        except subprocess.TimeoutExpired as error:
            seconds = time.monotonic() - start
            out = error.stdout or ""
            if isinstance(out, bytes):
                out = out.decode("utf-8", errors="replace")
            stdout_path.write_text(out + "\nTIMEOUT\n", encoding="utf-8", errors="replace")
            status = "timeout"

        manifest = sort_output / "Aaron_Sorted_Sounds_manifest.csv"
        summary = sort_output / "Aaron_Sorted_Sounds_summary.txt"
        if manifest.exists():
            shutil.copy2(manifest, output_dir / "Aaron_Sorted_Sounds_manifest.csv")
        if summary.exists():
            shutil.copy2(summary, output_dir / "Aaron_Sorted_Sounds_summary.txt")
        return status, seconds


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_summary(
    *,
    report_dir: Path,
    sample_root: Path,
    selected: list[SelectedFile],
    results: list[list[str]],
    suspicious: list[list[str]],
    args: argparse.Namespace,
) -> None:
    counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    folder_counts: dict[str, int] = {}
    for row in results:
        status_counts[row[4]] = status_counts.get(row[4], 0) + 1
        top = row[7] or "NO_TOP"
        counts[top] = counts.get(top, 0) + 1
        folder = row[8] or "NO_FOLDER"
        folder_counts[folder] = folder_counts.get(folder, 0) + 1
    with (report_dir / "random_folder_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Sample root: {sample_root}\n")
        handle.write(f"Seed: {args.seed}\n")
        handle.write(f"Folder count requested: {args.folder_count}\n")
        handle.write(f"Files per folder requested: {args.files_per_folder}\n")
        handle.write(f"Selected files: {len(selected)}\n")
        handle.write(f"Suspicious rows: {len(suspicious)}\n")
        handle.write(f"Timeout seconds: {'none' if args.timeout_sec <= 0 else args.timeout_sec}\n")
        handle.write("\nStatus counts:\n")
        for status, count in sorted(status_counts.items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"{count:5d}  {status}\n")
        handle.write("\nTop folder counts:\n")
        for top, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"{count:5d}  {top}\n")
        handle.write("\nMost common final folders:\n")
        for folder, count in sorted(folder_counts.items(), key=lambda item: (-item[1], item[0]))[:40]:
            handle.write(f"{count:5d}  {folder}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--sample-root", type=Path, default=Path("/path/to/sample-library"))
    parser.add_argument("--brain", default="stage4_folder_brain.json")
    parser.add_argument("--out-root", type=Path, default=Path("reports/random_sample_folder_log_panel"))
    parser.add_argument("--folder-count", type=int, default=20)
    parser.add_argument("--files-per-folder", type=int, default=3)
    parser.add_argument(
        "--max-total", type=int, default=0, help="0 means no extra cap beyond folder-count * files-per-folder"
    )
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--timeout-sec", type=float, default=0.0, help="0 means no subprocess timeout")
    parser.add_argument("--contains", default="", help="Only select files whose full path contains this text")
    parser.add_argument("--follow-symlinks", action="store_true")
    parser.add_argument("--copy-audio", action="store_true", help="Copy into temp input instead of symlinking")
    parser.add_argument("--include-akwf", action="store_true", help="Include AKWF/wavetable/tiny waveform folders")
    parser.add_argument("--include-sorted-samples", action="store_true", help="Include the curated Sorted samples tree")
    parser.add_argument(
        "--include-known-test-material",
        action="store_true",
        help="Include common project test packs such as FX_Aaron and percussion ZIP extracts",
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    sample_root = args.sample_root.expanduser().resolve()
    if not sample_root.exists():
        raise SystemExit(f"Sample root does not exist: {sample_root}")
    if not (project_root / "Aaron_Sound_Sorter.py").exists():
        raise SystemExit(f"Project root does not contain Aaron_Sound_Sorter.py: {project_root}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = (project_root / args.out_root / f"run_{stamp}").resolve()
    cases_dir = report_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    skip_path_words = set(DEFAULT_SKIP_PATH_WORDS)
    if args.include_akwf:
        skip_path_words.discard("akwf")
        skip_path_words.discard("adventure kid waveforms")
        skip_path_words.discard("wavetable")
        skip_path_words.discard("wavetables")
    if args.include_sorted_samples:
        skip_path_words.discard("sorted samples")
    if args.include_known_test_material:
        for word in [
            "fx_aaron",
            "one_shot_percussive_sounds",
            "loop 4",
            "combined_training",
            "regression_audio",
            "test_runner",
        ]:
            skip_path_words.discard(word)

    selected = select_random_files(
        sample_root=sample_root,
        folder_count=args.folder_count,
        files_per_folder=args.files_per_folder,
        max_total=args.max_total,
        seed=args.seed,
        follow_symlinks=args.follow_symlinks,
        contains=args.contains,
        skip_akwf=not args.include_akwf,
        skip_path_words=skip_path_words,
    )
    if not selected:
        raise SystemExit("No audio files selected. Check sample root, filters, and symlink settings.")

    selection_header = ["index", "folder_index", "file_index_in_folder", "folder_key", "source_path"]
    selection_rows = [
        [str(i), str(item.folder_index), str(item.file_index_in_folder), item.folder_key, str(item.source_path)]
        for i, item in enumerate(selected, start=1)
    ]
    write_csv(report_dir / "random_folder_selection.csv", selection_header, selection_rows)

    header = [
        "index",
        "folder_index",
        "file_index_in_folder",
        "folder_key",
        "status",
        "source_path",
        "source_name",
        "final_top",
        "folder_path",
        "consensus_status",
        "detected_parent_role",
        "shape_vote",
        "shape_confidence",
        "suspicious_reason",
        "seconds",
        "case_report_dir",
    ]
    results: list[list[str]] = []
    suspicious: list[list[str]] = []

    for index, item in enumerate(selected, start=1):
        case_dir = cases_dir / safe_case_name(index, item.source_path)
        case_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = case_dir / "sorter_stdout.log"
        status, seconds = run_sorter_one_file(
            project_root=project_root,
            source=item.source_path,
            output_dir=case_dir,
            brain=args.brain,
            timeout_sec=args.timeout_sec,
            stdout_path=stdout_path,
            copy_audio=args.copy_audio,
        )

        manifest_copy = case_dir / "Aaron_Sorted_Sounds_manifest.csv"
        row = read_first_manifest_row(manifest_copy)
        folder = row.get("folder_path", "") or row.get("placed_path", "")
        final_top = row.get("final_top", "") or (folder.split("/", 1)[0] if folder else "")
        reason = suspicious_reason(row, item.source_path, item.folder_key) if row else "no_manifest"
        result = [
            str(index),
            str(item.folder_index),
            str(item.file_index_in_folder),
            item.folder_key,
            status,
            str(item.source_path),
            item.source_path.name,
            final_top,
            folder,
            row.get("consensus_status", ""),
            parse_role(row),
            row.get("shape_vote", ""),
            row.get("shape_confidence", ""),
            reason,
            f"{seconds:.3f}",
            str(case_dir),
        ]
        results.append(result)
        if reason or status != "pass":
            suspicious.append(result)
        print(f"{index:04d}/{len(selected):04d} {status:8s} {item.source_path.name} => {folder or 'NO_MANIFEST'}")

    write_csv(report_dir / "random_folder_results.csv", header, results)
    write_csv(report_dir / "random_folder_suspicious.csv", header, suspicious)
    write_summary(
        report_dir=report_dir,
        sample_root=sample_root,
        selected=selected,
        results=results,
        suspicious=suspicious,
        args=args,
    )
    upload_zip = shutil.make_archive(str(report_dir / "random_folder_log_panel_upload_back"), "zip", report_dir)
    print(f"Report folder: {report_dir}")
    print(f"Upload-back logs zip: {upload_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
