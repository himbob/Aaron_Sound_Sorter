#!/usr/bin/env python3
"""Build a local failure review pack from random/single-file log-panel results.

Default mode creates a small local review folder using symlinks to the original
audio files. That is best for listening on Aaron's Mac because it avoids copying
large sample files.

Optional mode copies audio into the pack for upload/testing:
  --copy-audio --max-audio-files 20

The generated ZIP is logs + symlinks by default. Symlinks work locally on Aaron's
Mac but will not give another machine access to the audio targets. Use copied mode
only for a small curated mini-pack when the next AI needs to test real files.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

csv.field_size_limit(sys.maxsize)

DEFAULT_REPORT_ROOTS = [
    Path("reports/random_sample_folder_log_panel"),
    Path("reports/single_file_log_panel"),
    Path("_reports/test_runner"),
]


def safe_slug(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return text[:max_len] or "item"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def newest_existing_report(project_root: Path) -> Path:
    candidates: list[Path] = []
    for root_rel in DEFAULT_REPORT_ROOTS:
        root = project_root / root_rel
        if root.exists():
            candidates.extend([p for p in root.glob("run_*") if p.is_dir()])
    if not candidates:
        raise SystemExit("Could not find a report run folder. Pass --report-dir or --report-zip.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_report_zip(report_zip: Path, work_dir: Path) -> Path:
    extract_dir = work_dir / "extracted_report"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(report_zip, "r") as zip_file:
        zip_file.extractall(extract_dir)
    # Most upload-back zips contain files at root. Some may contain one run folder.
    direct_results = list(extract_dir.glob("*results.csv"))
    if direct_results:
        return extract_dir
    nested_results = list(extract_dir.rglob("*results.csv"))
    if nested_results:
        return nested_results[0].parent
    raise SystemExit(f"No results CSV found inside report zip: {report_zip}")


def find_report_dir(args: argparse.Namespace, project_root: Path, work_dir: Path) -> Path:
    if args.report_zip:
        return extract_report_zip(Path(args.report_zip).expanduser().resolve(), work_dir)
    if args.report_dir:
        return Path(args.report_dir).expanduser().resolve()
    return newest_existing_report(project_root)


def find_results_csv(report_dir: Path) -> Path:
    names = [
        "random_folder_results.csv",
        "single_file_results.csv",
        "test_runner_results.csv",
    ]
    for name in names:
        path = report_dir / name
        if path.exists():
            return path
    matches = sorted(report_dir.glob("*results.csv"))
    if matches:
        return matches[0]
    matches = sorted(report_dir.rglob("*results.csv"))
    if matches:
        return matches[0]
    raise SystemExit(f"No results CSV found in {report_dir}")


def find_suspicious_csv(report_dir: Path) -> Path | None:
    names = [
        "random_folder_suspicious.csv",
        "single_file_suspicious.csv",
        "name_hint_mismatches.csv",
    ]
    for name in names:
        path = report_dir / name
        if path.exists():
            return path
    matches = sorted(report_dir.glob("*suspicious*.csv"))
    if matches:
        return matches[0]
    return None


def source_path_for_row(row: dict[str, str]) -> Path | None:
    for key in ["source_path", "original_path", "input_path", "original_input_path"]:
        value = row.get(key, "").strip()
        if value:
            return Path(value)
    return None


def final_folder_for_row(row: dict[str, str]) -> str:
    for key in ["folder_path", "new_relative_path", "placed_path", "output_path"]:
        value = row.get(key, "").strip()
        if value:
            return value
    top = row.get("final_top", "").strip()
    return top


def text_for_row(row: dict[str, str]) -> str:
    bits = [
        row.get("folder_key", ""),
        row.get("source_name", ""),
        str(source_path_for_row(row) or ""),
        final_folder_for_row(row),
        row.get("suspicious_reason", ""),
        row.get("detected_parent_role", ""),
        row.get("shape_vote", ""),
    ]
    return " ".join(bits).lower()


def category_for_row(row: dict[str, str]) -> str:
    text = text_for_row(row)
    folder = final_folder_for_row(row)

    if "akfw" in text or "akwf" in text or "wavetable" in text:
        return "expected_wavetable_or_tiny"

    reason = row.get("suspicious_reason", "").lower()
    if "pitched_instrument_source_to_drums" in reason:
        return "instrument_one_shot_to_drums"
    if "non_bass_instrument_source_to_bass_loop" in reason:
        return "non_bass_to_bass_loop"
    if "mixed_or_non_woodwind_source_to_brass_woodwinds" in reason:
        return "mixed_or_vocal_to_brass_woodwinds"
    if "non_vocal_source_to_human_voice_fx" in reason:
        return "non_vocal_to_human_voice_fx"
    if "non_alarm_source_to_alarm" in reason:
        return "non_alarm_to_alarm_fx"
    if "instrument_source_to_fx" in reason:
        return "instrument_to_fx"
    if "source_hint_expected" in reason and "_got_drums" in reason:
        return "source_hint_to_wrong_drums"
    if "source_hint_expected" in reason and "_got_fx" in reason:
        return "source_hint_to_wrong_fx"
    if "source_hint_expected" in reason:
        return "source_hint_wrong_top"

    if "Human and Voice FX" in folder and not re.search(r"vocal|voice|vox|shout|rap|choir|breath|mouth", text):
        return "non_vocal_to_human_voice_fx"
    if "Bass Loops" in folder and re.search(
        r"rhodes|piano|keys|organ|synth|lead|pad|guitar|string|sax|brass|woodwind|vocal|voice|vox", text
    ):
        return "non_bass_to_bass_loop"
    if "Brass and Woodwinds" in folder and re.search(
        r"melody|mixed|kit|rhodes|piano|keys|organ|bell|chime|vocal|voice|vox|guitar|string|synth", text
    ):
        return "mixed_or_vocal_to_brass_woodwinds"
    if folder.startswith("Drums") and re.search(
        r"guitar|strum|chord|keys|organ|piano|rhodes|synth|vocal|voice|vox|string|sax|brass|woodwind", text
    ):
        return "instrument_or_vocal_to_drums"
    if folder.startswith("_TO_REVIEW"):
        return "review"

    return "other_suspicious_or_selected"


def copy_case_logs(row: dict[str, str], pack_case_dir: Path, report_dir: Path) -> None:
    case_report = Path(row.get("case_report_dir", "") or "")
    candidates = []
    if case_report.exists():
        candidates.append(case_report)
    case_name = row.get("case_report_dir", "").split("/")[-1]
    if case_name:
        candidates.extend(report_dir.rglob(case_name))
    for src_dir in candidates:
        if not src_dir.exists() or not src_dir.is_dir():
            continue
        dst = pack_case_dir / "_logs"
        if dst.exists():
            return
        ignore = shutil.ignore_patterns("*.wav", "*.aif", "*.aiff", "*.flac", "*.ogg", "*.au")
        shutil.copytree(src_dir, dst, ignore=ignore)
        return


def link_or_copy_audio(source: Path, dest: Path, copy_audio: bool) -> tuple[bool, str]:
    if not source.exists():
        return False, "missing_source"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        if copy_audio:
            shutil.copy2(source, dest)
            return True, "copied"
        os.symlink(str(source), str(dest))
        return True, "symlinked"
    except Exception as error:
        if not copy_audio:
            try:
                shutil.copy2(source, dest)
                return True, "copied_fallback"
            except Exception as copy_error:
                return False, f"link_error={error};copy_error={copy_error}"
        return False, str(error)


def select_rows(
    all_rows: list[dict[str, str]],
    suspicious_rows: list[dict[str, str]],
    include_all: bool,
    max_cases: int,
    include_reviews: bool,
) -> list[dict[str, str]]:
    rows = all_rows if include_all else suspicious_rows
    if include_reviews and not include_all:
        seen = {row.get("index", "") + row.get("source_path", "") for row in rows}
        for row in all_rows:
            if final_folder_for_row(row).startswith("_TO_REVIEW"):
                key = row.get("index", "") + row.get("source_path", "")
                if key not in seen:
                    rows.append(row)
                    seen.add(key)
    if max_cases > 0:
        return rows[:max_cases]
    return rows


def make_symlink_zip(pack_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    zip_bin = shutil.which("zip")
    if zip_bin:
        subprocess.run(
            [zip_bin, "-qry", str(zip_path), pack_dir.name],
            cwd=pack_dir.parent,
            check=True,
        )
        return

    # Fallback: Python zip does not preserve symlinks correctly on all systems.
    # This fallback stores link target text as .symlink_target files.
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in pack_dir.rglob("*"):
            rel = path.relative_to(pack_dir.parent)
            if path.is_symlink():
                zip_file.writestr(str(rel) + ".symlink_target", os.readlink(path))
            elif path.is_file():
                zip_file.write(path, rel)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--report-zip", type=Path)
    parser.add_argument("--out-root", type=Path, default=Path("reports/failure_review_packs"))
    parser.add_argument("--max-cases", type=int, default=80)
    parser.add_argument("--copy-audio", action="store_true")
    parser.add_argument("--max-audio-files", type=int, default=20)
    parser.add_argument("--include-all", action="store_true", help="Include every result row, not just suspicious rows")
    parser.add_argument("--include-reviews", action="store_true", help="Also include _TO_REVIEW rows")
    parser.add_argument("--only-category", default="", help="Only include one derived category")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="aaron_review_pack_report_") as tmp:
        report_dir = find_report_dir(args, project_root, Path(tmp))
        results_csv = find_results_csv(report_dir)
        suspicious_csv = find_suspicious_csv(report_dir)

        all_rows = read_csv(results_csv)
        suspicious_rows = (
            read_csv(suspicious_csv)
            if suspicious_csv
            else [row for row in all_rows if row.get("suspicious_reason", "").strip()]
        )
        selected_rows = select_rows(
            all_rows=all_rows,
            suspicious_rows=suspicious_rows,
            include_all=args.include_all,
            max_cases=args.max_cases,
            include_reviews=args.include_reviews,
        )

        if args.only_category:
            selected_rows = [row for row in selected_rows if category_for_row(row) == args.only_category]

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pack_root = (project_root / args.out_root / f"run_{stamp}").resolve()
        pack_dir = pack_root / "Aaron_Failure_Review_Pack"
        pack_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows: list[dict[str, str]] = []
        copied_audio_count = 0

        for n, row in enumerate(selected_rows, start=1):
            src = source_path_for_row(row)
            category = category_for_row(row)
            name = row.get("source_name", "") or (src.name if src else f"case_{n}")
            case_dir = pack_dir / category / f"{n:04d}_{safe_slug(name)}"
            case_dir.mkdir(parents=True, exist_ok=True)
            copy_case_logs(row, case_dir, report_dir)

            audio_status = "no_source_path"
            audio_rel = ""
            if src:
                audio_dest = case_dir / "audio" / src.name
                copy_this_audio = args.copy_audio and copied_audio_count < max(0, args.max_audio_files)
                ok, audio_status = link_or_copy_audio(src, audio_dest, copy_this_audio)
                if ok and copy_this_audio:
                    copied_audio_count += 1
                if ok:
                    audio_rel = str(audio_dest.relative_to(pack_dir))

            note = (
                f"Source: {src or ''}\n"
                f"Folder key: {row.get('folder_key', '')}\n"
                f"Placed: {final_folder_for_row(row)}\n"
                f"Consensus: {row.get('consensus_status', '')}\n"
                f"Role: {row.get('detected_parent_role', '')}\n"
                f"Shape: {row.get('shape_vote', '')} {row.get('shape_confidence', '')}\n"
                f"Suspicious reason: {row.get('suspicious_reason', '')}\n"
                f"Derived category: {category}\n"
                f"Audio status: {audio_status}\n"
            )
            (case_dir / "case_notes.txt").write_text(note, encoding="utf-8")

            out = dict(row)
            out["review_category"] = category
            out["audio_status"] = audio_status
            out["review_pack_audio_path"] = audio_rel
            manifest_rows.append(out)

        fields = sorted({key for row in manifest_rows for key in row})
        preferred = [
            "review_category",
            "audio_status",
            "review_pack_audio_path",
            "index",
            "folder_key",
            "source_path",
            "source_name",
            "final_top",
            "folder_path",
            "consensus_status",
            "detected_parent_role",
            "shape_vote",
            "shape_confidence",
            "suspicious_reason",
        ]
        fieldnames = preferred + [f for f in fields if f not in preferred]
        write_csv(pack_dir / "failure_review_pack_manifest.csv", manifest_rows, fieldnames)

        summary_lines = [
            f"Report dir: {report_dir}",
            f"Results CSV: {results_csv}",
            f"Suspicious CSV: {suspicious_csv or ''}",
            f"Selected rows: {len(selected_rows)}",
            f"Copy audio: {args.copy_audio}",
            f"Copied audio count: {copied_audio_count}",
            "",
            "Category counts:",
        ]
        counts: dict[str, int] = {}
        for row in manifest_rows:
            counts[row["review_category"]] = counts.get(row["review_category"], 0) + 1
        for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            summary_lines.append(f"{count:5d}  {category}")
        (pack_dir / "failure_review_pack_summary.txt").write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        zip_path = pack_root / (
            "failure_review_pack_with_audio.zip" if args.copy_audio else "failure_review_pack_symlinks.zip"
        )
        if not args.no_zip:
            make_symlink_zip(pack_dir, zip_path)

        print(f"Review pack folder: {pack_dir}")
        if not args.no_zip:
            print(f"Review pack zip:    {zip_path}")
        print(f"Copied audio files: {copied_audio_count}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
