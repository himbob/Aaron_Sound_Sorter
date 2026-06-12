#!/usr/bin/env python3
"""Durable external audit harness for FX-oriented project ZIPs.

This tool is intentionally outside production routing. It may use source paths
for external audit expectations, but it never feeds names back into the sorter.
Each row is flushed after classification so a timeout does not erase progress.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Aaron_Sound_Sorter import build_sorter, declare_voters  # noqa: E402

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".au"}


def is_audio_member(name: str) -> bool:
    parts = [part for part in name.replace("\\", "/").split("/") if part]
    if not parts:
        return False
    if any(part == "__MACOSX" or part.startswith("._") or part.startswith(".") for part in parts):
        return False
    return Path(parts[-1]).suffix.lower() in AUDIO_EXTS


def external_expected_family(member: str) -> str:
    """Broad external audit hint only; never used by production classifier.

    The FX coverage ZIP is FX-titled but intentionally contains a few real
    instrument and drum folders from source packs.  This audit must distinguish
    true FX-source rows from those mixed-in source folders, otherwise a correct
    bass/vocal/key placement is falsely counted as an FX failure.
    """
    text = member.replace("\\", "/").lower()
    if any(token in text for token in ["/loop/drums/", "/one_shot/drums/", "/drum/", "/kick/", "/snares", "/snare"]):
        return "Drums"
    if any(token in text for token in ["/loop/bass/", "/bass/", "sub bass", "808"]):
        return "Instruments_or_FX_Sub"
    if any(token in text for token in ["/loop/vocals/", "/one_shot/vocals/", "/vocal", "/voice", "breath"]):
        return "Instruments_or_FX_Human"
    if any(
        token in text
        for token in [
            "/loop/synth",
            "/one_shot/synth",
            "/loop/keys",
            "/keys/",
            "/loop/guitars",
            "/guitars/",
            "/strings/",
            "/brass_woodwind",
            "/multi_instrument",
            "/pitched_percussion",
        ]
    ):
        return "Instruments_or_FX_Designed"
    if any(token in text for token in ["/fx/", "/sound_effect/", "/sfx", "fx_aaron", "sfx library"]):
        return "FX"
    return "FX"


def audit_source_group(expected_family: str) -> str:
    """Return the broad source bucket used only by this external audit."""
    if expected_family == "Drums":
        return "source_drums"
    if expected_family.startswith("Instruments_or_FX"):
        return "source_instrument_or_hybrid"
    return "source_fx"


def audit_status(member: str, folder_path: str, error_status: str = "") -> str:
    if error_status:
        return error_status
    top_raw = folder_path.split("/", 1)[0] if folder_path else "ERROR"
    top = top_raw.upper() if top_raw != "_TO_REVIEW" else "_TO_REVIEW"
    low = member.lower()
    expected = external_expected_family(member)

    if top == "FX":
        return "PASS_FX"
    if top == "TEXTURES":
        return "PASS_TEXTURE_ALLOWED"
    if expected == "Drums" and top == "DRUMS":
        return "PASS_DRUM_SOURCE"
    if expected.startswith("Instruments_or_FX") and top == "INSTRUMENTS":
        return "PASS_INSTRUMENT_SOURCE"
    if expected.startswith("Instruments_or_FX") and top == "DRUMS":
        return "QUESTIONABLE_DRUM_IN_SOURCE_INSTRUMENT"
    if top == "DRUMS":
        if any(
            token in low
            for token in [
                "/drums/",
                "/drum",
                "/kick/",
                "/snares",
                "thump",
                "hit",
                "impact",
                "tray",
                "rattle",
                "crash",
                "bell",
                "body",
                "plate",
                "punch",
                "snare",
            ]
        ):
            return "PASS_DRUM_ALLOWED"
        return "QUESTIONABLE_DRUM"
    if top == "_TO_REVIEW":
        return "FAIL_REVIEW"
    if top == "INSTRUMENTS":
        return "FAIL_INSTRUMENT"
    return "FAIL_OTHER"


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "zip_member",
        "expected_external_audit",
        "source_group",
        "folder_path",
        "top_family",
        "audit_status",
        "seconds",
        "reason",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_brains(project_root: Path) -> tuple[dict, dict]:
    brain = json.loads((project_root / "stage4_folder_brain.json").read_text(encoding="utf-8"))
    babies: dict[str, dict | None] = {}
    for lane, filename in [
        ("core_baby", "stage4_folder_brain_core_baby.json"),
        ("spread_baby", "stage4_folder_brain_spread_baby.json"),
        ("outlier_baby", "stage4_folder_brain_outlier_baby.json"),
    ]:
        path = project_root / filename
        babies[lane] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return brain, babies


def main() -> int:
    parser = argparse.ArgumentParser(description="External FX ZIP audit harness.")
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--start", type=int, default=1, help="1-based first sorted audio member ordinal")
    parser.add_argument("--limit", type=int, default=0, help="number of rows to attempt; 0 means all")
    parser.add_argument("--candidate-count", type=int, default=40)
    parser.add_argument(
        "--disable-third-party",
        action="store_true",
        help="disable optional 3rd-party feature adapter for speed/debugging",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    if args.disable_third_party:
        os.environ["AARON_DISABLE_THIRD_PARTY_FEATURES"] = "1"
    os.environ.setdefault("AARON_THIRD_PARTY_TIMEOUT_SECONDS", "8")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    with zipfile.ZipFile(args.zip, "r") as zip_file:
        members = sorted(
            info.filename for info in zip_file.infolist() if not info.is_dir() and is_audio_member(info.filename)
        )
        selected = members[max(0, args.start - 1) :]
        if args.limit > 0:
            selected = selected[: args.limit]
        if not selected:
            raise SystemExit("No selected audio members")

        existing = read_existing_rows(args.out)
        rows = list(existing)
        done = {int(row["index"]) for row in rows if str(row.get("index", "")).isdigit()}

        with tempfile.TemporaryDirectory(prefix="aaron_fx_zip_probe_") as temp_dir:
            input_root = Path(temp_dir) / "input"
            for _ordinal, member in enumerate(members, start=1):
                if member not in selected:
                    continue
                target = input_root / member
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zip_file.read(member))

            brain, babies = load_brains(project_root)
            sorter = build_sorter(
                declare_voters(candidate_count=args.candidate_count), candidate_count=args.candidate_count
            )

            for member in selected:
                ordinal = members.index(member) + 1
                if ordinal in done:
                    continue
                sample_path = input_root / member
                started = time.monotonic()
                row = {
                    "index": str(ordinal),
                    "zip_member": member,
                    "expected_external_audit": external_expected_family(member),
                    "source_group": audit_source_group(external_expected_family(member)),
                    "folder_path": "",
                    "top_family": "",
                    "audit_status": "ERROR",
                    "seconds": "",
                    "reason": "",
                    "error": "",
                }
                try:
                    result = sorter.classify_one_file(sample_path, brain, babies, {}, use_baby_brains_in_sort=True)
                    folder_path = result.decision.folder_path
                    top_family = folder_path.split("/", 1)[0] if folder_path else ""
                    expected_family = external_expected_family(member)
                    row.update(
                        expected_external_audit=expected_family,
                        source_group=audit_source_group(expected_family),
                        folder_path=folder_path,
                        top_family=top_family,
                        reason=result.decision.reason,
                        audit_status=audit_status(member, folder_path),
                    )
                except Exception as error:  # pragma: no cover - external harness evidence path
                    row.update(
                        audit_status="ERROR",
                        error=repr(error) + "\n" + traceback.format_exc(limit=8),
                    )
                row["seconds"] = f"{time.monotonic() - started:.3f}"
                rows.append(row)
                write_rows(args.out, rows)
                print(
                    f"{ordinal:03d}/{len(members):03d} {row['audit_status']:24s} {member} => {row['folder_path']}",
                    flush=True,
                )

    counts = Counter(row.get("audit_status", "") for row in rows)
    tops = Counter(row.get("top_family", "") for row in rows)
    source_groups = Counter(row.get("source_group", "") for row in rows)
    true_fx_rows = [row for row in rows if row.get("source_group") == "source_fx"]
    true_fx_counts = Counter(row.get("audit_status", "") for row in true_fx_rows)
    summary_path = args.out.with_suffix(".summary.txt")
    summary_path.write_text(
        "FX ZIP probe summary\n"
        f"zip={args.zip}\n"
        f"rows={len(rows)}\n"
        f"status_counts={dict(counts)}\n"
        f"top_counts={dict(tops)}\n"
        f"source_group_counts={dict(source_groups)}\n"
        f"true_fx_status_counts={dict(true_fx_counts)}\n",
        encoding="utf-8",
    )
    print(summary_path.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
