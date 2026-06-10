#!/usr/bin/env python3
"""Repeatable neutral-name batch probe for one_shot_percussive_sounds.zip.

The tool is intentionally source-name blind during classification: files are
extracted to neutral names, and producer/source paths are kept only in the CSV
for external audit.  Each selected slice writes its own CSV immediately, and the
optional child-process mode records timeouts instead of losing the whole run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
import shutil
from collections import defaultdict
from pathlib import Path

AUDIO_EXT = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".au"}
PROJECT = Path(os.environ.get("AARON_SORTER_PROJECT", "/mnt/data/work_current/Aaron_Sound_Sorter"))
ZIP = Path(os.environ.get("PERCUSSION_ZIP", "/mnt/data/one_shot_percussive_sounds.zip"))
ROOT = Path(os.environ.get("PERC_SYSTEM_ROOT", "/mnt/data/perc_system_run"))
EXTRACT = ROOT / "neutral_extract"
MAP = ROOT / "member_map.csv"
DEFAULT_RESULTS = ROOT / "results.csv"

sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))


class Timeout(Exception):
    pass


def handler(signum, frame):
    raise Timeout("sample timeout")


if hasattr(signal, "SIGALRM"):
    signal.signal(signal.SIGALRM, handler)


def clean_member(member_name: str) -> bool:
    parts = [p for p in member_name.replace("\\", "/").split("/") if p]
    return bool(
        parts
        and "__MACOSX" not in parts
        and not any(p.startswith("._") or p.startswith(".") for p in parts)
        and Path(parts[-1]).suffix.lower() in AUDIO_EXT
    )


def enumerate_members() -> dict[str, list[str]]:
    by: dict[str, list[str]] = defaultdict(list)
    with zipfile.ZipFile(ZIP) as z:
        for info in z.infolist():
            if info.is_dir() or not clean_member(info.filename):
                continue
            folder = str(Path(info.filename).parent)
            by[folder].append(info.filename)
    for key in by:
        by[key].sort()
    return dict(sorted(by.items()))


def selected_members(batch_start: int, per_folder: int) -> list[tuple[int, str, int, str]]:
    by = enumerate_members()
    rows: list[tuple[int, str, int, str]] = []
    for group, (folder, members) in enumerate(by.items(), 1):
        for ordinal, member in enumerate(members[batch_start : batch_start + per_folder], batch_start + 1):
            rows.append((group, folder, ordinal, member))
    return rows


def default_results_path(batch_start: int, per_folder: int) -> Path:
    return ROOT / f"results_offset_{batch_start:05d}_per_folder_{per_folder:03d}.csv"


def extract_rows(rows: list[tuple[int, str, int, str]], force: bool = False) -> list[dict[str, object]]:
    EXTRACT.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, object]] = []
    with zipfile.ZipFile(ZIP) as z:
        for group, folder, ordinal, member in rows:
            ext = Path(member).suffix.lower() or ".wav"
            dest = EXTRACT / f"g{group:02d}" / f"neutral_g{group:02d}_{ordinal:05d}{ext}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if force or not dest.exists() or dest.stat().st_size == 0:
                with z.open(member) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
            rows_out.append(
                {
                    "group": group,
                    "folder": folder,
                    "ordinal": ordinal,
                    "member": member,
                    "neutral_path": str(dest),
                }
            )
    ROOT.mkdir(parents=True, exist_ok=True)
    with MAP.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["group", "folder", "ordinal", "member", "neutral_path"])
        writer.writeheader()
        writer.writerows(rows_out)
    return rows_out


def load_sorter():
    from Aaron_Sound_Sorter import build_sorter, declare_voters
    from aaron_sound_sorter.infrastructure.brain_repository import BrainRepository

    repo = BrainRepository()
    brain = repo.load(PROJECT / "stage4_folder_brain.json")
    baby = {}
    for lane, name in [
        ("core_baby", "stage4_folder_brain_core_baby.json"),
        ("spread_baby", "stage4_folder_brain_spread_baby.json"),
        ("outlier_baby", "stage4_folder_brain_outlier_baby.json"),
    ]:
        path = PROJECT / name
        baby[lane] = repo.load(path) if path.exists() else None
    sorter = build_sorter(declare_voters(candidate_count=100), candidate_count=100)
    return sorter, brain, baby


def classify_path_once(path: Path) -> dict[str, object]:
    sorter, brain, baby = load_sorter()
    t0 = time.time()
    result = sorter.classify_one_file(
        path,
        brain,
        baby,
        {},
        use_baby_brains_in_sort=True,
        use_harmonic_brains_in_sort=False,
    )
    decision = result.decision
    folder_path = getattr(decision, "folder_path", "") or getattr(decision, "final_label", "") or ""
    evidence = getattr(result.facts, "evidence", {}) or {}
    auth = evidence.get("authority_trace", {}) if isinstance(evidence.get("authority_trace", {}), dict) else {}
    shape = evidence.get("shape_vote", {}) if isinstance(evidence.get("shape_vote", {}), dict) else {}
    raw = evidence.get("raw_consensus_claim", {}) if isinstance(evidence.get("raw_consensus_claim", {}), dict) else {}
    return {
        "path": folder_path,
        "top": folder_path.split("/")[0] if folder_path else "",
        "ok_broad": str(folder_path.startswith("Drums/") or folder_path.startswith("FX/")),
        "review": str(folder_path.startswith("_TO_REVIEW")),
        "instrument_steal": str(folder_path.startswith("Instruments/")),
        "status": getattr(decision, "consensus_status", ""),
        "final_source": auth.get("final_source", ""),
        "shape": shape.get("primary_shape", shape.get("shape", "")),
        "raw_consensus": raw.get("path", "") or raw.get("label", "") or "",
        "raw_family": raw.get("family", ""),
        "seconds": round(time.time() - t0, 2),
        "error": "",
    }


def classify_one_child_main(neutral_path: str) -> int:
    try:
        record = classify_path_once(Path(neutral_path))
        print("JSON_RESULT:" + json.dumps(record, sort_keys=True), flush=True)
        return 0
    except Exception as exc:  # pragma: no cover - exercised by subprocess harness
        print("JSON_RESULT:" + json.dumps({"error": repr(exc), "ok_broad": "False"}, sort_keys=True), flush=True)
        return 1


FIELDS = [
    "seq",
    "group",
    "folder",
    "ordinal",
    "member",
    "neutral_file",
    "path",
    "top",
    "ok_broad",
    "review",
    "instrument_steal",
    "status",
    "final_source",
    "shape",
    "raw_consensus",
    "raw_family",
    "seconds",
    "error",
]


def read_done_neutral(results_path: Path) -> set[str]:
    done: set[str] = set()
    if not results_path.exists() or results_path.stat().st_size == 0:
        return done
    with results_path.open(newline="", encoding="utf-8") as file_handle:
        for row in csv.DictReader(file_handle):
            neutral = row.get("neutral_file", "")
            if neutral:
                done.add(neutral)
    return done


def child_classify(row: dict[str, object], timeout: int) -> dict[str, object]:
    neutral = str(row["neutral_path"])
    env = os.environ.copy()
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--classify-one", neutral],
        cwd=str(PROJECT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=max(1, int(timeout)))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        # Do not wait for decoder/numba cleanup after a hard timeout; record
        # the failure and let the next sample continue from the CSV ledger.
        return {"ok_broad": "False", "review": "False", "instrument_steal": "False", "seconds": round(time.time() - t0, 2), "error": f"child_timeout_after_{timeout}s"}
    completed_returncode = proc.returncode
    record: dict[str, object] | None = None
    for line in stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            try:
                record = json.loads(line.split(":", 1)[1])
            except Exception:
                record = None
    if record is None:
        err = (stderr or stdout or "child produced no JSON_RESULT").strip().replace("\n", " ")
        return {"ok_broad": "False", "review": "False", "instrument_steal": "False", "seconds": round(time.time() - t0, 2), "error": err[:500]}
    record.setdefault("seconds", round(time.time() - t0, 2))
    if completed_returncode != 0 and not record.get("error"):
        record["error"] = (stderr or f"child_exit_{completed_returncode}").strip()[:500]
    return record


def classify_rows(
    rows: list[dict[str, object]],
    *,
    results_path: Path,
    start_at: int = 0,
    max_rows: int | None = None,
    timeout: int = 45,
    child_process: bool = False,
) -> None:
    sorter = brain = baby = None
    if not child_process:
        sorter, brain, baby = load_sorter()
    done_neutral = read_done_neutral(results_path)
    write_header = not results_path.exists() or results_path.stat().st_size == 0
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", newline="", encoding="utf-8") as outfh:
        writer = csv.DictWriter(outfh, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        count = 0
        for seq, row in enumerate(rows, 1):
            if seq <= start_at:
                continue
            if max_rows is not None and count >= max_rows:
                break
            neutral = str(row["neutral_path"])
            if neutral in done_neutral:
                continue
            rec: dict[str, object] = {
                "seq": seq,
                "group": row["group"],
                "folder": row["folder"],
                "ordinal": row["ordinal"],
                "member": row["member"],
                "neutral_file": neutral,
            }
            t0 = time.time()
            try:
                if child_process:
                    rec.update(child_classify(row, timeout))
                else:
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(timeout)
                    rec.update(classify_path_once(Path(neutral)))
                    if hasattr(signal, "SIGALRM"):
                        signal.alarm(0)
                print(
                    f"{seq:04d} g{row['group']} {Path(str(row['member'])).name[:55]:55s} -> {rec.get('path','')} ({rec.get('seconds', round(time.time()-t0,2))}s) ok={rec.get('ok_broad')}",
                    flush=True,
                )
            except Exception as exc:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
                rec.update({"path": "", "top": "", "ok_broad": "False", "review": "False", "instrument_steal": "False", "seconds": round(time.time() - t0, 2), "error": repr(exc)})
                print(f"ERR {seq:04d} g{row['group']} {Path(str(row['member'])).name[:55]:55s} {repr(exc)}", flush=True)
            writer.writerow(rec)
            outfh.flush()
            count += 1


def summarize(results_path: Path) -> dict[str, int]:
    if not results_path.exists() or results_path.stat().st_size == 0:
        return {}
    rows = list(csv.DictReader(results_path.open(newline="", encoding="utf-8")))
    bad = [row for row in rows if row.get("ok_broad") != "True"]
    review = [row for row in rows if row.get("review") == "True"]
    inst = [row for row in rows if row.get("instrument_steal") == "True"]
    drums = [row for row in rows if row.get("top") == "Drums"]
    fx = [row for row in rows if row.get("top") == "FX"]
    print("\nSUMMARY")
    print("results", results_path)
    print("tested", len(rows), "drums", len(drums), "fx", len(fx), "bad", len(bad), "review", len(review), "instrument", len(inst))
    by: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    for row in rows:
        bucket = by[row["folder"]]
        bucket[0] += 1
        bucket[1] += row.get("top") == "Drums"
        bucket[2] += row.get("top") == "FX"
        bucket[3] += row.get("review") == "True"
        bucket[4] += row.get("instrument_steal") == "True"
    for folder, counts in by.items():
        print(counts, folder)
    if bad:
        print("\nBAD FIRST 30")
        for row in bad[:30]:
            print(row["seq"], row["member"], "=>", row.get("path"), row.get("error"), "shape", row.get("shape"), "source", row.get("final_source"))
    return {"tested": len(rows), "drums": len(drums), "fx": len(fx), "bad": len(bad), "review": len(review), "instrument": len(inst)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-start", type=int, default=0, help="0-based offset within each source folder")
    parser.add_argument("--per-folder", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--start-at", type=int, default=0, help="1-based selected-row index to skip before classifying")
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--child-process", action="store_true", help="classify each sample in a child process so hard hangs are recorded")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--classify-one", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.classify_one:
        return classify_one_child_main(args.classify_one)
    ROOT.mkdir(parents=True, exist_ok=True)
    results_path = args.results or default_results_path(args.batch_start, args.per_folder)
    if args.reset and results_path.exists():
        results_path.unlink()
    rows = extract_rows(selected_members(args.batch_start, args.per_folder))
    print(
        "selected",
        len(rows),
        "from",
        len({row["folder"] for row in rows}),
        "folders",
        "results",
        results_path,
        "child_process",
        args.child_process,
        flush=True,
    )
    classify_rows(
        rows,
        results_path=results_path,
        start_at=args.start_at,
        max_rows=args.max_rows,
        timeout=args.timeout,
        child_process=args.child_process,
    )
    summarize(results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
