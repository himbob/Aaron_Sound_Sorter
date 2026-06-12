#!/usr/bin/env python3
"""Hard-isolated percussion ZIP probe.

Runs one neutralized audio sample per external `timeout` process so decoder or
third-party DSP hangs cannot stop the sweep. Source paths are kept only in CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Import the existing neutral extraction/enumeration helpers.
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
import percussion_zip_batch_probe as base  # noqa: E402

FIELDS = base.FIELDS


def parse_json_result(stdout: str) -> dict[str, object] | None:
    for line in stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            try:
                return json.loads(line.split(":", 1)[1])
            except Exception:
                return None
    return None


def read_done(results_path: Path) -> set[str]:
    if not results_path.exists() or results_path.stat().st_size == 0:
        return set()
    with results_path.open(newline="", encoding="utf-8") as fh:
        return {row.get("neutral_file", "") for row in csv.DictReader(fh) if row.get("neutral_file")}


def classify_with_external_timeout(neutral: str, seconds: int, third_party_timeout: float) -> dict[str, object]:
    env = os.environ.copy()
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    if third_party_timeout >= 0:
        env["AARON_THIRD_PARTY_TIMEOUT_SECONDS"] = str(third_party_timeout)
    if os.environ.get("AARON_DISABLE_THIRD_PARTY_FEATURES"):
        env["AARON_DISABLE_THIRD_PARTY_FEATURES"] = os.environ["AARON_DISABLE_THIRD_PARTY_FEATURES"]
    else:
        env.pop("AARON_DISABLE_THIRD_PARTY_FEATURES", None)
    cmd = [
        "/usr/bin/timeout",
        "-k",
        "2s",
        f"{max(1, int(seconds))}s",
        sys.executable,
        str(Path(base.__file__).resolve()),
        "--classify-one",
        neutral,
    ]
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = round(time.time() - t0, 2)
    rec = parse_json_result(proc.stdout)
    if rec is None:
        msg = (proc.stderr or proc.stdout or f"external_timeout_exit_{proc.returncode}").strip().replace("\n", " ")
        return {
            "ok_broad": "False",
            "review": "False",
            "instrument_steal": "False",
            "seconds": elapsed,
            "error": msg[:500] if proc.returncode != 124 else f"external_timeout_after_{seconds}s",
        }
    rec.setdefault("seconds", elapsed)
    if proc.returncode != 0 and not rec.get("error"):
        rec["error"] = (proc.stderr or f"external_exit_{proc.returncode}").strip()[:500]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-start", type=int, required=True)
    ap.add_argument("--per-folder", type=int, default=5)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=35)
    ap.add_argument("--third-party-timeout", type=float, default=4.0)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.reset and args.results.exists():
        args.results.unlink()
    rows = base.extract_rows(base.selected_members(args.batch_start, args.per_folder))
    done = read_done(args.results)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.results.exists() or args.results.stat().st_size == 0
    count = 0
    with args.results.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for seq, row in enumerate(rows, 1):
            if args.max_rows is not None and count >= args.max_rows:
                break
            neutral = str(row["neutral_path"])
            if neutral in done:
                continue
            rec = {
                "seq": seq,
                "group": row["group"],
                "folder": row["folder"],
                "ordinal": row["ordinal"],
                "member": row["member"],
                "neutral_file": neutral,
            }
            out = classify_with_external_timeout(neutral, args.timeout, args.third_party_timeout)
            rec.update(out)
            writer.writerow(rec)
            fh.flush()
            print(
                f"{seq:04d} g{row['group']} ord={row['ordinal']} {Path(str(row['member'])).name:50s} -> {rec.get('path', '')} ok={rec.get('ok_broad')} err={rec.get('error', '')} sec={rec.get('seconds')}",
                flush=True,
            )
            count += 1
    base.summarize(args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
