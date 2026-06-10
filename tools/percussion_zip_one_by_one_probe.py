#!/usr/bin/env python3
"""One-file-at-a-time neutral percussion ZIP probe.

This wrapper uses the existing percussion_zip_batch_probe extraction and
--classify-one entrypoint, but runs every sample as an isolated subprocess.
It is slower than the in-process harness, but much safer when a decoder or
optional feature adapter can leave Python hanging after a row.
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

import percussion_zip_batch_probe as base

FIELDS = base.FIELDS


def _load_done(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    out = set()
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            neutral = row.get('neutral_file', '')
            if neutral:
                out.add(neutral)
    return out


def _classify_one(neutral: str, *, timeout: int, disable_third_party: bool) -> dict[str, object]:
    env = os.environ.copy()
    env.setdefault('NUMBA_DISABLE_JIT', '1')
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    env.setdefault('AARON_THIRD_PARTY_TIMEOUT_SECONDS', '5')
    if disable_third_party:
        env['AARON_DISABLE_THIRD_PARTY_FEATURES'] = '1'
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(base.__file__).resolve()), '--classify-one', neutral],
            cwd=str(base.PROJECT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(1, int(timeout)),
        )
    except subprocess.TimeoutExpired:
        return {
            'path': '', 'top': '', 'ok_broad': 'False', 'review': 'False',
            'instrument_steal': 'False', 'status': '', 'final_source': '',
            'shape': '', 'raw_consensus': '', 'raw_family': '',
            'seconds': round(time.time() - t0, 2),
            'error': f'child_timeout_after_{timeout}s',
        }
    record = None
    for line in proc.stdout.splitlines():
        if line.startswith('JSON_RESULT:'):
            try:
                record = json.loads(line.split(':', 1)[1])
            except Exception:
                record = None
    if record is None:
        err = (proc.stderr or proc.stdout or 'child produced no JSON_RESULT').strip().replace('\n', ' ')
        return {
            'path': '', 'top': '', 'ok_broad': 'False', 'review': 'False',
            'instrument_steal': 'False', 'status': '', 'final_source': '',
            'shape': '', 'raw_consensus': '', 'raw_family': '',
            'seconds': round(time.time() - t0, 2), 'error': err[:500],
        }
    record.setdefault('seconds', round(time.time() - t0, 2))
    if proc.returncode != 0 and not record.get('error'):
        record['error'] = (proc.stderr or f'child_exit_{proc.returncode}').strip()[:500]
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch-start', type=int, default=70)
    ap.add_argument('--per-folder', type=int, default=40)
    ap.add_argument('--timeout', type=int, default=30)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--reset', action='store_true')
    ap.add_argument('--disable-third-party', action='store_true')
    args = ap.parse_args()

    if args.reset and args.results.exists():
        args.results.unlink()
    rows = base.extract_rows(base.selected_members(args.batch_start, args.per_folder))
    args.results.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(args.results)
    write_header = not args.results.exists() or args.results.stat().st_size == 0
    with args.results.open('a', newline='', encoding='utf-8') as outfh:
        writer = csv.DictWriter(outfh, fieldnames=FIELDS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        for seq, row in enumerate(rows, 1):
            neutral = str(row['neutral_path'])
            if neutral in done:
                continue
            rec = {
                'seq': seq,
                'group': row['group'],
                'folder': row['folder'],
                'ordinal': row['ordinal'],
                'member': row['member'],
                'neutral_file': neutral,
            }
            rec.update(_classify_one(neutral, timeout=args.timeout, disable_third_party=args.disable_third_party))
            writer.writerow(rec)
            outfh.flush()
            print(f"{seq:04d} g{row['group']} {Path(str(row['member'])).name[:52]:52s} -> {rec.get('path','')} ({rec.get('seconds')}) ok={rec.get('ok_broad')} err={rec.get('error','')}", flush=True)
    base.summarize(args.results)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
