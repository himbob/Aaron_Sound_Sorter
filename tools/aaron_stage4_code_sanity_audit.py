#!/usr/bin/env python3
"""Report-only sanity audit for Stage 4 main code and latest run reports.

This does not change training data and does not tune thresholds. It checks for:
  - heavy unsafe upload ZIPs
  - LONG FX leaking into learned/report labels
  - malformed top-level structure marker folders
  - FX loop folders
  - whether source mentions _LONG_FX support at all
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

BAD_TOP_STRUCTURE_RE = re.compile(r"(^|/)(Drums|Instruments|FX|Textures)/(_ONE_SHOTS|_LOOPS|_LONG_FX)(/|$)")
FX_LOOP_RE = re.compile(r"(^|/)FX/.*/_LOOPS(/|$)")
LONG_FX_TEXT_RE = re.compile(r"LONG[ _-]?FX", re.IGNORECASE)


def project_root_from_arg(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "reports").exists():
        return cwd
    if cwd.name == "commands" and (cwd.parent / "reports").exists():
        return cwd.parent
    return Path(
        "/Users/aaron/Documents/Codex/2026-04-25/files-mentioned-by-the-user-create/Aaron_Sound_Sorter"
    ).resolve()


def latest_run(project_root: Path) -> Path | None:
    base = project_root / "reports" / "stage4_big_real_reviews"
    runs = sorted((p for p in base.glob("run_*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    try:
        data = path.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def scan_source(project_root: Path) -> dict[str, Any]:
    src = project_root / "src" / "aaron_stage4" / "phase3_pure_brain_lab.py"
    result: dict[str, Any] = {"source_path": str(src), "exists": src.exists()}
    if not src.exists():
        return result
    text = read_text(src, max_bytes=10_000_000)
    result.update(
        {
            "mentions__LONG_FX": "_LONG_FX" in text,
            "mentions_long_fx_structure_literal": "long_fx" in text,
            "mentions_long_fx_as_one_shot_warning": "LONG_FX" in text and "one_shot" in text,
            "phase3_pure_brain_lab_size_bytes": file_size(src),
        }
    )
    return result


def scan_uploads(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for z in sorted(run_dir.rglob("*.zip")):
        rows.append({"path": str(z), "size_bytes": file_size(z), "size_mb": round(file_size(z) / (1024 * 1024), 2)})
    return rows


def scan_report_texts(run_dir: Path) -> dict[str, Any]:
    long_fx_hits: list[str] = []
    malformed_hits: list[str] = []
    fx_loop_hits: list[str] = []
    checked = 0
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() not in {".csv", ".json", ".jsonl", ".txt", ".md", ".log", ".out", ".err"}:
            continue
        if file_size(path) > 20_000_000:
            continue
        checked += 1
        text = read_text(path, max_bytes=20_000_000)
        rel = str(path.relative_to(run_dir))
        if LONG_FX_TEXT_RE.search(text):
            # LONG_FX in notes can be OK, but this flags it for review.
            long_fx_hits.append(rel)
        if BAD_TOP_STRUCTURE_RE.search(text):
            malformed_hits.append(rel)
        if FX_LOOP_RE.search(text):
            fx_loop_hits.append(rel)
    return {
        "text_report_files_checked": checked,
        "long_fx_text_hit_files": long_fx_hits[:100],
        "long_fx_text_hit_count": len(long_fx_hits),
        "malformed_top_structure_hit_files": malformed_hits[:100],
        "malformed_top_structure_hit_count": len(malformed_hits),
        "fx_loop_hit_files": fx_loop_hits[:100],
        "fx_loop_hit_count": len(fx_loop_hits),
    }


def summarize_preview_manifest(run_dir: Path) -> dict[str, Any]:
    mf = run_dir / "rebuild_and_real_preview" / "reports" / "real_sort_preview_manifest.csv"
    if not mf.exists():
        return {"real_sort_preview_manifest_exists": False}
    counts: dict[str, int] = {}
    raw_counts: dict[str, int] = {}
    try:
        with mf.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            for row in csv.DictReader(fh):
                final_top = row.get("final_top") or row.get("top") or ""
                raw_top = row.get("raw_predicted_top") or ""
                if final_top:
                    counts[final_top] = counts.get(final_top, 0) + 1
                if raw_top:
                    raw_counts[raw_top] = raw_counts.get(raw_top, 0) + 1
    except Exception as exc:
        return {"real_sort_preview_manifest_exists": True, "read_error": str(exc)}
    return {
        "real_sort_preview_manifest_exists": True,
        "final_top_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "raw_top_counts": dict(sorted(raw_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--run", default="latest")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    project_root = project_root_from_arg(args.project_root)
    if args.run == "latest":
        run_dir = latest_run(project_root)
        if run_dir is None:
            raise SystemExit("No Stage 4 run folders found")
    else:
        p = Path(args.run).expanduser()
        run_dir = (
            p.resolve() if p.exists() else (project_root / "reports" / "stage4_big_real_reviews" / args.run).resolve()
        )
    out_dir = Path(args.out).expanduser().resolve() if args.out else (project_root / "reports" / "stage4_code_sanity")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "source_audit": scan_source(project_root),
        "upload_zips": scan_uploads(run_dir) if run_dir.exists() else [],
        "report_text_audit": scan_report_texts(run_dir) if run_dir.exists() else {},
        "real_preview_summary": summarize_preview_manifest(run_dir) if run_dir.exists() else {},
    }
    json_path = out_dir / f"stage4_code_sanity_{stamp}.json"
    txt_path = out_dir / f"stage4_code_sanity_{stamp}.txt"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "Aaron Stage 4 code/run sanity audit",
        "===================================",
        f"Project root: {project_root}",
        f"Run dir:      {run_dir}",
        "",
        "Source audit:",
    ]
    for k, v in payload["source_audit"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Upload ZIPs:")
    for row in payload["upload_zips"]:
        lines.append(f"  {row['size_mb']} MB  {row['path']}")
    if not payload["upload_zips"]:
        lines.append("  none found")
    lines.append("")
    rta = payload["report_text_audit"]
    for key in ["long_fx_text_hit_count", "malformed_top_structure_hit_count", "fx_loop_hit_count"]:
        if key in rta:
            lines.append(f"{key}: {rta[key]}")
    lines.append("")
    lines.append("Wrote JSON:")
    lines.append(str(json_path))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(txt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
