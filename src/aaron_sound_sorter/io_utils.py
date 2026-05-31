# Auto-split from Aaron_Sound_Sorter.py.
# This is a component module, not a legacy wrapper.
from __future__ import annotations

from .core import *


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def find_latest_physics_report(root: Path) -> Path:
    manifests = root / "_MANIFESTS"
    if not manifests.exists():
        raise SystemExit(f"No _MANIFESTS folder found: {manifests}")
    patterns = [
        "physics_shape_outliers_*",
    ]
    for pat in patterns:
        dirs = sorted(
            p for p in manifests.glob(pat) if p.is_dir() and (p / "candidate_clean_training_set.csv").exists()
        )
        if dirs:
            return dirs[-1]
    raise SystemExit(f"No candidate_clean_training_set.csv found under {manifests}")


def parse_allowed(text: str) -> set[str]:
    items = {x.strip() for x in re.split(r"[, ]+", str(text or "")) if x.strip()}
    return items or set(DEFAULT_ALLOWED_TOP)


def row_float(row: Dict[str, str], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except Exception:
            pass
    return float(default)


def row_int(row: Dict[str, str], *names: str, default: int = 0) -> int:
    for name in names:
        value = row.get(name)
        if value is None or value == "":
            continue
        try:
            return int(float(value))
        except Exception:
            pass
    return int(default)


def stable_random_key(*parts: str, seed: int = 20260503) -> str:
    payload = "||".join([str(seed)] + [str(p) for p in parts])
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def source_pack_key_for_path(path_text: str) -> str:
    """Return a stable source-pack key for balanced random sampling.

    The sorted training tree is mostly symlinks back into Aaron's sample library.
    Use the resolved target path when possible so one commercial pack cannot
    dominate a category just because many symlinks landed in one folder.
    This is curation bookkeeping only. It is not used to classify mystery audio.
    """
    try:
        p = Path(str(path_text)).expanduser()
        if p.exists() or p.is_symlink():
            p = p.resolve()
        parts = [part for part in p.parts if part not in {"", "/"}]
        lowered = [part.lower() for part in parts]
        if "samples" in lowered:
            idx = lowered.index("samples")
            if idx + 1 < len(parts):
                return clean_hierarchy_part(parts[idx + 1])
        if len(parts) >= 2:
            return clean_hierarchy_part(parts[-2])
        return clean_hierarchy_part(p.stem or "unknown_source")
    except Exception:
        return "unknown_source"


def source_balance_key_for_path(path_text: str) -> str:
    """Granular curation bucket for per-folder random training balance.

    This is not classifier evidence. It only prevents one big pack or one
    overstuffed folder from filling all 100 teacher slots for a label. When
    files are symlinks, resolve them so balance follows the real source folder.
    """
    try:
        raw = Path(str(path_text)).expanduser()
        p = raw.resolve() if (raw.exists() or raw.is_symlink()) else raw
        parts = [part for part in p.parts if part not in {"", "/"}]
        lowered = [part.lower() for part in parts]
        if "samples" in lowered:
            idx = lowered.index("samples")
            rel = parts[idx + 1 : -1]
            low_rel = [part.lower() for part in rel]
            if rel and low_rel[0] == "sorted samples":
                # If this is a real copied file inside the trusted tree, the
                # original source folder is gone. Balance within the trusted
                # category using a stable filename bucket so the first files in
                # one folder cannot dominate the teacher set.
                label_context = "/".join(clean_hierarchy_part(part) for part in rel[1:4]) or "Sorted samples"
                bucket = stable_random_key(str(raw), "balance_bucket")[:2]
                return f"{label_context}/bucket_{bucket}"
            if rel:
                # Pack + one or two child folders gives better variety than the
                # old pack-only key while still keeping reports readable.
                return "/".join(clean_hierarchy_part(part) for part in rel[:3])
        parent = clean_hierarchy_part(p.parent.name if p.parent else "unknown_source")
        bucket = stable_random_key(str(p), "balance_bucket")[:2]
        return f"{parent}/bucket_{bucket}"
    except Exception:
        return "unknown_source/bucket_unknown"


def is_near_silent_or_too_tiny_training_row(row: Dict[str, str]) -> bool:
    duration = row_float(row, "duration_sec", "duration", default=0.0)
    rms = row_float(row, "rms", "rms_mean", "loudness_rms", default=999.0)
    peak = row_float(row, "peak", "peak_abs", "peak_amplitude", default=999.0)
    active_ratio = row_float(row, "active_ratio", "non_silent_ratio", default=1.0)
    if duration > 0 and duration <= 0.05:
        return True
    if rms != 999.0 and rms < 0.0005:
        return True
    if peak != 999.0 and peak < 0.003:
        return True
    return active_ratio < 0.05
