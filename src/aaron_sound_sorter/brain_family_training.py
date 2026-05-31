"""Build full/core/spread/outlier brain training trees.

This module does not classify from filenames. Folder path remains the trusted
label. It uses measured audio fingerprints only to choose which examples become
anchors in the small baby-brain training trees.
"""

from __future__ import annotations

import contextlib
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aaron_sound_sorter.core import AUDIO_EXTS
from aaron_sound_sorter.features import make_fingerprint_safe


@dataclass(frozen=True)
class AnchorSelectionConfig:
    core_anchors: int = 3
    spread_anchors: int = 3
    outlier_anchors: int = 3
    central_pool_fraction: float = 0.70
    spread_pool_fraction: float = 0.85
    max_files_per_label_to_scan: int = 0
    fingerprint_timeout_sec: float = 45.0


@dataclass(frozen=True)
class BrainFamilyTrees:
    full_root: Path
    core_root: Path
    spread_root: Path
    outlier_root: Path
    manifest_path: Path
    summary_path: Path


def is_audio_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.startswith("._") or path.name.startswith("."):
        return False
    if "__MACOSX" in path.parts:
        return False
    return path.suffix.lower() in AUDIO_EXTS


def direct_label_folders(training_root: Path) -> list[Path]:
    """Return folders that directly contain audio files."""
    folders: list[Path] = []
    for folder in sorted(p for p in training_root.rglob("*") if p.is_dir()):
        if any(is_audio_file(child) for child in folder.iterdir()):
            folders.append(folder)
    return folders


def label_audio_files(folder: Path, limit: int = 0) -> list[Path]:
    files = sorted([p for p in folder.iterdir() if is_audio_file(p)], key=lambda p: p.name.lower())
    if limit and limit > 0 and len(files) > limit:
        # Deterministic downsample by index so giant folders do not dominate runtime.
        idxs = np.linspace(0, len(files) - 1, num=limit, dtype=int)
        return [files[int(i)] for i in idxs]
    return files


def robust_scaled_vectors(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(vectors, dtype=np.float32)
    center = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - center), axis=0)
    scale = np.maximum(mad * 1.4826, 0.05)
    z = (arr - center) / scale
    return z, center, scale


def centrality_distances(vectors: np.ndarray) -> np.ndarray:
    z, _center, _scale = robust_scaled_vectors(vectors)
    return np.sqrt(np.mean(z * z, axis=1))


def select_core_indices(distances: np.ndarray, count: int) -> list[int]:
    if count <= 0 or len(distances) == 0:
        return []
    return [int(i) for i in np.argsort(distances)[: min(count, len(distances))]]


def select_spread_indices(
    vectors: np.ndarray, distances: np.ndarray, count: int, pool_fraction: float, exclude: set[int] | None = None
) -> list[int]:
    """Pick diverse clean anchors from the central-ish pool."""
    n = len(distances)
    if count <= 0 or n == 0:
        return []
    exclude = set(exclude or set())
    order = np.argsort(distances)
    pool_n = max(count, int(round(n * max(0.05, min(1.0, pool_fraction)))))
    pool = [int(i) for i in order[:pool_n] if int(i) not in exclude]
    if not pool:
        pool = [int(i) for i in order if int(i) not in exclude]
    if not pool:
        return []
    z, _center, _scale = robust_scaled_vectors(vectors)
    chosen: list[int] = []
    # Start with the most central remaining sample, then greedily maximize
    # distance from the selected set while staying in the clean pool.
    chosen.append(pool[0])
    while len(chosen) < min(count, len(pool)):
        best_i = None
        best_score = -1.0
        for idx in pool:
            if idx in chosen:
                continue
            d_to_chosen = [float(np.sqrt(np.mean((z[idx] - z[j]) ** 2))) for j in chosen]
            diversity = min(d_to_chosen) if d_to_chosen else 0.0
            # Mild centrality preference keeps spread anchors from becoming trash.
            score = diversity - 0.10 * float(distances[idx])
            if score > best_score:
                best_score = score
                best_i = idx
        if best_i is None:
            break
        chosen.append(int(best_i))
    return chosen


def select_outlier_indices(
    distances: np.ndarray, count: int, pool_fraction: float, exclude: set[int] | None = None
) -> list[int]:
    """Pick plausible edge anchors, not the most broken extremes when possible."""
    n = len(distances)
    if count <= 0 or n == 0:
        return []
    exclude = set(exclude or set())
    order = [int(i) for i in np.argsort(distances)]
    if n <= count:
        return [i for i in reversed(order) if i not in exclude][:count]
    # Avoid the most extreme tail; choose from upper middle/edge band.
    upper_start = int(round(n * max(0.0, min(0.95, pool_fraction))))
    candidate_band = order[upper_start:]
    if len(candidate_band) < count:
        candidate_band = order[max(0, n - max(count * 3, count)) :]
    candidate_band = [i for i in candidate_band if i not in exclude]
    candidate_band = sorted(candidate_band, key=lambda i: distances[i], reverse=True)
    return candidate_band[: min(count, len(candidate_band))]


def safe_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        destination.unlink()
    try:
        os.symlink(source.resolve(), destination)
    except Exception:
        shutil.copy2(source, destination)


def build_brain_family_training_trees(
    *,
    training_root: Path,
    output_root: Path,
    config: AnchorSelectionConfig,
) -> BrainFamilyTrees:
    training_root = Path(training_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not training_root.exists():
        raise FileNotFoundError(f"training root not found: {training_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    full_root = training_root
    core_root = output_root / "core_baby_training_tree"
    spread_root = output_root / "spread_baby_training_tree"
    outlier_root = output_root / "outlier_baby_training_tree"
    for root in [core_root, spread_root, outlier_root]:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    for folder in direct_label_folders(training_root):
        rel_folder = folder.relative_to(training_root)
        files = label_audio_files(folder, config.max_files_per_label_to_scan)
        fingerprints: list[np.ndarray] = []
        valid_files: list[Path] = []
        statuses: dict[str, str] = {}
        durations: dict[str, float] = {}
        for path in files:
            fp, duration, status = make_fingerprint_safe(path, timeout_sec=config.fingerprint_timeout_sec)
            statuses[str(path)] = status
            durations[str(path)] = float(duration)
            if status == "ok" and np.any(np.asarray(fp, dtype=np.float32)):
                fingerprints.append(np.asarray(fp, dtype=np.float32))
                valid_files.append(path)
            else:
                manifest_rows.append(
                    {
                        "label_folder": str(rel_folder),
                        "brain_lane": "skipped",
                        "selection_role": "unreadable_or_tiny",
                        "rank": "",
                        "source_path": str(path),
                        "file_name": path.name,
                        "duration_sec": f"{float(duration):.6f}",
                        "centrality_distance": "",
                        "read_status": status,
                        "folder_valid_count": "0",
                        "folder_total_count": str(len(files)),
                    }
                )
        if not valid_files:
            summary_rows.append(
                {
                    "label_folder": str(rel_folder),
                    "valid_count": "0",
                    "core": "0",
                    "spread": "0",
                    "outlier": "0",
                    "note": "no valid fingerprints",
                }
            )
            continue
        vectors = np.vstack(fingerprints).astype(np.float32)
        distances = centrality_distances(vectors)
        core = select_core_indices(distances, config.core_anchors)
        spread = select_spread_indices(
            vectors, distances, config.spread_anchors, config.spread_pool_fraction, exclude=set()
        )
        outlier = select_outlier_indices(
            distances, config.outlier_anchors, config.central_pool_fraction, exclude=set(core)
        )
        lane_defs = [
            ("core", core_root, core, "clean_center_precision"),
            ("spread", spread_root, spread, "diverse_clean_recall"),
            ("outlier", outlier_root, outlier, "plausible_edge_recall"),
        ]
        for lane, lane_root, indices, role in lane_defs:
            for rank, idx in enumerate(indices, start=1):
                src = valid_files[int(idx)]
                dst = lane_root / rel_folder / src.name
                safe_link(src, dst)
                manifest_rows.append(
                    {
                        "label_folder": str(rel_folder),
                        "brain_lane": lane,
                        "selection_role": role,
                        "rank": str(rank),
                        "source_path": str(src),
                        "file_name": src.name,
                        "duration_sec": f"{durations.get(str(src), 0.0):.6f}",
                        "centrality_distance": f"{float(distances[int(idx)]):.6f}",
                        "read_status": statuses.get(str(src), "ok"),
                        "folder_valid_count": str(len(valid_files)),
                        "folder_total_count": str(len(files)),
                    }
                )
        summary_rows.append(
            {
                "label_folder": str(rel_folder),
                "valid_count": str(len(valid_files)),
                "core": str(len(core)),
                "spread": str(len(spread)),
                "outlier": str(len(outlier)),
                "note": "ok",
            }
        )

    manifest_path = output_root / "brain_family_anchor_selection_manifest.csv"
    summary_path = output_root / "brain_family_anchor_selection_summary.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "label_folder",
            "brain_lane",
            "selection_role",
            "rank",
            "source_path",
            "file_name",
            "duration_sec",
            "centrality_distance",
            "read_status",
            "folder_valid_count",
            "folder_total_count",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        fields = ["label_folder", "valid_count", "core", "spread", "outlier", "note"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    return BrainFamilyTrees(
        full_root=full_root,
        core_root=core_root,
        spread_root=spread_root,
        outlier_root=outlier_root,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )
