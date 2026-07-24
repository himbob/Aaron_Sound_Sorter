# SOURCE-NAME BLINDNESS INVARIANT:
# This module caches internal brain arrays and label metadata only. It must
# never parse producer filenames, source paths, ZIP member names, or folders as
# classification evidence.
"""In-memory runtime cache for loaded brain JSON data.

The trained brains are stored as JSON so users can inspect, back up, and move
them easily. JSON is slow for hot scoring loops, though: every file used to
rebuild numpy arrays from the same nested lists for every brain lane and label.
This module converts source-blind brain data into numpy arrays once per loaded
brain object and reuses those arrays during the current process.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np

from aaron_sound_sorter.core import FEATURE_WEIGHTS, FP_SIZE
from aaron_sound_sorter.training_labels import label_default_structure, public_label, top_for_public_label


@dataclass(frozen=True)
class RuntimeLabelModel:
    """Precomputed arrays and adaptive-model metadata for one brain label."""

    label: str
    model_mode: str
    spread_mean: float
    training_count: int
    centroids_weighted: np.ndarray
    exemplars_weighted: np.ndarray
    anchors_weighted: np.ndarray


@dataclass(frozen=True)
class BrainRuntimeCache:
    """Precomputed source-blind runtime data for one loaded brain object."""

    signature: tuple[object, ...]
    labels: tuple[str, ...]
    feature_weights: np.ndarray
    folder_map: dict[str, str]
    top_map: dict[str, str]
    structure_by_label: dict[str, str]
    global_scaler: tuple[np.ndarray, np.ndarray]
    scaler_by_structure: dict[str, tuple[np.ndarray, np.ndarray]]
    model_by_label: dict[str, RuntimeLabelModel]

    def scaler_for_label(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the already-normalized scaler vectors for one label."""
        structure = self.structure_by_label.get(label, label_default_structure(label))
        return self.scaler_by_structure.get(structure, self.global_scaler)


_CACHE_LOCK = Lock()
_CACHE_BY_BRAIN_ID: dict[int, BrainRuntimeCache] = {}


def get_brain_runtime_cache(brain: dict[str, Any]) -> BrainRuntimeCache:
    """Return the runtime cache for one loaded brain dictionary.

    Args:
        brain: Loaded brain JSON dictionary. The dictionary is not modified.

    Returns:
        A process-local cache containing numpy arrays and label metadata.

    Side Effects:
        Creates or replaces a module-level cache entry keyed by ``id(brain)``.

    Raises:
        No intentional exceptions. Malformed brain fields are treated like empty
        or default fields, matching the older scorer behavior.

    Important Constraints:
        The cache stores no final decisions, no source paths, and no training
        labels beyond the internal category labels already present in the brain.
    """
    signature = _brain_signature(brain)
    brain_id = id(brain)
    with _CACHE_LOCK:
        cached = _CACHE_BY_BRAIN_ID.get(brain_id)
        if cached is not None and cached.signature == signature:
            return cached
        runtime = _build_brain_runtime_cache(brain, signature)
        _CACHE_BY_BRAIN_ID[brain_id] = runtime
        return runtime


def clear_brain_runtime_cache() -> None:
    """Clear all runtime cache entries.

    This is mainly for tests and long-lived tooling that intentionally mutates a
    loaded brain object between phases.
    """
    with _CACHE_LOCK:
        _CACHE_BY_BRAIN_ID.clear()


def _brain_signature(brain: dict[str, Any]) -> tuple[object, ...]:
    """Return a cheap mutation signature for one loaded brain object."""
    labels = brain.get("labels", []) if isinstance(brain, dict) else []
    centroids = brain.get("centroids", {}) if isinstance(brain, dict) else {}
    exemplars = brain.get("exemplars_by_label", {}) if isinstance(brain, dict) else {}
    anchors = brain.get("anchors_by_label", {}) if isinstance(brain, dict) else {}
    models = brain.get("label_models_by_label", {}) if isinstance(brain, dict) else {}
    scalers = brain.get("scalers_by_structure", {}) if isinstance(brain, dict) else {}
    structures = brain.get("structure_by_label", {}) if isinstance(brain, dict) else {}
    return (
        len(labels) if isinstance(labels, list) else 0,
        len(centroids) if isinstance(centroids, dict) else 0,
        len(exemplars) if isinstance(exemplars, dict) else 0,
        len(anchors) if isinstance(anchors, dict) else 0,
        len(models) if isinstance(models, dict) else 0,
        len(scalers) if isinstance(scalers, dict) else 0,
        len(structures) if isinstance(structures, dict) else 0,
        id(brain.get("labels")) if isinstance(brain, dict) else 0,
        id(brain.get("feature_weights")) if isinstance(brain, dict) else 0,
        id(brain.get("centroids")) if isinstance(brain, dict) else 0,
        id(brain.get("exemplars_by_label")) if isinstance(brain, dict) else 0,
        id(brain.get("anchors_by_label")) if isinstance(brain, dict) else 0,
        id(brain.get("label_models_by_label")) if isinstance(brain, dict) else 0,
    )


def _build_brain_runtime_cache(brain: dict[str, Any], signature: tuple[object, ...]) -> BrainRuntimeCache:
    """Build all reusable arrays and metadata for one brain."""
    labels = _brain_labels(brain)
    weights = _finite_vector(brain.get("feature_weights", FEATURE_WEIGHTS), fill=1.0)
    global_scaler = _global_scaler(brain)
    scaler_by_structure = _scaler_by_structure(brain, global_scaler)
    structure_by_label = _structure_by_label(brain, labels)
    folder_map = {label: public_label(label) for label in labels}
    top_map = _top_map(brain, labels)
    model_by_label = {
        label: _runtime_label_model(label=label, brain=brain, feature_weights=weights) for label in labels
    }
    return BrainRuntimeCache(
        signature=signature,
        labels=labels,
        feature_weights=weights,
        folder_map=folder_map,
        top_map=top_map,
        structure_by_label=structure_by_label,
        global_scaler=global_scaler,
        scaler_by_structure=scaler_by_structure,
        model_by_label=model_by_label,
    )


def _brain_labels(brain: dict[str, Any]) -> tuple[str, ...]:
    """Return all labels declared or implied by one brain dictionary."""
    labels: list[str] = []
    seen: set[str] = set()
    sources = (
        brain.get("labels", []),
        (brain.get("centroids", {}) or {}).keys() if isinstance(brain.get("centroids", {}), dict) else (),
        (brain.get("exemplars_by_label", {}) or {}).keys()
        if isinstance(brain.get("exemplars_by_label", {}), dict)
        else (),
        (brain.get("anchors_by_label", {}) or {}).keys() if isinstance(brain.get("anchors_by_label", {}), dict) else (),
        (brain.get("label_models_by_label", {}) or {}).keys()
        if isinstance(brain.get("label_models_by_label", {}), dict)
        else (),
    )
    for source in sources:
        for raw_label in source or ():
            label = str(raw_label)
            if label and label not in seen:
                seen.add(label)
                labels.append(label)
    return tuple(labels)


def _global_scaler(brain: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized global mean/std vectors."""
    mean = _finite_vector(brain.get("scaler_mean", np.zeros((FP_SIZE,), dtype=np.float32)), fill=0.0)
    std = _positive_std_vector(brain.get("scaler_std", np.ones((FP_SIZE,), dtype=np.float32)))
    return mean, std


def _scaler_by_structure(
    brain: dict[str, Any],
    global_scaler: tuple[np.ndarray, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return normalized structure-specific scaler vectors."""
    scalers = brain.get("scalers_by_structure", {}) if isinstance(brain.get("scalers_by_structure", {}), dict) else {}
    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if not isinstance(scalers, dict):
        return normalized
    for structure, scaler in scalers.items():
        if not isinstance(scaler, dict) or "mean" not in scaler or "std" not in scaler:
            normalized[str(structure)] = global_scaler
            continue
        normalized[str(structure)] = (
            _finite_vector(scaler.get("mean", []), fill=0.0),
            _positive_std_vector(scaler.get("std", [])),
        )
    return normalized


def _structure_by_label(brain: dict[str, Any], labels: tuple[str, ...]) -> dict[str, str]:
    """Return internal label to structure map with defaults filled in."""
    raw = brain.get("structure_by_label", {}) if isinstance(brain.get("structure_by_label", {}), dict) else {}
    return {label: str(raw.get(label, "") or label_default_structure(label) or "one_shot") for label in labels}


def _top_map(brain: dict[str, Any], labels: tuple[str, ...]) -> dict[str, str]:
    """Return internal label to top-family map with defaults filled in."""
    top_by_label = brain.get("top_by_label", {}) if isinstance(brain.get("top_by_label", {}), dict) else {}
    return {label: str(top_by_label.get(label, "") or top_for_public_label(label) or "_TO_REVIEW") for label in labels}


def _runtime_label_model(label: str, brain: dict[str, Any], feature_weights: np.ndarray) -> RuntimeLabelModel:
    """Return preweighted model arrays for one label."""
    label_models = brain.get("label_models_by_label", {})
    model_info = label_models.get(label, {}) if isinstance(label_models, dict) else {}
    model = model_info if isinstance(model_info, dict) else {}
    return RuntimeLabelModel(
        label=label,
        model_mode=str(model.get("model_mode", "centroid")),
        spread_mean=_safe_float(model.get("spread_mean", 1.0), default=1.0),
        training_count=int(_safe_float(model.get("training_count", 0), default=0.0)),
        centroids_weighted=_weighted_matrix(brain.get("centroids", {}), label, feature_weights),
        exemplars_weighted=_weighted_matrix(brain.get("exemplars_by_label", {}), label, feature_weights),
        anchors_weighted=_weighted_matrix(brain.get("anchors_by_label", {}), label, feature_weights),
    )


def _weighted_matrix(table: object, label: str, feature_weights: np.ndarray) -> np.ndarray:
    """Return one label's model rows already multiplied by feature weights."""
    rows = table.get(label, []) if isinstance(table, dict) else []
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim == 1 and matrix.size:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or not matrix.size:
        return np.zeros((0, FP_SIZE), dtype=np.float32)
    usable = min(matrix.shape[1], feature_weights.size, FP_SIZE)
    weighted = np.zeros((matrix.shape[0], FP_SIZE), dtype=np.float32)
    weighted[:, :usable] = matrix[:, :usable] * feature_weights[:usable][None, :]
    return np.nan_to_num(weighted, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _finite_vector(values: object, *, fill: float) -> np.ndarray:
    """Return a finite FP_SIZE vector."""
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size < FP_SIZE:
        vector = np.pad(vector, (0, FP_SIZE - vector.size), mode="constant", constant_values=fill)
    return np.nan_to_num(vector[:FP_SIZE], nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)


def _positive_std_vector(values: object) -> np.ndarray:
    """Return a finite, nonzero standard-deviation vector."""
    vector = _finite_vector(values, fill=1.0)
    return np.where(np.abs(vector) < 1e-6, 1.0, vector).astype(np.float32, copy=False)


def _safe_float(value: object, *, default: float) -> float:
    """Return ``value`` as a float, or ``default`` for malformed input."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
