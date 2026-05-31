# Auto-split from Aaron_Sound_Sorter.py.
# This is a component module, not a legacy wrapper.
from __future__ import annotations

from .core import *


def label_reliability_requirements(
    count: int, active_status: str = "", source_group_count: int = 0
) -> Tuple[str, float, float]:
    """Return learned-support tier and extra auto-place requirements.

    This is not semantic routing.  It only says labels trained from fewer
    examples are less reliable and need stronger brain evidence before placing.
    """
    n = int(count or 0)
    status = str(active_status or "")
    if status == "ACTIVE_PROVISIONAL_LOW_COUNT":
        return (
            "provisional_tiny",
            max(TINY_LABEL_MIN_SIMILARITY, PROVISIONAL_TINY_MIN_SIMILARITY),
            max(TINY_LABEL_MIN_MARGIN, PROVISIONAL_TINY_MIN_MARGIN),
        )
    if status == "ACTIVE_PROVISIONAL_SINGLE_SOURCE" or int(source_group_count or 0) == 1:
        base_tier = "provisional_single_source"
        if n <= TINY_LABEL_COUNT:
            return (
                base_tier,
                max(TINY_LABEL_MIN_SIMILARITY, PROVISIONAL_TINY_MIN_SIMILARITY),
                max(TINY_LABEL_MIN_MARGIN, PROVISIONAL_TINY_MIN_MARGIN),
            )
        return (
            base_tier,
            max(SMALL_LABEL_MIN_SIMILARITY, PROVISIONAL_SINGLE_SOURCE_MIN_SIMILARITY),
            max(SMALL_LABEL_MIN_MARGIN, PROVISIONAL_SINGLE_SOURCE_MIN_MARGIN),
        )
    if n <= TINY_LABEL_COUNT:
        return "tiny", TINY_LABEL_MIN_SIMILARITY, TINY_LABEL_MIN_MARGIN
    if n <= SMALL_LABEL_COUNT:
        return "small", SMALL_LABEL_MIN_SIMILARITY, SMALL_LABEL_MIN_MARGIN
    if n <= MEDIUM_LABEL_COUNT:
        return "medium", MEDIUM_LABEL_MIN_SIMILARITY, MEDIUM_LABEL_MIN_MARGIN
    return "strong", 0.0, 0.0


def _effective_training_cap_for_label(raw_count: int) -> int:
    """Return max competitive rows for one learned folder.

    This does not delete training data. It only caps the effective rows used to
    fit scalers, centroids, anchors, and broad top-family geometry.
    """
    n = int(raw_count or 0)
    if n <= EFFECTIVE_TRAIN_TARGET_PER_LABEL:
        return n
    if n >= EFFECTIVE_TRAIN_HUGE_LABEL_THRESHOLD:
        return min(n, EFFECTIVE_TRAIN_HUGE_MAX_PER_LABEL)
    return min(n, EFFECTIVE_TRAIN_MAX_PER_LABEL)


def _weighted_rows_for_selection(rows: List[FeatureRow]) -> np.ndarray:
    arr = np.asarray([r.fingerprint for r in rows], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    mean = np.mean(arr, axis=0).astype(np.float32)
    std = np.std(arr, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((arr - mean) / std) * FEATURE_WEIGHTS.astype(np.float32)[None, :]


def _select_effective_label_rows(rows: List[FeatureRow], cap: int) -> Tuple[List[FeatureRow], Dict[str, object]]:
    """Deterministically select a central + diverse subset.

    No random sampling and no category-specific logic.  The goal is to represent
    a giant folder without letting it dominate the whole brain by raw count.
    v0.4.68 deliberately removes the old edge/outlier slice because the logs
    proved edge rows amplify structure contradictions in mixed or weak labels.
    """
    n = len(rows)
    cap = max(0, min(int(cap), n))
    if cap <= 0:
        return [], {"selection_method": "none", "cap": 0}
    if n <= cap:
        return list(rows), {
            "selection_method": "all_rows_under_cap",
            "cap": int(cap),
            "central_selected": n,
            "diverse_selected": 0,
            "edge_selected": 0,
        }

    ordered = sorted(rows, key=lambda r: (str(r.source_pack), str(r.group_key), str(r.path)))
    W = _weighted_rows_for_selection(ordered)
    center = np.mean(W, axis=0, keepdims=True)
    dist_to_center = np.linalg.norm(W - center, axis=1)

    central_target = max(1, int(round(cap * EFFECTIVE_TRAIN_CENTRAL_FRACTION)))
    diverse_target = max(0, cap - central_target)

    chosen: List[int] = []
    chosen_set = set()

    # Central examples: the safest representative core of the folder.
    for idx in np.argsort(dist_to_center, kind="mergesort"):
        if len(chosen) >= central_target:
            break
        i = int(idx)
        chosen.append(i)
        chosen_set.add(i)

    # Diverse examples: farthest-point traversal from the central core.
    while len(chosen) < central_target + diverse_target and len(chosen) < cap:
        chosen_vecs = W[chosen] if chosen else center
        d = np.linalg.norm(W[:, None, :] - chosen_vecs[None, :, :], axis=2)
        min_d = np.min(d, axis=1)
        for i in chosen_set:
            min_d[i] = -1.0
        nxt = int(np.argmax(min_d))
        if nxt in chosen_set or min_d[nxt] < 0:
            break
        chosen.append(nxt)
        chosen_set.add(nxt)

    # No edge/outlier slice in v0.4.68. If anything was still short due to
    # duplicates or degenerate vectors, fill deterministically from the central
    # ordered pool rather than deliberately selecting the farthest outliers.

    # If anything was still short due to duplicates, fill deterministically.
    for i in range(n):
        if len(chosen) >= cap:
            break
        if i not in chosen_set:
            chosen.append(i)
            chosen_set.add(i)

    selected = [ordered[i] for i in chosen[:cap]]
    return selected, {
        "selection_method": "central_diverse_no_edge_cap",
        "cap": int(cap),
        "central_selected": int(min(central_target, cap)),
        "diverse_selected": int(max(0, min(diverse_target, cap - min(central_target, cap)))),
        "edge_selected": int(
            max(0, cap - min(central_target, cap) - min(diverse_target, max(0, cap - min(central_target, cap))))
        ),
        "mean_distance_to_center_raw_pool": float(np.mean(dist_to_center)) if dist_to_center.size else 0.0,
        "p90_distance_to_center_raw_pool": float(np.percentile(dist_to_center, 90)) if dist_to_center.size else 0.0,
    }


def training_row_math_exclusion_reason(row: FeatureRow) -> str:
    """Return a dynamic reason to exclude a row from prototype math, or blank.

    This does not delete, move, rename, or relabel anything in the trusted tree.
    It only prevents high-confidence folder/audio structure contradictions from
    shaping scalers, centroids, anchors, and broad family prototypes.
    """
    if not CONTRADICTION_FILTER_EFFECTIVE_TRAINING_ENABLED_DEFAULT:
        return ""
    if row.read_status != "ok":
        return ""
    folder_structure = row.structure or label_default_structure(row.label)
    has_loop_audio, loop_reason = fingerprint_loop_evidence_detail(row.fingerprint, row.duration_sec)
    if folder_structure == "loop" and not has_loop_audio:
        return f"exclude_from_math:FOLDER_LOOP_AUDIO_ONE_SHOT:{loop_reason}"
    if folder_structure == "one_shot" and has_loop_audio:
        return f"exclude_from_math:FOLDER_ONE_SHOT_AUDIO_LOOP:{loop_reason}"
    return ""


def select_balanced_effective_training_rows(
    good_rows: List[FeatureRow],
) -> Tuple[List[FeatureRow], Dict[str, Dict[str, object]]]:
    """Build the competitive brain pool from balanced effective label support.

    The full trusted tree remains the source of truth. This only changes the
    math pool used for scaler/centroid/anchor fitting so huge folders cannot
    swamp small correct folders and so high-confidence structure contradictions
    do not poison label geometry.
    """
    by_label_rows: Dict[str, List[FeatureRow]] = defaultdict(list)
    for row in good_rows:
        by_label_rows[row.label].append(row)

    effective: List[FeatureRow] = []
    report: Dict[str, Dict[str, object]] = {}
    for label, rows in sorted(by_label_rows.items()):
        raw_count = len(rows)
        excluded_rows: List[Tuple[FeatureRow, str]] = []
        eligible_rows: List[FeatureRow] = []
        for row in rows:
            reason = training_row_math_exclusion_reason(row)
            if reason:
                excluded_rows.append((row, reason))
            else:
                eligible_rows.append(row)

        # Preserve label existence. If every row is contradictory, keep a small
        # central-only fallback and mark it loudly instead of deleting the label.
        used_fallback = False
        pool_rows = eligible_rows
        if not pool_rows:
            used_fallback = True
            pool_rows = rows

        raw_cap = _effective_training_cap_for_label(raw_count)
        cap = min(raw_cap, len(pool_rows))
        selected, meta = _select_effective_label_rows(pool_rows, cap)
        effective.extend(selected)
        top = selected[0].top if selected else (rows[0].top if rows else "")
        structure_counts = Counter((r.structure or label_default_structure(r.label)) for r in selected)
        raw_structure_counts = Counter((r.structure or label_default_structure(r.label)) for r in rows)
        eligible_structure_counts = Counter((r.structure or label_default_structure(r.label)) for r in eligible_rows)
        source_pack_counts = Counter(source_pack_key_for_path(r.path) for r in rows)
        exclusion_reason_counts = Counter(
            reason.split(":", 2)[1] if ":" in reason else reason for _row, reason in excluded_rows
        )
        report[label] = {
            "label": label,
            "top": top,
            "raw_count": int(raw_count),
            "eligible_count_after_contradiction_filter": int(len(eligible_rows)),
            "excluded_from_math_count": int(len(excluded_rows)),
            "used_fallback_when_all_rows_excluded": "yes" if used_fallback else "no",
            "effective_count": int(len(selected)),
            "cap_applied": "yes" if len(selected) < raw_count else "no",
            "effective_cap": int(cap),
            "selection_method": str(meta.get("selection_method", "")),
            "central_selected": int(meta.get("central_selected", 0) or 0),
            "diverse_selected": int(meta.get("diverse_selected", 0) or 0),
            "edge_selected": int(meta.get("edge_selected", 0) or 0),
            "raw_structure_counts": dict(sorted(raw_structure_counts.items())),
            "eligible_structure_counts": dict(sorted(eligible_structure_counts.items())),
            "effective_structure_counts": dict(sorted(structure_counts.items())),
            "source_pack_count": int(len(source_pack_counts)),
            "source_pack_counts_top10": dict(source_pack_counts.most_common(10)),
            "exclusion_reason_counts": dict(exclusion_reason_counts.most_common()),
            "excluded_examples_top10": [
                {"source_path": erow.path, "reason": reason[:240]} for erow, reason in excluded_rows[:10]
            ],
            "mean_distance_to_center_raw_pool": float(meta.get("mean_distance_to_center_raw_pool", 0.0) or 0.0),
            "p90_distance_to_center_raw_pool": float(meta.get("p90_distance_to_center_raw_pool", 0.0) or 0.0),
        }
    return effective, report


def write_effective_training_balance_report(brain: dict, reports_dir: Path) -> Path:
    rows = []
    info = brain.get("effective_training_balance_by_label", {})
    if isinstance(info, dict):
        for label, meta in sorted(info.items()):
            if not isinstance(meta, dict):
                continue
            rows.append(
                {
                    "label": label,
                    "top": meta.get("top", ""),
                    "raw_count": meta.get("raw_count", ""),
                    "effective_count": meta.get("effective_count", ""),
                    "eligible_count_after_contradiction_filter": meta.get(
                        "eligible_count_after_contradiction_filter", ""
                    ),
                    "excluded_from_math_count": meta.get("excluded_from_math_count", ""),
                    "used_fallback_when_all_rows_excluded": meta.get("used_fallback_when_all_rows_excluded", ""),
                    "cap_applied": meta.get("cap_applied", ""),
                    "effective_cap": meta.get("effective_cap", ""),
                    "selection_method": meta.get("selection_method", ""),
                    "central_selected": meta.get("central_selected", ""),
                    "diverse_selected": meta.get("diverse_selected", ""),
                    "edge_selected": meta.get("edge_selected", ""),
                    "source_pack_count": meta.get("source_pack_count", ""),
                    "raw_structure_counts": json.dumps(meta.get("raw_structure_counts", {}), sort_keys=True),
                    "eligible_structure_counts": json.dumps(meta.get("eligible_structure_counts", {}), sort_keys=True),
                    "effective_structure_counts": json.dumps(
                        meta.get("effective_structure_counts", {}), sort_keys=True
                    ),
                    "exclusion_reason_counts": json.dumps(meta.get("exclusion_reason_counts", {}), sort_keys=True),
                    "excluded_examples_top10": json.dumps(meta.get("excluded_examples_top10", []), sort_keys=True),
                    "source_pack_counts_top10": json.dumps(meta.get("source_pack_counts_top10", {}), sort_keys=True),
                    "mean_distance_to_center_raw_pool": f"{float(meta.get('mean_distance_to_center_raw_pool', 0.0) or 0.0):.6f}",
                    "p90_distance_to_center_raw_pool": f"{float(meta.get('p90_distance_to_center_raw_pool', 0.0) or 0.0):.6f}",
                }
            )
    path = reports_dir / "effective_training_balance_report.csv"
    write_csv(
        path,
        rows,
        [
            "label",
            "top",
            "raw_count",
            "eligible_count_after_contradiction_filter",
            "excluded_from_math_count",
            "used_fallback_when_all_rows_excluded",
            "effective_count",
            "cap_applied",
            "effective_cap",
            "selection_method",
            "central_selected",
            "diverse_selected",
            "edge_selected",
            "source_pack_count",
            "raw_structure_counts",
            "eligible_structure_counts",
            "effective_structure_counts",
            "exclusion_reason_counts",
            "excluded_examples_top10",
            "source_pack_counts_top10",
            "mean_distance_to_center_raw_pool",
            "p90_distance_to_center_raw_pool",
        ],
    )
    return path


def _finite_global_stats(mean: np.ndarray, std: np.ndarray, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite scaler arrays for linear heads.

    The prototype scorer can tolerate some degenerate dimensions, but the ridge
    heads build X.T @ X.  One zero/NaN/Inf standard deviation or one bad feature
    row can create Inf/NaN in the entire matrix and poison the router brain.
    """
    m = np.asarray(mean, dtype=np.float64).reshape(-1)
    s = np.asarray(std, dtype=np.float64).reshape(-1)
    if m.shape[0] != n_features:
        m = np.zeros((n_features,), dtype=np.float64)
    if s.shape[0] != n_features:
        s = np.ones((n_features,), dtype=np.float64)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    s = np.nan_to_num(s, nan=1.0, posinf=1.0, neginf=1.0)
    s = np.where(np.isfinite(s) & (np.abs(s) >= 1e-6), s, 1.0)
    return m, s


def _safe_linear_design_matrix(
    fingerprints: Sequence[Sequence[float]],
    global_mean: np.ndarray,
    global_std: np.ndarray,
    *,
    feature_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Build a finite clipped design matrix for the linear/router heads."""
    X = np.asarray(list(fingerprints), dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n_rows = int(X.shape[0]) if X.ndim == 2 else 0
    n_features = int(X.shape[1]) if X.ndim == 2 and X.shape[1] > 0 else FP_SIZE
    if n_rows <= 0:
        return np.zeros((0, FP_SIZE), dtype=np.float64), {"ok": False, "reason": "empty_matrix"}
    if n_features != FP_SIZE:
        return np.zeros((0, FP_SIZE), dtype=np.float64), {"ok": False, "reason": f"bad_feature_width_{n_features}"}
    raw_nonfinite = int(np.size(X) - np.isfinite(X).sum())
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mean, std = _finite_global_stats(global_mean, global_std, n_features)
    weights = np.asarray(feature_weights if feature_weights is not None else FEATURE_WEIGHTS, dtype=np.float64).reshape(
        -1
    )
    if weights.shape[0] != n_features:
        weights = np.ones((n_features,), dtype=np.float64)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    Z = ((X - mean[None, :]) / std[None, :]) * weights[None, :]
    pre_clip_nonfinite = int(np.size(Z) - np.isfinite(Z).sum())
    Z = np.nan_to_num(Z, nan=0.0, posinf=LINEAR_RIDGE_FEATURE_CLIP, neginf=-LINEAR_RIDGE_FEATURE_CLIP)
    Z = np.clip(Z, -float(LINEAR_RIDGE_FEATURE_CLIP), float(LINEAR_RIDGE_FEATURE_CLIP))
    zero_var_dims = (
        int(np.sum(np.asarray(global_std, dtype=np.float64).reshape(-1)[:n_features] < 1e-6))
        if np.asarray(global_std).size >= n_features
        else -1
    )
    return Z.astype(np.float64, copy=False), {
        "ok": True,
        "raw_nonfinite_values": raw_nonfinite,
        "standardized_nonfinite_values": pre_clip_nonfinite,
        "feature_clip": float(LINEAR_RIDGE_FEATURE_CLIP),
        "zero_or_tiny_std_dims": zero_var_dims,
    }


def _safe_balanced_sample_weights(targets: Sequence[str]) -> Tuple[np.ndarray, Dict[str, object]]:
    counts = Counter(str(t) for t in targets)
    sw = np.asarray([1.0 / math.sqrt(max(1.0, float(counts.get(str(t), 1)))) for t in targets], dtype=np.float64)
    if sw.size == 0 or not np.any(sw > 0):
        return np.zeros((0,), dtype=np.float64), {"ok": False, "reason": "no_sample_weights"}
    sw = np.nan_to_num(sw, nan=1.0, posinf=1.0, neginf=1.0)
    sw = sw / max(1e-9, float(np.mean(sw[sw > 0])))
    sw = np.clip(sw, 1.0 / float(LINEAR_RIDGE_MAX_SAMPLE_WEIGHT), float(LINEAR_RIDGE_MAX_SAMPLE_WEIGHT))
    return sw, {"ok": True, "max_sample_weight": float(np.max(sw)), "min_sample_weight": float(np.min(sw))}


def _safe_ridge_weights(
    X_aug: np.ndarray, Y: np.ndarray, sw: np.ndarray, lam: float
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    """Solve a ridge system without letting Inf/NaN contaminate the brain JSON."""
    if X_aug.size == 0 or Y.size == 0 or sw.size == 0:
        return None, {"ok": False, "reason": "empty_ridge_input"}
    X_aug = np.nan_to_num(
        np.asarray(X_aug, dtype=np.float64),
        nan=0.0,
        posinf=LINEAR_RIDGE_FEATURE_CLIP,
        neginf=-LINEAR_RIDGE_FEATURE_CLIP,
    )
    X_aug = np.clip(X_aug, -float(LINEAR_RIDGE_FEATURE_CLIP), float(LINEAR_RIDGE_FEATURE_CLIP))
    Y = np.nan_to_num(np.asarray(Y, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    sw = np.nan_to_num(np.asarray(sw, dtype=np.float64), nan=1.0, posinf=1.0, neginf=1.0)
    sw = np.clip(sw, 1.0 / float(LINEAR_RIDGE_MAX_SAMPLE_WEIGHT), float(LINEAR_RIDGE_MAX_SAMPLE_WEIGHT))
    root_w = np.sqrt(sw).reshape(-1, 1)
    Xw = X_aug * root_w
    Yw = Y * root_w
    with np.errstate(all="ignore"):
        A = Xw.T @ Xw
        B = Xw.T @ Yw
    nonfinite_a = int(np.size(A) - np.isfinite(A).sum())
    nonfinite_b = int(np.size(B) - np.isfinite(B).sum())
    if nonfinite_a or nonfinite_b:
        A = np.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)
        B = np.nan_to_num(B, nan=0.0, posinf=1e6, neginf=-1e6)
        A = np.clip(A, -1e6, 1e6)
        B = np.clip(B, -1e6, 1e6)
    reg = np.eye(A.shape[0], dtype=np.float64) * max(1e-6, float(lam))
    reg[-1, -1] = max(1e-6, float(lam)) * 0.05
    try:
        W = np.linalg.solve(A + reg, B)
        solver = "solve"
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(A + reg) @ B
        solver = "pinv"
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(W)):
        return None, {"ok": False, "reason": "nonfinite_weights_after_solve"}
    weight_abs_max = float(np.max(np.abs(W))) if W.size else 0.0
    if weight_abs_max > float(LINEAR_RIDGE_MAX_ABS_WEIGHT):
        return None, {
            "ok": False,
            "reason": "linear_weight_abs_max_too_large",
            "weight_abs_max": weight_abs_max,
            "max_allowed_abs_weight": float(LINEAR_RIDGE_MAX_ABS_WEIGHT),
            "solver": solver,
            "nonfinite_A_values_repaired": nonfinite_a,
            "nonfinite_B_values_repaired": nonfinite_b,
        }
    return W, {
        "ok": True,
        "solver": solver,
        "nonfinite_A_values_repaired": nonfinite_a,
        "nonfinite_B_values_repaired": nonfinite_b,
        "weight_abs_max": weight_abs_max,
        "max_allowed_abs_weight": float(LINEAR_RIDGE_MAX_ABS_WEIGHT),
    }


def build_linear_ridge_head(
    rows: List[FeatureRow], labels: List[str], global_mean: np.ndarray, global_std: np.ndarray
) -> Dict[str, object]:
    """Build a small balanced linear classifier head with only NumPy.

    This is the non-prototype model in the tournament. It is a regularized
    one-vs-rest ridge classifier over the same 50-feature fingerprint. Class
    influence is balanced by giving each learned label roughly equal total
    weight regardless of raw folder count.  No filenames and no semantic label
    rules are used.
    """
    if not LINEAR_RIDGE_HEAD_ENABLED_DEFAULT or not rows or not labels:
        return {"enabled": False, "reason": "no_rows_or_disabled"}
    label_to_idx = {label: i for i, label in enumerate(labels)}
    X, matrix_diag = _safe_linear_design_matrix([r.fingerprint for r in rows], global_mean, global_std)
    if not matrix_diag.get("ok", False):
        return {
            "enabled": False,
            "reason": str(matrix_diag.get("reason", "bad_design_matrix")),
            "diagnostics": matrix_diag,
        }
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    Counter(r.label for r in rows)
    C = len(labels)
    Y = np.zeros((X_aug.shape[0], C), dtype=np.float64)
    targets_for_weights: List[str] = []
    for i, r in enumerate(rows):
        j = label_to_idx.get(r.label)
        if j is None:
            targets_for_weights.append("_UNKNOWN")
            continue
        Y[i, j] = 1.0
        targets_for_weights.append(str(r.label))
    sw, weight_diag = _safe_balanced_sample_weights(targets_for_weights)
    if not weight_diag.get("ok", False):
        return {
            "enabled": False,
            "reason": str(weight_diag.get("reason", "no_sample_weights")),
            "diagnostics": weight_diag,
        }
    lam = float(LINEAR_RIDGE_L2)
    W, ridge_diag = _safe_ridge_weights(X_aug, Y, sw, lam)
    if W is None:
        return {
            "enabled": False,
            "reason": str(ridge_diag.get("reason", "ridge_failed")),
            "diagnostics": {"matrix": matrix_diag, "weights": weight_diag, "ridge": ridge_diag},
        }
    return {
        "enabled": True,
        "model_type": "balanced_linear_ridge_ovr_numpy",
        "labels": list(labels),
        "weights": W.astype(float).tolist(),
        "global_mean": np.asarray(global_mean, dtype=np.float32).astype(float).tolist(),
        "global_std": np.asarray(global_std, dtype=np.float32).astype(float).tolist(),
        "feature_weights": FEATURE_WEIGHTS.astype(float).tolist(),
        "l2": lam,
        "numeric_diagnostics": {"matrix": matrix_diag, "weights": weight_diag, "ridge": ridge_diag},
        "training_rows": int(len(rows)),
        "label_count": int(len(labels)),
    }


def learned_parent_route_for_label(label: str) -> str:
    """Return a learned parent-neighborhood route target from a folder label.

    This uses only the trusted folder path.  It strips terminal structure leaves
    and then drops the leaf identity, so related learned leaves can share a
    parent route without hard-coded category names.
    """
    parts = [p.strip() for p in public_label(label).replace("\\", "/").split("/") if p.strip()]
    if not parts:
        return "_UNKNOWN"
    if parts[-1].strip().lower() in {"one shots", "one shot", "loops", "loop", "long fx"}:
        parts = parts[:-1]
    if len(parts) >= 2:
        parts = parts[:-1]
    return "/".join(parts) if parts else "_UNKNOWN"


def build_balanced_linear_ridge_head_for_targets(
    rows: List[FeatureRow],
    targets: Sequence[str],
    global_mean: np.ndarray,
    global_std: np.ndarray,
    *,
    head_name: str,
) -> Dict[str, object]:
    """Build a balanced one-vs-rest ridge head for arbitrary route targets.

    Used by the front-end router brain.  Unlike the prototype brain, this head
    learns a linear boundary over all features with inverse-support weighting.
    That gives the sorter a different kind of evidence for broad routing and
    tie-breaking.
    """
    if not rows or len(rows) != len(targets):
        return {"enabled": False, "reason": "target_length_mismatch", "head_name": head_name}
    clean = [(r, str(t or "_UNKNOWN")) for r, t in zip(rows, targets) if r.read_status == "ok" and str(t or "").strip()]
    classes = sorted({t for _r, t in clean})
    if len(clean) < 2 or len(classes) < 2:
        return {
            "enabled": False,
            "reason": "need_at_least_two_route_classes",
            "head_name": head_name,
            "classes": classes,
        }
    class_to_idx = {c: i for i, c in enumerate(classes)}
    X, matrix_diag = _safe_linear_design_matrix([r.fingerprint for r, _t in clean], global_mean, global_std)
    if not matrix_diag.get("ok", False):
        return {
            "enabled": False,
            "reason": str(matrix_diag.get("reason", "bad_design_matrix")),
            "head_name": head_name,
            "diagnostics": matrix_diag,
        }
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    counts = Counter(t for _r, t in clean)
    Y = np.zeros((X_aug.shape[0], len(classes)), dtype=np.float64)
    targets_for_weights: List[str] = []
    for i, (_r, target) in enumerate(clean):
        j = class_to_idx[target]
        Y[i, j] = 1.0
        targets_for_weights.append(str(target))
    sw, weight_diag = _safe_balanced_sample_weights(targets_for_weights)
    if not weight_diag.get("ok", False):
        return {
            "enabled": False,
            "reason": str(weight_diag.get("reason", "no_sample_weights")),
            "head_name": head_name,
            "diagnostics": weight_diag,
        }
    lam = float(LINEAR_RIDGE_L2)
    W, ridge_diag = _safe_ridge_weights(X_aug, Y, sw, lam)
    if W is None:
        return {
            "enabled": False,
            "reason": str(ridge_diag.get("reason", "ridge_failed")),
            "head_name": head_name,
            "diagnostics": {"matrix": matrix_diag, "weights": weight_diag, "ridge": ridge_diag},
        }
    return {
        "enabled": True,
        "head_name": head_name,
        "model_type": "frontend_balanced_linear_ridge_ovr_numpy",
        "labels": classes,
        "weights": W.astype(float).tolist(),
        "global_mean": np.asarray(global_mean, dtype=np.float32).astype(float).tolist(),
        "global_std": np.asarray(global_std, dtype=np.float32).astype(float).tolist(),
        "feature_weights": FEATURE_WEIGHTS.astype(float).tolist(),
        "l2": lam,
        "numeric_diagnostics": {"matrix": matrix_diag, "weights": weight_diag, "ridge": ridge_diag},
        "training_rows": int(len(clean)),
        "class_count": int(len(classes)),
        "class_counts": {k: int(v) for k, v in sorted(counts.items())},
    }


def _fact_profile_strength(n: int) -> str:
    if n >= FACT_STRONG_MIN_COUNT:
        return "strong"
    if n >= FACT_GOOD_MIN_COUNT:
        return "good"
    if n >= FACT_OK_MIN_COUNT:
        return "ok"
    if n >= FACT_TENTATIVE_MIN_COUNT:
        return "tentative"
    return "weak"


def compute_category_fact_profiles(
    good_rows: List[FeatureRow],
    feature_names: List[str],
) -> Dict[str, object]:
    """Build per-label, per-feature robust statistics from effective training rows.

    Facts are derived from raw fingerprint values — absolute audio measurements —
    not from the normalized vectors used for distance scoring. This preserves
    physical interpretability: sub_bass_ratio of 0.4 means 40 percent of energy
    below 150 Hz regardless of how training data is scaled.

    Each feature gets: median, p10, p25, p75, p90, IQR, MAD, reliability, and
    a valid sample count. Reliability scores how consistent the feature is within
    the label. High MAD relative to IQR means the feature is noisy and should
    carry less weight in fact scoring. Labels with fewer than FACT_MIN_ELIGIBLE_COUNT
    eligible rows get a weak profile and are excluded from contrast scoring.
    """
    by_label: Dict[str, List[np.ndarray]] = defaultdict(list)
    for row in good_rows:
        if row.read_status == "ok":
            fp = np.asarray(row.fingerprint, dtype=np.float32)
            if fp.size == len(feature_names):
                by_label[row.label].append(fp)

    profiles: Dict[str, object] = {}
    for label, fps in sorted(by_label.items()):
        n = len(fps)
        strength = _fact_profile_strength(n)
        eligible = min(n, 120)
        arr = np.vstack(fps[:eligible]).astype(np.float32)

        feature_stats: Dict[str, object] = {}
        reliable_count = 0
        weak_count = 0
        for fi, fname in enumerate(feature_names):
            if fi >= arr.shape[1]:
                continue
            col = arr[:, fi]
            valid_mask = np.isfinite(col)
            valid = col[valid_mask]
            n_valid = int(valid.size)
            if n_valid < 1:
                feature_stats[fname] = {"valid_count": 0, "reliability": 0.0}
                weak_count += 1
                continue
            median = float(np.median(valid))
            p10 = float(np.percentile(valid, 10))
            p25 = float(np.percentile(valid, 25))
            p75 = float(np.percentile(valid, 75))
            p90 = float(np.percentile(valid, 90))
            iqr = float(p75 - p25)
            mad = float(np.median(np.abs(valid - median)))
            spread = max(p90 - p10, iqr, 1e-9)
            reliability = float(np.clip(1.0 - (2.0 * mad / (spread + 1e-9)), 0.0, 1.0))
            if reliability >= RIVAL_CONTRAST_MIN_RELIABILITY:
                reliable_count += 1
            else:
                weak_count += 1
            feature_stats[fname] = {
                "median": round(median, 6),
                "p10": round(p10, 6),
                "p25": round(p25, 6),
                "p75": round(p75, 6),
                "p90": round(p90, 6),
                "iqr": round(iqr, 6),
                "mad": round(mad, 6),
                "reliability": round(reliability, 4),
                "valid_count": n_valid,
            }

        structure_counter: Counter = Counter()
        for row in good_rows:
            if row.read_status == "ok" and row.label == label:
                structure_counter[row.structure or label_default_structure(row.label)] += 1
        total_struct = max(1, sum(structure_counter.values()))

        profiles[label] = {
            "raw_count": n,
            "eligible_count": eligible,
            "effective_count": eligible,
            "fact_profile_strength": strength,
            "reliable_feature_count": reliable_count,
            "weak_feature_count": weak_count,
            "structure_stats": {
                "one_shot_ratio": round(float(structure_counter.get("one_shot", 0)) / total_struct, 4),
                "loop_ratio": round(float(structure_counter.get("loop", 0)) / total_struct, 4),
            },
            "feature_stats": feature_stats,
        }

    return profiles


def compute_category_rival_contrast_facts(
    fact_profiles: Dict[str, object],
    centroids: Dict[str, object],
    top_by_label: Dict[str, str],
    weights: np.ndarray,
    feature_names: List[str],
    max_rivals: int = 6,
) -> Dict[str, object]:
    """For each label, find nearest rival labels and compute discriminating features.

    Rivals are found by centroid distance in weighted feature space. For each
    label-rival pair, features are ranked by a robust effect size — the difference
    in medians divided by the sum of MADs. High absolute effect size means the
    feature reliably separates the two labels. Only labels with at least a
    'tentative' fact profile participate so weak/empty profiles cannot distort
    contrast scoring with made-up ranges.
    """
    labels = sorted(fact_profiles.keys())
    if len(labels) < 2:
        return {}

    label_mean_centroid: Dict[str, np.ndarray] = {}
    for label in labels:
        c = np.asarray(centroids.get(label, []), dtype=np.float32)
        if c.ndim == 1 and c.size:
            c = c.reshape(1, -1)
        if c.size:
            label_mean_centroid[label] = np.mean(c, axis=0) * weights
        else:
            label_mean_centroid[label] = np.zeros(len(feature_names), dtype=np.float32)

    rivalry: Dict[str, object] = {}
    for label in labels:
        profile = fact_profiles.get(label)
        if not isinstance(profile, dict):
            continue
        if _fact_profile_strength(int(profile.get("eligible_count", 0))) == "weak":
            continue

        lc = label_mean_centroid.get(label)
        if lc is None:
            continue

        rival_dists: List[Tuple[str, float]] = []
        for other in labels:
            if other == label:
                continue
            oc = label_mean_centroid.get(other)
            if oc is None:
                continue
            rival_dists.append((other, float(np.linalg.norm(lc - oc))))
        rival_dists.sort(key=lambda t: t[1])

        label_rivalry: Dict[str, object] = {}
        for rival, rival_dist in rival_dists[:max_rivals]:
            rival_profile = fact_profiles.get(rival)
            if not isinstance(rival_profile, dict):
                continue
            if _fact_profile_strength(int(rival_profile.get("eligible_count", 0))) == "weak":
                continue

            label_fstats = profile.get("feature_stats", {})
            rival_fstats = rival_profile.get("feature_stats", {})
            discriminators: List[Dict[str, object]] = []

            for fname in feature_names:
                lfs = label_fstats.get(fname)
                rfs = rival_fstats.get(fname)
                if not isinstance(lfs, dict) or not isinstance(rfs, dict):
                    continue
                l_median = float(lfs.get("median", 0.0))
                r_median = float(rfs.get("median", 0.0))
                l_mad = float(lfs.get("mad", 0.0))
                r_mad = float(rfs.get("mad", 0.0))
                combined_rel = float(min(lfs.get("reliability", 0.0), rfs.get("reliability", 0.0)))
                if combined_rel < RIVAL_CONTRAST_MIN_RELIABILITY:
                    continue
                effect_size = (l_median - r_median) / max(l_mad + r_mad, 1e-9)
                if abs(effect_size) < RIVAL_CONTRAST_MIN_EFFECT_SIZE:
                    continue
                discriminators.append(
                    {
                        "feature": fname,
                        "direction": "label_higher" if effect_size > 0 else "rival_higher",
                        "robust_effect_size": round(abs(effect_size), 4),
                        "label_median": round(l_median, 6),
                        "rival_median": round(r_median, 6),
                        "label_mad": round(l_mad, 6),
                        "rival_mad": round(r_mad, 6),
                        "reliability": round(combined_rel, 4),
                    }
                )

            discriminators.sort(key=lambda d: -float(d["robust_effect_size"]))
            for rank, d in enumerate(discriminators[:10], 1):
                d["rank"] = rank
            if discriminators:
                label_rivalry[rival] = {
                    "rival_distance": round(rival_dist, 6),
                    "rival_top": top_by_label.get(rival, ""),
                    "best_discriminators": discriminators[:10],
                }

        if label_rivalry:
            rivalry[label] = label_rivalry

    return rivalry


def score_sample_against_label_facts(
    fingerprint: Sequence[float],
    fact_profile: Dict[str, object],
    feature_names: List[str],
) -> float:
    """Score a sample against a label's fact profile.

    Returns a score in [-1, +1]. Positive means the sample's measured features
    agree with the learned fact distribution for this label. Negative means
    the sample's features are atypical for this label.

    Scoring per feature:
      inside p25-p75 on a reliable feature  → strong positive
      inside p10-p90 (not p25-p75)           → moderate positive
      outside p10-p90 by less than IQR       → small penalty
      outside p10-p90 by more than IQR       → strong penalty

    Weighted by reliability so noisy features contribute less.
    Returns 0.0 if the fact profile is too weak to score reliably.
    """
    n = int(fact_profile.get("eligible_count", 0) or 0)
    if n < FACT_MIN_ELIGIBLE_COUNT:
        return 0.0

    feature_stats = fact_profile.get("feature_stats", {})
    if not isinstance(feature_stats, dict):
        return 0.0

    fp = np.asarray(fingerprint, dtype=np.float32)
    total_weight = 0.0
    weighted_score = 0.0

    for fi, fname in enumerate(feature_names):
        if fi >= fp.size:
            continue
        fstats = feature_stats.get(fname)
        if not isinstance(fstats, dict):
            continue
        reliability = float(fstats.get("reliability", 0.0))
        if reliability < RIVAL_CONTRAST_MIN_RELIABILITY:
            continue
        valid_count = int(fstats.get("valid_count", 0))
        if valid_count < FACT_MIN_ELIGIBLE_COUNT:
            continue

        val = float(fp[fi])
        if not math.isfinite(val):
            continue

        p10 = float(fstats.get("p10", 0.0))
        p25 = float(fstats.get("p25", 0.0))
        p75 = float(fstats.get("p75", 0.0))
        p90 = float(fstats.get("p90", 0.0))
        iqr = float(fstats.get("iqr", 0.0))

        if p25 <= val <= p75:
            score = 2.0
        elif p10 <= val <= p90:
            score = 1.0
        elif (val < p10 and (p10 - val) <= iqr) or (val > p90 and (val - p90) <= iqr):
            score = -0.5
        else:
            score = -1.5

        total_weight += 2.0 * reliability
        weighted_score += score * reliability

    if total_weight < 1e-9:
        return 0.0
    raw = weighted_score / total_weight
    return float(np.clip(raw, -1.0, 1.0))


def score_sample_against_rival_contrast_facts(
    fingerprint: Sequence[float],
    rival_contrast: Dict[str, object],
    feature_names: List[str],
    matched_facts: List[str],
    failed_facts: List[str],
) -> float:
    """Score a sample against the discriminating features between a label and one rival.

    Returns a score in [-1, +1]. Positive means the sample's features favor the
    label over the rival on the key discriminating dimensions. Negative means the
    sample looks more like the rival.

    Writes matched and failed fact names into the provided lists for the manifest.
    """
    discriminators = rival_contrast.get("best_discriminators", [])
    if not discriminators:
        return 0.0

    fp = np.asarray(fingerprint, dtype=np.float32)
    name_to_idx = {fname: fi for fi, fname in enumerate(feature_names)}

    total_weight = 0.0
    weighted_score = 0.0

    for disc in discriminators[:8]:
        fname = str(disc.get("feature", ""))
        fi = name_to_idx.get(fname, -1)
        if fi < 0 or fi >= fp.size:
            continue
        val = float(fp[fi])
        if not math.isfinite(val):
            continue

        direction = str(disc.get("direction", ""))
        effect_size = float(disc.get("robust_effect_size", 0.0))
        reliability = float(disc.get("reliability", 0.0))
        label_median = float(disc.get("label_median", 0.0))
        rival_median = float(disc.get("rival_median", 0.0))
        midpoint = (label_median + rival_median) / 2.0
        w = effect_size * reliability

        favors_label = val > midpoint if direction == "label_higher" else val < midpoint

        if favors_label:
            weighted_score += w
            matched_facts.append(fname)
        else:
            weighted_score -= w
            failed_facts.append(fname)
        total_weight += w

    if total_weight < 1e-9:
        return 0.0
    raw = weighted_score / total_weight
    return float(np.clip(raw, -1.0, 1.0))


def build_frontend_router_brain(
    rows: List[FeatureRow], global_mean: np.ndarray, global_std: np.ndarray
) -> Dict[str, object]:
    """Train the v0.4.81 numerically safe front-end router brain.

    Heads:
      - top family: Drums/Instruments/FX/Textures as learned from folder paths
      - structure: one_shot vs loop from learned training row structure
      - parent neighborhood: learned folder parent after dropping leaf identity
    """
    if not FRONTEND_ROUTER_BRAIN_ENABLED_DEFAULT:
        return {"enabled": False, "reason": "disabled"}
    top_targets = [str(r.top or top_for_public_label(r.label)) for r in rows]
    structure_targets = [str(r.structure or label_default_structure(r.label)) for r in rows]
    parent_targets = [learned_parent_route_for_label(r.label) for r in rows]
    return {
        "enabled": True,
        "model_type": "frontend_router_three_head_balanced_linear",
        "top_head": build_balanced_linear_ridge_head_for_targets(
            rows, top_targets, global_mean, global_std, head_name="top_family"
        ),
        "structure_head": build_balanced_linear_ridge_head_for_targets(
            rows, structure_targets, global_mean, global_std, head_name="structure"
        ),
        "parent_head": build_balanced_linear_ridge_head_for_targets(
            rows, parent_targets, global_mean, global_std, head_name="learned_parent"
        ),
        "policy": "Soft tie-breaker and broad router. Never reads filenames and never invents categories.",
    }


def build_brain(train_features: List[FeatureRow], max_centroids: int) -> dict:
    raw_good = [r for r in train_features if r.read_status == "ok"]
    if not raw_good:
        raise SystemExit("No readable training rows.")

    if BALANCED_EFFECTIVE_TRAINING_ENABLED_DEFAULT:
        good, effective_balance_by_label = select_balanced_effective_training_rows(raw_good)
    else:
        good = list(raw_good)
        effective_balance_by_label = {
            label: {
                "label": label,
                "top": rows[0].top if rows else "",
                "raw_count": len(rows),
                "effective_count": len(rows),
                "cap_applied": "no",
                "effective_cap": len(rows),
                "selection_method": "balanced_effective_training_disabled",
            }
            for label, rows in defaultdict(list, {}).items()
        }

    if not good:
        raise SystemExit("Balanced effective training selected no readable rows.")

    raw_counts_by_label = Counter(r.label for r in raw_good)
    effective_counts_by_label = Counter(r.label for r in good)

    X = np.asarray([r.fingerprint for r in good], dtype=np.float32)
    global_mean = np.mean(X, axis=0).astype(np.float32)
    global_std = np.std(X, axis=0).astype(np.float32)
    global_std = np.where(global_std < 1e-6, 1.0, global_std).astype(np.float32)

    raw_by_structure: Dict[str, List[np.ndarray]] = defaultdict(list)
    for row in good:
        raw_by_structure[row.structure or label_default_structure(row.label)].append(
            np.asarray(row.fingerprint, dtype=np.float32)
        )

    scalers_by_structure: Dict[str, Dict[str, object]] = {}
    for structure, vecs in sorted(raw_by_structure.items()):
        arr = np.vstack(vecs).astype(np.float32)
        mean = np.mean(arr, axis=0).astype(np.float32)
        std = np.std(arr, axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        scalers_by_structure[structure] = {
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "count": int(arr.shape[0]),
        }

    by_label: Dict[str, List[np.ndarray]] = defaultdict(list)
    counts = Counter()
    top_by_label = {}
    structure_votes_by_label: Dict[str, Counter] = defaultdict(Counter)
    examples_by_label: Dict[str, List[str]] = defaultdict(list)
    training_examples_detailed_by_label: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    active_status_by_label: Dict[str, str] = {}
    clean_available_by_label: Dict[str, int] = {}
    source_group_count_by_label: Dict[str, int] = {}

    for row in good:
        structure = row.structure or label_default_structure(row.label)
        scaler = scalers_by_structure.get(structure)
        if scaler:
            mean = np.asarray(scaler["mean"], dtype=np.float32)
            std = np.asarray(scaler["std"], dtype=np.float32)
        else:
            mean = global_mean
            std = global_std
        z = (np.asarray(row.fingerprint, dtype=np.float32) - mean) / std
        by_label[row.label].append(np.asarray(z, dtype=np.float32).reshape(-1))
        counts[row.label] += 1
        top_by_label[row.label] = row.top
        structure_votes_by_label[row.label][structure] += 1
        active_status_by_label[row.label] = row.training_active_status or active_status_by_label.get(
            row.label, "ACTIVE_UNKNOWN"
        )
        clean_available_by_label[row.label] = max(
            clean_available_by_label.get(row.label, 0), int(row.label_clean_available or 0)
        )
        source_group_count_by_label[row.label] = max(
            source_group_count_by_label.get(row.label, 0), int(row.label_source_group_count or 0)
        )
        if len(examples_by_label[row.label]) < 5:
            examples_by_label[row.label].append(row.path)
        if len(training_examples_detailed_by_label[row.label]) < 120:
            training_examples_detailed_by_label[row.label].append(
                {
                    "source_path": row.path,
                    "group_key": row.group_key,
                    "top": row.top,
                    "structure": structure,
                    "duration_sec": float(row.duration_sec),
                    "source_pack": row.source_pack or source_pack_key_for_path(row.path),
                    "physics_tags": fingerprint_physics_tags(row.fingerprint, row.duration_sec),
                    "physics_summary": fingerprint_physics_summary(row.fingerprint, row.duration_sec),
                    "fingerprint": [float(x) for x in row.fingerprint],
                }
            )

    centroids = {}
    exemplars_by_label = {}
    anchors_by_label = {}
    centroid_counts = {}
    label_models_by_label = {}
    label_reliability_by_label = {}
    for label, vecs in sorted(by_label.items()):
        arr = np.vstack(vecs).astype(np.float32)
        model = choose_adaptive_label_model(label, arr, max_centroids=max_centroids)
        k = int(model["centroid_count"])
        centroid_counts[label] = int(k)
        label_models_by_label[label] = dict(model)
        if str(model.get("model_mode")) == "exemplar_only":
            centroids[label] = select_deterministic_exemplars(arr, k).tolist()
        else:
            centroids[label] = deterministic_centroids(arr, max_k=k).tolist()
        exemplar_count = min(int(model["exemplar_count"]), MAX_EXEMPLARS_PER_LABEL_CAP, int(arr.shape[0]))
        exemplars_by_label[label] = select_deterministic_exemplars(arr, exemplar_count).tolist()
        # Teacher anchors preserve recall for tiny/messy labels, but are capped
        # so giant folders do not get hundreds of nearest-neighbor lottery tickets.
        anchor_count = min(MAX_ANCHORS_PER_LABEL, int(arr.shape[0]))
        anchors_by_label[label] = select_deterministic_exemplars(arr, anchor_count).astype(float).tolist()
        if label in label_models_by_label:
            label_models_by_label[label]["exemplar_count"] = int(exemplar_count)
            label_models_by_label[label]["anchor_count"] = int(anchor_count)
        label_status = active_status_by_label.get(label, "ACTIVE_UNKNOWN")
        label_source_groups = int(source_group_count_by_label.get(label, 0) or 0)
        tier, req_similarity, req_margin = label_reliability_requirements(
            int(counts[label]), label_status, label_source_groups
        )
        label_reliability_by_label[label] = {
            "training_count": int(counts[label]),
            "effective_training_count": int(effective_counts_by_label.get(label, counts[label])),
            "raw_training_count": int(raw_counts_by_label.get(label, counts[label])),
            "clean_available": int(
                clean_available_by_label.get(label, raw_counts_by_label.get(label, counts[label]))
                or raw_counts_by_label.get(label, counts[label])
            ),
            "source_group_count": label_source_groups,
            "active_status": label_status,
            "centroid_count": int(k),
            "model_mode": str(model.get("model_mode", "")),
            "exemplar_count": int(model.get("exemplar_count", 0) or 0),
            "spread_mean": float(model.get("spread_mean", 0.0) or 0.0),
            "spread_p90": float(model.get("spread_p90", 0.0) or 0.0),
            "support_tier": tier,
            "min_similarity_for_auto_place": float(req_similarity),
            "min_margin_for_auto_place": float(req_margin),
        }

    labels = sorted(by_label.keys())

    linear_ridge_head = build_linear_ridge_head(good, labels, global_mean, global_std)

    # Learn broad top-category contest spaces from the brain itself.
    # These do not use filenames or hand-written routing rules.  They let pure
    # prediction first ask "which learned top families are plausible?" before
    # the more specific label contest.  Top 3 are allowed during prediction so
    # this cannot hide the correct family too aggressively while the brain is young.
    super_centroids_by_top_structure: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    for top in sorted(set(top_by_label.values())):
        structures = sorted(
            set(
                (
                    structure_votes_by_label[label].most_common(1)[0][0]
                    if structure_votes_by_label[label]
                    else label_default_structure(label)
                )
                for label in labels
                if top_by_label.get(label, "") == top
            )
        )
        for structure in structures:
            top_vecs = []
            for label in labels:
                if top_by_label.get(label, "") != top:
                    continue
                label_structure = (
                    structure_votes_by_label[label].most_common(1)[0][0]
                    if structure_votes_by_label[label]
                    else label_default_structure(label)
                )
                if label_structure != structure:
                    continue
                # Use one vote per learned label, not one vote per raw file.
                # Otherwise giant labels dominate the broad top-family prototype.
                label_centroids = np.asarray(centroids.get(label, []), dtype=np.float32)
                if label_centroids.ndim == 1:
                    label_centroids = label_centroids.reshape(1, -1)
                if label_centroids.size:
                    top_vecs.append(np.mean(label_centroids, axis=0))
                elif by_label.get(label):
                    top_vecs.append(np.mean(np.asarray(by_label[label], dtype=np.float32), axis=0))
            if top_vecs:
                super_centroids_by_top_structure[top][structure] = (
                    np.mean(np.vstack(top_vecs), axis=0).astype(float).tolist()
                )

    # Learned structure head.  This is brain math, not a filename/duration rule.
    # A mystery file is compared against learned One Shots and Loops prototypes,
    # then label scoring gets a soft structure-distance penalty.
    structure_centroids = {}
    for structure, raw_vecs in sorted(raw_by_structure.items()):
        scaler = scalers_by_structure.get(structure)
        if scaler:
            mean = np.asarray(scaler["mean"], dtype=np.float32)
            std = np.asarray(scaler["std"], dtype=np.float32)
        else:
            mean = global_mean
            std = global_std
        arr = np.vstack([((np.asarray(v, dtype=np.float32) - mean) / std) for v in raw_vecs]).astype(np.float32)
        k = 1 if arr.shape[0] <= 4 else (2 if arr.shape[0] <= 15 else (3 if arr.shape[0] <= 60 else 4))
        structure_centroids[structure] = deterministic_centroids(arr, max_k=k).astype(float).tolist()

    # Structure-relative calibration. The structure head uses its own weighted
    # feature space, so raw one-shot/loop gaps cannot be added directly to label
    # distances. Store a typical within-structure distance and normalize gaps
    # during prediction before applying the small structure penalty.
    typical_structure_dist: Dict[str, float] = {}
    structure_weights = STRUCTURE_FEATURE_WEIGHTS.astype(np.float32)
    for structure, raw_vecs in sorted(raw_by_structure.items()):
        scaler = scalers_by_structure.get(structure)
        if scaler:
            mean = np.asarray(scaler["mean"], dtype=np.float32)
            std = np.asarray(scaler["std"], dtype=np.float32)
        else:
            mean = global_mean
            std = global_std
        cent = np.asarray(structure_centroids.get(structure, []), dtype=np.float32)
        if cent.ndim == 1:
            cent = cent.reshape(1, -1)
        if cent.size == 0:
            continue
        cw = cent * structure_weights[None, :]
        dists = []
        for v in raw_vecs:
            xw = ((np.asarray(v, dtype=np.float32) - mean) / std) * structure_weights
            dists.append(float(np.min(np.linalg.norm(cw - xw[None, :], axis=1))))
        if dists:
            typical_structure_dist[structure] = float(max(0.1, np.mean(dists)))

    # Category-relative calibration. FX, textures, drums, and instruments occupy
    # different natural spreads, so raw distance should not use one global scale.
    typical_dist_by_top: Dict[str, List[float]] = defaultdict(list)
    weights = FEATURE_WEIGHTS.astype(np.float32)
    for label, vecs in by_label.items():
        top = top_by_label.get(label, top_for_public_label(label))
        for v in vecs:
            xw = np.asarray(v, dtype=np.float32) * weights
            _rank, raw_d, _mode = label_model_distance_score(
                {
                    "feature_weights": FEATURE_WEIGHTS.astype(float).tolist(),
                    "centroids": centroids,
                    "exemplars_by_label": exemplars_by_label,
                    "label_models_by_label": label_models_by_label,
                },
                label,
                xw,
            )
            typical_dist_by_top[top].append(float(raw_d))
    # v0.4.72 similarity calibration:
    # Training-recall distances can be near zero when labels keep exact teacher
    # anchors. Using that mean as the similarity scale makes every unseen file
    # look falsely low-confidence. Calibrate the top-family similarity scale from
    # both teacher recall distance and learned label spread, so a small but coherent
    # folder can generalize to a nearby unseen sample without faking certainty.
    spread_scale_by_top: Dict[str, List[float]] = defaultdict(list)
    for lbl, model in label_models_by_label.items():
        top = top_by_label.get(lbl, top_for_public_label(lbl))
        if isinstance(model, dict):
            spread_scale_by_top[top].append(
                float(
                    max(
                        float(model.get("spread_mean", 0.0) or 0.0),
                        float(model.get("spread_p90", 0.0) or 0.0),
                    )
                )
            )
    typical_dist_mean_by_top = {}
    for top, ds in sorted(typical_dist_by_top.items()):
        recall_scale = float(np.mean(ds)) if ds else 0.0
        spread_values = [v for v in spread_scale_by_top.get(top, []) if math.isfinite(v) and v > 0.0]
        spread_scale = float(np.median(spread_values)) if spread_values else 0.0
        typical_dist_mean_by_top[top] = float(max(1.0, recall_scale, spread_scale))

    label_intra_spread_by_label: Dict[str, Dict[str, float]] = {}
    for label, vecs in sorted(by_label.items()):
        dists = []
        for v in vecs:
            xw = np.asarray(v, dtype=np.float32) * weights
            _rank, raw_d, _mode = label_model_distance_score(
                {
                    "feature_weights": FEATURE_WEIGHTS.astype(float).tolist(),
                    "centroids": centroids,
                    "exemplars_by_label": exemplars_by_label,
                    "label_models_by_label": label_models_by_label,
                },
                label,
                xw,
            )
            dists.append(float(raw_d))
        if dists:
            arr = np.asarray(dists, dtype=np.float32)
            top = top_by_label.get(label, top_for_public_label(label))
            top_typical = float(typical_dist_mean_by_top.get(top, 1.0) or 1.0)
            label_intra_spread_by_label[label] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)),
                "max": float(np.max(arr)),
                "relative_to_top_mean": float(np.mean(arr) / max(0.1, top_typical)),
            }
            if label in label_reliability_by_label:
                label_reliability_by_label[label].update(
                    {
                        "intra_spread_mean": label_intra_spread_by_label[label]["mean"],
                        "intra_spread_p90": label_intra_spread_by_label[label]["p90"],
                        "intra_spread_relative_to_top_mean": label_intra_spread_by_label[label]["relative_to_top_mean"],
                    }
                )

    # v0.5.5: Build category fact profiles, rival contrast facts, and second-pass rival audit support.
    # Facts are derived from raw fingerprints — not normalized vectors — so they
    # preserve physical meaning (band ratios, ZCR, etc. in absolute terms).
    # This must happen after centroids are built so rival distance can use centroid space.
    weights_for_contrast = np.asarray(FEATURE_WEIGHTS, dtype=np.float32)
    category_fact_profiles = compute_category_fact_profiles(good, FEATURE_NAMES)
    category_rival_contrast_facts = compute_category_rival_contrast_facts(
        category_fact_profiles,
        centroids,
        top_by_label,
        weights_for_contrast,
        FEATURE_NAMES,
        max_rivals=RIVAL_CONTRAST_MAX_RIVALS,
    )

    brain = {
        "brain_type": "phase3_pure_prototype_brain",
        "version": "0.5.5-committee-physics-agreement",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Built by Aaron Sound Sorter Stage 4 v0.5.7 COMMITTEE_PHYSICS_GAP_LOCKS. This brain uses a 100-feature fingerprint: the stable first 50 features plus appended harmonic/pitch identity, attack/body/tail split, noise-burst/decay, low-end source identity, and spectral peak/formant-like descriptors; per-structure label-side scalers; adaptive per-folder model selection, centroid/exemplar anchors, balanced effective training, capped giant-folder anchors, support-aware folder balancing, category fact profiles, and rival contrast scoring. It does not use Aaron_Sound_Sorter.py runtime judge or filename routing.",
        "feature_names": FEATURE_NAMES,
        "feature_size": FP_SIZE,
        "feature_weights": FEATURE_WEIGHTS.astype(float).tolist(),
        "scaler_mean": global_mean.astype(float).tolist(),
        "scaler_std": global_std.astype(float).tolist(),
        "scalers_by_structure": scalers_by_structure,
        "labels": labels,
        # counts are effective competitive counts by design. Raw folder support is kept separately.
        "counts": {k: int(counts[k]) for k in labels},
        "effective_counts": {k: int(effective_counts_by_label.get(k, counts[k])) for k in labels},
        "raw_counts": {k: int(raw_counts_by_label.get(k, counts[k])) for k in labels},
        "effective_training_balance_by_label": effective_balance_by_label,
        "balanced_effective_training_enabled": bool(BALANCED_EFFECTIVE_TRAINING_ENABLED_DEFAULT),
        "contradiction_filter_effective_training_enabled": bool(
            CONTRADICTION_FILTER_EFFECTIVE_TRAINING_ENABLED_DEFAULT
        ),
        "effective_train_central_fraction": float(EFFECTIVE_TRAIN_CENTRAL_FRACTION),
        "effective_train_diverse_fraction": float(EFFECTIVE_TRAIN_DIVERSE_FRACTION),
        "effective_train_edge_fraction": float(EFFECTIVE_TRAIN_EDGE_FRACTION),
        "effective_train_target_per_label": int(EFFECTIVE_TRAIN_TARGET_PER_LABEL),
        "effective_train_max_per_label": int(EFFECTIVE_TRAIN_MAX_PER_LABEL),
        "effective_train_huge_threshold": int(EFFECTIVE_TRAIN_HUGE_LABEL_THRESHOLD),
        "effective_train_huge_max_per_label": int(EFFECTIVE_TRAIN_HUGE_MAX_PER_LABEL),
        "max_anchors_per_label": int(MAX_ANCHORS_PER_LABEL),
        "max_exemplars_per_label_cap": int(MAX_EXEMPLARS_PER_LABEL_CAP),
        "top_by_label": {k: top_by_label.get(k, "") for k in labels},
        "structure_by_label": {
            k: (
                structure_votes_by_label[k].most_common(1)[0][0]
                if structure_votes_by_label[k]
                else label_default_structure(k)
            )
            for k in labels
        },
        "centroid_counts": {k: int(centroid_counts.get(k, 0)) for k in labels},
        "label_models_by_label": label_models_by_label,
        "label_reliability_by_label": label_reliability_by_label,
        "centroids": centroids,
        "exemplars_by_label": exemplars_by_label,
        "anchors_by_label": anchors_by_label,
        "examples_by_label": examples_by_label,
        "training_examples_detailed_by_label": {k: v for k, v in training_examples_detailed_by_label.items()},
        "super_centroids_by_top_structure": {top: dict(v) for top, v in super_centroids_by_top_structure.items()},
        "structure_centroids": structure_centroids,
        "structure_feature_weights": STRUCTURE_FEATURE_WEIGHTS.astype(float).tolist(),
        "typical_structure_dist": typical_structure_dist,
        "structure_score_weight": 0.20,
        "structure_review_gap": 1.00,
        "typical_dist_mean_by_top": typical_dist_mean_by_top,
        "label_intra_spread_by_label": label_intra_spread_by_label,
        "linear_ridge_head": linear_ridge_head,
        "top_prefilter_n": 3,
        "max_label_training_count": int(max(counts.values()) if counts else 0),
        "support_balance_enabled": SUPPORT_BALANCE_ENABLED_DEFAULT,
        "support_balance_threshold_ratio": SUPPORT_BALANCE_THRESHOLD_RATIO,
        "support_balance_max_distance_bonus": SUPPORT_BALANCE_MAX_DISTANCE_BONUS,
        "support_balance_tiny_count_full_bonus": SUPPORT_BALANCE_TINY_COUNT_FULL_BONUS,
        "support_balance_tiny_multiplier": SUPPORT_BALANCE_TINY_MULTIPLIER,
        "model_ensemble_enabled": MODEL_ENSEMBLE_ENABLED_DEFAULT,
        "model_ensemble_max_labels": MODEL_ENSEMBLE_MAX_LABELS,
        "model_ensemble_min_switch_advantage": MODEL_ENSEMBLE_MIN_SWITCH_ADVANTAGE,
        "model_ensemble_max_score_penalty_ratio": MODEL_ENSEMBLE_MAX_SCORE_PENALTY_RATIO,
        "model_ensemble_current_rank_weight": MODEL_ENSEMBLE_CURRENT_RANK_WEIGHT,
        "model_ensemble_neutral_rank_weight": MODEL_ENSEMBLE_NEUTRAL_RANK_WEIGHT,
        "model_ensemble_spread_rank_weight": MODEL_ENSEMBLE_SPREAD_RANK_WEIGHT,
        "model_ensemble_anchor_rank_weight": MODEL_ENSEMBLE_ANCHOR_RANK_WEIGHT,
        "model_ensemble_linear_rank_weight": MODEL_ENSEMBLE_LINEAR_RANK_WEIGHT,
        "model_ensemble_frontend_router_rank_weight": MODEL_ENSEMBLE_FRONTEND_ROUTER_RANK_WEIGHT,
        "frontend_router_brain_enabled": FRONTEND_ROUTER_BRAIN_ENABLED_DEFAULT,
        "frontend_router_top_penalty": FRONTEND_ROUTER_TOP_PENALTY,
        "frontend_router_structure_penalty": FRONTEND_ROUTER_STRUCTURE_PENALTY,
        "frontend_router_parent_penalty": FRONTEND_ROUTER_PARENT_PENALTY,
        "frontend_router_rank_weight": FRONTEND_ROUTER_RANK_WEIGHT,
        "linear_ridge_head_enabled": LINEAR_RIDGE_HEAD_ENABLED_DEFAULT,
        "linear_ridge_l2": LINEAR_RIDGE_L2,
        "adaptive_depth_placement_enabled": ADAPTIVE_DEPTH_PLACEMENT_ENABLED_DEFAULT,
        "adaptive_depth_max_similarity": ADAPTIVE_DEPTH_MAX_SIMILARITY,
        "adaptive_depth_max_margin": ADAPTIVE_DEPTH_MAX_MARGIN,
        "top_score_weight": 0.35,
        "top_review_gap": 0.25,
        "top_prefilter_policy": "Prediction scores learned super-centroids first and allows the closest 3 top families before label scoring. This is learned brain math, not filename/rule routing.",
        "distance_policy": "Centroids are stored scaled by each label structure scaler. Prediction scores each label with its label-side structure scaler; no expected structure is passed during pure eval.",
        "structure_policy": "A learned structure head scores One Shots versus Loops from audio features and adds a soft label-distance penalty. It does not use filenames, folder hints, or expected eval structure.",
        "label_score_policy": "Each folder chooses an internal model from its physics spread: exemplar_only for tiny labels, single_centroid for uniform labels, multi_centroid for medium-spread labels, and multi_centroid_exemplar for messy labels. Output folder remains the trusted training folder path; no fixed terminal category vocabulary is used.",
        "small_folder_recall_contract": "Labels at or below SMALL_LABEL_COUNT use nearest teacher distance for anchor scoring so exact teacher recall cannot be weakened by averaging farther neighbors. Larger labels still use neighborhood consensus.",
        "similarity_calibration_policy": "Similarity scale is calibrated from both teacher-recall distances and learned label spread, not from exact-anchor recall alone.",
        "dynamic_borderline_policy": "Global borderline review remains for medium/large labels; tiny/small labels can auto-place below 0.56 only when they satisfy their learned support threshold and have a strong margin.",
        "label_reliability_policy": "Labels trained from tiny/small pools may predict, but auto-placement requires stronger similarity/margin. This is learned support calibration, not filename or semantic routing.",
        "support_balance_policy": "Underrepresented learned folders receive a small capped ranking-distance bonus when they are below the configured support ratio. This uses only training counts and never checks category names or filenames. v0.4.67 also caps giant folders into a deterministic effective pool before fitting scaler/centroids/anchors.",
        "model_tournament_policy": "v0.4.81 compares prototype/exemplar ranking, support-neutral ranking, spread-normalized ranking, capped-anchor ranking, label linear-ridge ranking, and a front-end router rank. It may switch only when the ensemble clearly beats the current winner without being much worse under production distance. This is learned matching math, not a category or filename rescue.",
        "frontend_router_policy": "The front-end router brain learns top family, structure, and learned parent neighborhood with balanced linear heads. It is a soft route/tie-breaker before deep folder selection, not a hard-coded category rule.",
        "adaptive_depth_policy": "When top learned sibling labels are risky/close, usable audio is still sorted under the learned tree, but placed in a dynamic _Ambiguous Leaf folder under the shared learned parent instead of a false exact leaf or _TO_REVIEW dumping.",
        "build_speed_policy": "v0.4.67: train-brain skips expensive O(rows x labels) developer diagnostics by default, writes progress checkpoints during brain build, and saves the normal reusable brain at the requested save path.",
        "adaptive_model_policy": "Folder path is ground truth. Physics spread chooses only the internal model shape; it never renames, deletes, or overrides a training label.",
        "fact_scoring_policy": "v0.5.5: category_fact_profiles stores per-label, per-feature robust statistics (median, p10-p90, IQR, MAD, reliability) from raw fingerprint values. fact_scoring shifts primary ranking so first-guess already uses measured audio facts. category_rival_contrast_facts stores per-(label,rival) feature discriminators ranked by robust effect size. No instrument names or category-specific rescue rules are used.",
        "category_fact_profiles": category_fact_profiles,
        "category_rival_contrast_facts": category_rival_contrast_facts,
        "fact_scoring_enabled": FACT_SCORING_ENABLED_DEFAULT,
        "fact_score_weight": float(FACT_SCORE_WEIGHT),
        "rival_contrast_enabled": RIVAL_CONTRAST_ENABLED_DEFAULT,
        "rival_contrast_weight": float(RIVAL_CONTRAST_WEIGHT),
    }
    return brain


def raw_label_centroid_score(brain: dict, label: str, fingerprint: Sequence[float]) -> float:
    """Distance from one fingerprint to one label's centroid set.

    This uses the same label-side scaler and feature weights as prediction, but
    no top/structure penalties. It is for training diagnostics, especially
    cross-boundary teacher discovery.
    """
    centroids = brain.get("centroids", {})
    if label not in centroids:
        return float("inf")
    weights = np.asarray(brain.get("feature_weights", FEATURE_WEIGHTS), dtype=np.float32)
    mean, std = scaler_for_label(brain, label)
    x = (np.asarray(fingerprint, dtype=np.float32) - mean) / std
    xw = x * weights
    score, _raw, _mode = label_model_distance_score(brain, label, xw)
    return float(score)


def populate_label_isolation_metrics(brain: dict, train_features: List[FeatureRow]) -> None:
    """Store report-only label isolation metrics in the brain.

    This measures how much room a label has before another label's centroid
    territory starts. It does not alter prediction or auto-placement gates.
    Low isolation and high boundary-conflict fraction mean the label probably
    needs cleaner/more diverse teachers.
    """
    labels = [str(x) for x in brain.get("labels", [])]
    by_label: Dict[str, List[FeatureRow]] = defaultdict(list)
    for row in train_features:
        if row.read_status == "ok" and row.label in labels:
            by_label[row.label].append(row)

    isolation: Dict[str, Dict[str, object]] = {}
    for label in labels:
        rows = by_label.get(label, [])
        if not rows:
            continue
        own_scores: List[float] = []
        nearest_other_scores: List[float] = []
        nearest_other_counter: Counter = Counter()
        boundary_conflicts = 0
        for row in rows:
            own_score = raw_label_centroid_score(brain, label, row.fingerprint)
            best_other_label = ""
            best_other_score = float("inf")
            for other_label in labels:
                if other_label == label:
                    continue
                score = raw_label_centroid_score(brain, other_label, row.fingerprint)
                if score < best_other_score:
                    best_other_score = score
                    best_other_label = other_label
            if not math.isfinite(own_score) or not math.isfinite(best_other_score):
                continue
            own_scores.append(float(own_score))
            nearest_other_scores.append(float(best_other_score))
            if best_other_label:
                nearest_other_counter[best_other_label] += 1
            if best_other_score < own_score:
                boundary_conflicts += 1
        if not own_scores or not nearest_other_scores:
            continue
        own_mean = float(np.mean(own_scores))
        other_mean = float(np.mean(nearest_other_scores))
        isolation_score = own_mean / max(0.1, other_mean)
        boundary_fraction = boundary_conflicts / max(1, len(own_scores))
        nearest_label = nearest_other_counter.most_common(1)[0][0] if nearest_other_counter else ""
        isolation[label] = {
            "own_centroid_score_mean": own_mean,
            "nearest_other_centroid_score_mean": other_mean,
            "nearest_other_centroid_score_min": float(np.min(nearest_other_scores)),
            "label_isolation_score": float(isolation_score),
            "boundary_conflict_fraction": float(boundary_fraction),
            "most_common_nearest_other_label": nearest_label,
            "most_common_nearest_other_label_count": int(nearest_other_counter[nearest_label]) if nearest_label else 0,
        }

    brain["label_isolation_by_label"] = isolation
    reliability = brain.get("label_reliability_by_label", {})
    if isinstance(reliability, dict):
        for label, info in isolation.items():
            if label in reliability and isinstance(reliability[label], dict):
                reliability[label].update(info)


def write_training_label_spread_report(brain: dict, reports_dir: Path) -> Path:
    spread = brain.get("label_intra_spread_by_label", {})
    isolation = brain.get("label_isolation_by_label", {})
    rel = brain.get("label_reliability_by_label", {})
    rows: List[Dict[str, str]] = []
    for label in sorted(brain.get("labels", [])):
        info = spread.get(label, {}) if isinstance(spread, dict) else {}
        isolation_info = isolation.get(label, {}) if isinstance(isolation, dict) else {}
        rel_info = rel.get(label, {}) if isinstance(rel, dict) else {}
        if not isinstance(info, dict):
            info = {}
        if not isinstance(isolation_info, dict):
            isolation_info = {}
        if not isinstance(rel_info, dict):
            rel_info = {}
        rel_mean = float(info.get("relative_to_top_mean", 0.0) or 0.0)
        isolation_score = float(isolation_info.get("label_isolation_score", 0.0) or 0.0)
        boundary_fraction = float(isolation_info.get("boundary_conflict_fraction", 0.0) or 0.0)
        flags = []
        if rel_mean >= 1.35:
            flags.append("wide_label_spread")
        if isolation_score >= 0.90:
            flags.append("crowded_label_space")
        if boundary_fraction >= 0.20:
            flags.append("many_cross_boundary_teachers")
        if int(rel_info.get("training_count", 0) or 0) <= SMALL_LABEL_COUNT:
            flags.append("low_count")
        rows.append(
            {
                "internal_label": label,
                "public_label": public_label(label),
                "top": brain.get("top_by_label", {}).get(label, top_for_public_label(label)),
                "structure": brain.get("structure_by_label", {}).get(label, label_default_structure(label)),
                "training_count": str(rel_info.get("training_count", brain.get("counts", {}).get(label, ""))),
                "support_tier": str(rel_info.get("support_tier", "")),
                "intra_spread_mean": f"{float(info.get('mean', 0.0) or 0.0):.6f}",
                "intra_spread_median": f"{float(info.get('median', 0.0) or 0.0):.6f}",
                "intra_spread_p90": f"{float(info.get('p90', 0.0) or 0.0):.6f}",
                "intra_spread_max": f"{float(info.get('max', 0.0) or 0.0):.6f}",
                "relative_to_top_mean": f"{rel_mean:.6f}",
                "nearest_other_centroid_score_mean": f"{float(isolation_info.get('nearest_other_centroid_score_mean', 0.0) or 0.0):.6f}",
                "nearest_other_centroid_score_min": f"{float(isolation_info.get('nearest_other_centroid_score_min', 0.0) or 0.0):.6f}",
                "label_isolation_score": f"{isolation_score:.6f}",
                "boundary_conflict_fraction": f"{boundary_fraction:.6f}",
                "most_common_nearest_other_label": str(isolation_info.get("most_common_nearest_other_label", "")),
                "most_common_nearest_other_label_count": str(
                    isolation_info.get("most_common_nearest_other_label_count", "")
                ),
                "risk_flags": ";".join(flags),
            }
        )
    out = reports_dir / "training_label_intra_spread_report.csv"
    write_csv(
        out,
        rows,
        [
            "internal_label",
            "public_label",
            "top",
            "structure",
            "training_count",
            "support_tier",
            "intra_spread_mean",
            "intra_spread_median",
            "intra_spread_p90",
            "intra_spread_max",
            "relative_to_top_mean",
            "nearest_other_centroid_score_mean",
            "nearest_other_centroid_score_min",
            "label_isolation_score",
            "boundary_conflict_fraction",
            "most_common_nearest_other_label",
            "most_common_nearest_other_label_count",
            "risk_flags",
        ],
    )
    return out


def write_training_cross_boundary_report(
    brain: dict, train_features: List[FeatureRow], reports_dir: Path, max_rows: int = 5000
) -> Path:
    """Report teachers closer to another label than their own label.

    These examples are not auto-removed.  This report is the honest way to find
    training rows that blur boundaries such as kick/snare/bass/808 or
    clap/hat/rim without adding semantic hacks to prediction.
    """
    labels = [str(x) for x in brain.get("labels", [])]
    rows: List[Dict[str, str]] = []
    for row in train_features:
        if row.read_status != "ok" or row.label not in labels:
            continue
        scored = []
        for label in labels:
            scored.append((label, raw_label_centroid_score(brain, label, row.fingerprint)))
        scored.sort(key=lambda item: item[1])
        best_label, best_score = scored[0]
        own_score = raw_label_centroid_score(brain, row.label, row.fingerprint)
        if best_label == row.label:
            continue
        gap = own_score - best_score
        rows.append(
            {
                "source_path": row.path,
                "source_pack": row.source_pack,
                "balance_group": row.balance_group,
                "expected_internal_label": row.label,
                "expected_label": public_label(row.label),
                "expected_top": row.top,
                "expected_structure": row.structure,
                "nearest_other_internal_label": best_label,
                "nearest_other_label": public_label(best_label),
                "nearest_other_top": brain.get("top_by_label", {}).get(best_label, top_for_public_label(best_label)),
                "nearest_other_structure": brain.get("structure_by_label", {}).get(
                    best_label, label_default_structure(best_label)
                ),
                "own_score": f"{own_score:.6f}",
                "nearest_other_score": f"{best_score:.6f}",
                "score_gap_own_minus_other": f"{gap:.6f}",
                "duration_sec": f"{row.duration_sec:.6f}",
                **fingerprint_physics_dict(row.fingerprint, row.duration_sec),
                "physics_tags": fingerprint_physics_tags(row.fingerprint, row.duration_sec),
                "physics_summary": fingerprint_physics_summary(row.fingerprint, row.duration_sec),
                "top5_labels": " || ".join(f"{public_label(label)}:{score:.4f}" for label, score in scored[:5]),
            }
        )
    rows.sort(key=lambda r: float(r["score_gap_own_minus_other"]), reverse=True)
    if max_rows and len(rows) > max_rows:
        rows = rows[:max_rows]
    out = reports_dir / "training_cross_boundary_examples.csv"
    fields = [
        "source_path",
        "source_pack",
        "balance_group",
        "expected_internal_label",
        "expected_label",
        "expected_top",
        "expected_structure",
        "nearest_other_internal_label",
        "nearest_other_label",
        "nearest_other_top",
        "nearest_other_structure",
        "own_score",
        "nearest_other_score",
        "score_gap_own_minus_other",
        "duration_sec",
        *PHYSICS_REPORT_FIELDS,
        "top5_labels",
    ]
    write_csv(out, rows, fields)
    return out


def write_training_recall_contract_report(brain: dict, train_features: List[FeatureRow], reports_dir: Path) -> Path:
    """Verify recall only for active math-pool teachers.

    v0.4.82 cleanup: contradiction-filtered rows remain in the trusted tree and
    reports, but they are intentionally excluded from the competitive math pool.
    They must not be counted as recall failures. They are reported separately as
    SKIP_EXCLUDED_FROM_MATH so the report is honest and actionable.
    """
    rows: List[Dict[str, str]] = []
    fail_count = 0
    pass_count = 0
    skipped_excluded = 0
    skipped_unreadable = 0
    for row in train_features:
        if row.read_status != "ok":
            skipped_unreadable += 1
            continue
        exclusion_reason = training_row_math_exclusion_reason(row)
        if exclusion_reason:
            skipped_excluded += 1
            rows.append(
                {
                    "status": "SKIP_EXCLUDED_FROM_MATH",
                    "math_pool_exclusion_reason": exclusion_reason,
                    "source_path": row.path,
                    "expected_internal_label": row.label,
                    "expected_label": public_label(row.label),
                    "predicted_internal_label": "",
                    "predicted_label": "",
                    "predicted_top": "",
                    "similarity": "",
                    "margin_distance_gap": "",
                    "duration_sec": f"{row.duration_sec:.6f}",
                    **fingerprint_physics_dict(row.fingerprint, row.duration_sec),
                    "top5_labels": "",
                }
            )
            continue
        fingerprint_values = np.asarray(row.fingerprint, dtype=np.float32).tolist()
        predicted_label, predicted_top, sim, margin, top5 = predict(brain, fingerprint_values, "")
        status = "PASS" if predicted_label == row.label else "FAIL"
        if status == "FAIL":
            fail_count += 1
        else:
            pass_count += 1
        rows.append(
            {
                "status": status,
                "math_pool_exclusion_reason": "",
                "source_path": row.path,
                "expected_internal_label": row.label,
                "expected_label": public_label(row.label),
                "predicted_internal_label": predicted_label,
                "predicted_label": public_label(predicted_label),
                "predicted_top": predicted_top,
                "similarity": f"{sim:.6f}",
                "margin_distance_gap": f"{margin:.6f}",
                "duration_sec": f"{row.duration_sec:.6f}",
                **fingerprint_physics_dict(row.fingerprint, row.duration_sec),
                "top5_labels": " || ".join(f"{public_label(label)}:{score:.4f}" for label, score in top5[:5]),
            }
        )
    rows.sort(key=lambda r: (r["status"], r["expected_label"], r["source_path"]))
    out = reports_dir / "training_recall_contract_report.csv"
    write_csv(
        out,
        rows,
        [
            "status",
            "math_pool_exclusion_reason",
            "source_path",
            "expected_internal_label",
            "expected_label",
            "predicted_internal_label",
            "predicted_label",
            "predicted_top",
            "similarity",
            "margin_distance_gap",
            "duration_sec",
            *PHYSICS_REPORT_FIELDS,
            "top5_labels",
        ],
    )
    summary = {
        "total_readable_rows_seen": int(pass_count + fail_count + skipped_excluded),
        "active_math_pool_rows_tested": int(pass_count + fail_count),
        "active_math_pool_recall_passes": int(pass_count),
        "active_math_pool_recall_failures": int(fail_count),
        "excluded_from_math_not_expected_to_recall": int(skipped_excluded),
        "unreadable_rows_skipped": int(skipped_unreadable),
        "status": "PASS" if fail_count == 0 else "FAIL",
        "report": str(out),
    }
    write_json(reports_dir / "training_recall_contract_summary.json", summary)
    return out


def support_balance_bonus_for_label(brain: dict, label: str) -> float:
    """Return zero: count-based support bonuses are disabled.

    The two-voter redesign keeps category ranking count-neutral. Training support
    can be reported as confidence evidence, but it must not make an
    underrepresented folder artificially closer than a better physics match.
    Keeping this compatibility function makes old tests/imports stable while
    preventing hidden support-balance reranking.
    """
    return 0.0


def label_distance_score(cw: np.ndarray, xw: np.ndarray) -> float:
    """Score a label by nearest centroid with a soft outlier-centroid penalty.

    Lower is better.  The best centroid still dominates, but if a label only looks
    close because of one fringe centroid while the rest of that label prototypes
    are far away, the score is made worse instead of pretending the label is broad.
    """
    d = np.linalg.norm(cw - xw[None, :], axis=1).astype(np.float32)
    min_dist = float(np.min(d))
    if d.size <= 1:
        return min_dist
    order = np.argsort(d)
    others = d[order[1:]]
    mean_other = float(np.mean(others)) if others.size else min_dist
    penalty = 0.15 * max(0.0, mean_other - min_dist)
    return float(min_dist + penalty)


def label_model_distance_score(brain: dict, label: str, xw: np.ndarray) -> Tuple[float, float, str]:
    """Score one label with its adaptive folder model.

    Returns ranking_distance, similarity_distance, model_mode.

    v0.4.58: two changes from v0.4.57.

    1. Anchor rescue gap is now spread-scaled per label instead of using
       universal constants (0.04, 0.12, 0.20). A tight label like kick drums
       (spread ~0.3) gets a small gap (~0.024) so anchors help recall without
       overriding a clearly wrong centroid. A wide label like mixed FX (spread
       ~1.5) gets a proportionally larger gap (~0.12). The formula is:
         rescue_gap = clamp(0.08 * spread_mean, min=0.03, max=0.35)

    2. similarity_distance now equals ranking_distance. The old code returned
       min(raw_centroid_score, nearest_teacher_score) for similarity, which was
       lower (more confident) than the ranking distance when an anchor rescued a
       label. This made anchor-rescued predictions look more confident than they
       were. Anchors should improve recall, not manufacture high confidence.
    """
    weights = np.asarray(brain.get("feature_weights", FEATURE_WEIGHTS), dtype=np.float32)
    label_models = brain.get("label_models_by_label", {})
    model_info = label_models.get(label, {}) if isinstance(label_models, dict) else {}
    mode = str(model_info.get("model_mode", "centroid")) if isinstance(model_info, dict) else "centroid"

    # Spread-scaled rescue gap: scales with the label's own learned spread so
    # dense labels get a tight gap and diffuse labels get a looser one.
    # 0.08 is the base scale. Clamp keeps extreme labels from breaking the contest.
    spread = float(model_info.get("spread_mean", 1.0) or 1.0) if isinstance(model_info, dict) else 1.0
    rescue_gap = float(np.clip(0.08 * spread, 0.03, 0.35))

    centroid_score = float("inf")
    c = np.asarray(brain.get("centroids", {}).get(label, []), dtype=np.float32)
    if c.ndim == 1 and c.size:
        c = c.reshape(1, -1)
    if c.size:
        cw = c * weights[None, :]
        centroid_score = label_distance_score(cw, xw)

    exemplar_score = float("inf")
    e = np.asarray(brain.get("exemplars_by_label", {}).get(label, []), dtype=np.float32)
    if e.ndim == 1 and e.size:
        e = e.reshape(1, -1)
    if e.size:
        ew = e * weights[None, :]
        exemplar_score = float(np.min(np.linalg.norm(ew - xw[None, :], axis=1)))

    anchor_score = float("inf")
    a = np.asarray(brain.get("anchors_by_label", {}).get(label, []), dtype=np.float32)
    if a.ndim == 1 and a.size:
        a = a.reshape(1, -1)
    if a.size:
        aw = a * weights[None, :]
        ad = np.sort(np.linalg.norm(aw - xw[None, :], axis=1).astype(np.float32))
        # v0.4.65 product reset: do not let a large folder win from one lucky
        # nearest anchor.  Tiny labels still use their closest teacher.  Larger
        # labels need a small neighborhood of similar teachers, which is the
        # correct way to handle uneven folder counts without hard-coded category
        # names or filename hints.
        # v0.4.69 small-folder recall contract:
        # If a learned folder only has a small number of examples, do not average
        # the nearest teacher with other farther teachers. That averaging made an
        # exact training-file recall look weak for a 5-example folder. Small folders
        # are already controlled by stricter support/margin gates, so their distance
        # should be nearest-teacher distance. Larger folders still require a local
        # neighborhood so one lucky anchor cannot swallow unrelated sounds.
        train_count = (
            int(model_info.get("training_count", ad.size) or ad.size) if isinstance(model_info, dict) else int(ad.size)
        )
        if train_count <= SMALL_LABEL_COUNT or ad.size <= SMALL_LABEL_COUNT:
            anchor_score = float(ad[0])
        else:
            k = int(min(5, max(2, math.ceil(math.sqrt(float(ad.size)) / 2.0))))
            anchor_score = float(np.mean(ad[:k]))

    nearest_teacher_score = min(exemplar_score, anchor_score)

    # v0.4.70 small-folder direct anchor contract:
    # A small learned folder has too few teachers for neighborhood averaging or
    # rescue-gap inflation. If an exact teacher or very close teacher exists, the
    # brain should see that distance directly. The stricter small-label support
    # gates still decide whether an unseen file is safe to auto-place. Larger
    # folders keep the rescue-gap rule so one lucky anchor cannot dominate.
    train_count_for_model = int(model_info.get("training_count", 0) or 0) if isinstance(model_info, dict) else 0
    if mode == "exemplar_only" or train_count_for_model <= SMALL_LABEL_COUNT:
        return nearest_teacher_score, nearest_teacher_score, mode

    # Larger labels: anchor rescue within the spread-scaled gap.
    # similarity_distance = ranking_distance so confidence reflects the
    # actual contest result, not an artificially low anchor distance.
    ranking = min(centroid_score, nearest_teacher_score + rescue_gap)
    return ranking, ranking, mode


def scaler_for_label(brain: dict, label: str) -> Tuple[np.ndarray, np.ndarray]:
    structure_by_label = brain.get("structure_by_label", {})
    structure = structure_by_label.get(label, label_default_structure(label))
    scalers = brain.get("scalers_by_structure", {})
    scaler = scalers.get(structure) if isinstance(scalers, dict) else None
    if isinstance(scaler, dict) and "mean" in scaler and "std" in scaler:
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)
    else:
        mean = np.asarray(brain["scaler_mean"], dtype=np.float32)
        std = np.asarray(brain["scaler_std"], dtype=np.float32)
    return mean, std


def top_prefilter_allowed_tops(
    brain: dict, fingerprint: Sequence[float], top_n: int = 3
) -> Tuple[set, List[Tuple[str, float]]]:
    """Return learned top families allowed into the label contest.

    This is intentionally brain-only.  It uses learned super-centroids, not
    filenames, folder hints, or hand-written semantic vetoes.
    """
    super_by_top = brain.get("super_centroids_by_top_structure", {})
    if not isinstance(super_by_top, dict) or not super_by_top:
        return set(), []
    weights = np.asarray(brain["feature_weights"], dtype=np.float32)
    raw_x = np.asarray(fingerprint, dtype=np.float32)
    top_scores: List[Tuple[str, float]] = []
    scalers = brain.get("scalers_by_structure", {})
    for top_name, by_structure in super_by_top.items():
        if not isinstance(by_structure, dict):
            continue
        best_top_score = None
        for structure, centroid in by_structure.items():
            scaler = scalers.get(structure) if isinstance(scalers, dict) else None
            if isinstance(scaler, dict) and "mean" in scaler and "std" in scaler:
                mean = np.asarray(scaler["mean"], dtype=np.float32)
                std = np.asarray(scaler["std"], dtype=np.float32)
            else:
                mean = np.asarray(brain["scaler_mean"], dtype=np.float32)
                std = np.asarray(brain["scaler_std"], dtype=np.float32)
            xw = ((raw_x - mean) / std) * weights
            scw = np.asarray(centroid, dtype=np.float32) * weights
            score = float(np.linalg.norm(scw - xw))
            if best_top_score is None or score < best_top_score:
                best_top_score = score
        if best_top_score is not None:
            top_scores.append((str(top_name), float(best_top_score)))
    top_scores.sort(key=lambda t: t[1])
    allowed = {top for top, _ in top_scores[: max(1, int(top_n))]}
    return allowed, top_scores


def structure_distance_scores(brain: dict, fingerprint: Sequence[float]) -> List[Tuple[str, float]]:
    """Brain-only one-shot/loop structure scores.

    Lower is better.  Uses learned structure centroids and per-structure scalers.
    This does not inspect filenames, durations, or expected eval labels.
    """
    structure_centroids = brain.get("structure_centroids", {})
    if not isinstance(structure_centroids, dict) or not structure_centroids:
        return []
    weights = np.asarray(brain.get("structure_feature_weights", STRUCTURE_FEATURE_WEIGHTS), dtype=np.float32)
    raw_x = np.asarray(fingerprint, dtype=np.float32)
    scalers = brain.get("scalers_by_structure", {})
    scores: List[Tuple[str, float]] = []
    for structure, centroids in structure_centroids.items():
        scaler = scalers.get(structure) if isinstance(scalers, dict) else None
        if isinstance(scaler, dict) and "mean" in scaler and "std" in scaler:
            mean = np.asarray(scaler["mean"], dtype=np.float32)
            std = np.asarray(scaler["std"], dtype=np.float32)
        else:
            mean = np.asarray(brain["scaler_mean"], dtype=np.float32)
            std = np.asarray(brain["scaler_std"], dtype=np.float32)
        xw = ((raw_x - mean) / std) * weights
        cw = np.asarray(centroids, dtype=np.float32) * weights[None, :]
        if cw.ndim == 1:
            cw = cw.reshape(1, -1)
        scores.append((str(structure), float(np.min(np.linalg.norm(cw - xw[None, :], axis=1)))))
    scores.sort(key=lambda t: t[1])
    return scores


def structure_penalty_for_label(
    brain: dict, fingerprint: Sequence[float], label: str, structure_scores: Optional[List[Tuple[str, float]]] = None
) -> float:
    """Soft penalty when a label's structure disagrees with learned structure head.

    v0.4.5 fix: structure scores are measured in a separate weighted feature
    space from label distances, so the raw gap must be normalized by the learned
    typical distance for that structure before it is added to the label contest.
    Old v0.4.4 brains stored structure_score_weight=1.25; clamp that bad value
    so patched prediction remains safe even when reading an older brain JSON.
    """
    scores = structure_scores if structure_scores is not None else structure_distance_scores(brain, fingerprint)
    if not scores:
        return 0.0
    label_structure = brain.get("structure_by_label", {}).get(label, label_default_structure(label))
    best_score = float(scores[0][1])
    label_score = None
    for structure, score in scores:
        if structure == label_structure:
            label_score = float(score)
            break
    if label_score is None:
        return 0.0
    gap = max(0.0, label_score - best_score)
    typical = 1.0
    typical_by_structure = brain.get("typical_structure_dist", {})
    if isinstance(typical_by_structure, dict):
        typical = float(typical_by_structure.get(label_structure, 1.0) or 1.0)
    normalized_gap = gap / max(0.1, typical)
    raw_weight = float(brain.get("structure_score_weight", 0.20) or 0.20)
    weight = min(max(raw_weight, 0.0), 0.20)
    return weight * normalized_gap


def top_penalty_for_label(brain: dict, label: str, top_scores: List[Tuple[str, float]]) -> float:
    if not top_scores:
        return 0.0
    label_top = brain.get("top_by_label", {}).get(label, top_for_public_label(label))
    best_score = float(top_scores[0][1])
    label_score = None
    for top, score in top_scores:
        if top == label_top:
            label_score = float(score)
            break
    if label_score is None:
        return 0.0
    gap = max(0.0, label_score - best_score)
    return float(brain.get("top_score_weight", 0.35) or 0.35) * gap


def learned_structure_conflict_reason(brain: dict, fingerprint: Sequence[float], predicted_label: str) -> str:
    scores = structure_distance_scores(brain, fingerprint)
    if len(scores) < 2:
        return ""
    label_structure = brain.get("structure_by_label", {}).get(predicted_label, label_default_structure(predicted_label))
    best_structure, best_score = scores[0]
    label_score = None
    for structure, score in scores:
        if structure == label_structure:
            label_score = float(score)
            break
    if label_score is None or best_structure == label_structure:
        return ""
    raw_gap = max(0.0, label_score - float(best_score))
    typical = 1.0
    typical_by_structure = brain.get("typical_structure_dist", {})
    if isinstance(typical_by_structure, dict):
        typical = float(typical_by_structure.get(label_structure, 1.0) or 1.0)
    normalized_gap = raw_gap / max(0.1, typical)
    # v0.4.7: v0.4.5/v0.4.6 stored/used a raw-space threshold here.  The
    # structure penalty was normalized, but this review gate was not, so correct
    # identities with ordinary raw structure gaps could be sent to review.  Clamp
    # older low thresholds up to the normalized calibrated floor.
    raw_review_gap = float(brain.get("structure_review_gap", 1.00) or 1.00)
    review_gap = max(raw_review_gap, 1.00)
    if normalized_gap >= review_gap:
        return (
            f"Structure Conflict: learned_structure={best_structure} label_structure={label_structure} "
            f"normalized_gap={normalized_gap:.3f} >= {review_gap:.3f} raw_gap={raw_gap:.3f}"
        )
    return ""


def learned_top_conflict_reason(brain: dict, fingerprint: Sequence[float], predicted_top: str) -> str:
    _allowed, scores = top_prefilter_allowed_tops(
        brain, fingerprint, top_n=max(3, int(brain.get("top_prefilter_n", 3) or 3))
    )
    if len(scores) < 2:
        return ""
    best_top, best_score = scores[0]
    if best_top == predicted_top:
        return ""
    pred_score = None
    for top, score in scores:
        if top == predicted_top:
            pred_score = float(score)
            break
    if pred_score is None:
        return ""
    gap = pred_score - float(best_score)
    # v0.4.6 safety: v0.4.5 allowed top-family disagreement up to 0.80,
    # which let some third-place family labels auto-place.  Clamp older brains
    # to the stricter pure-confidence gate instead of needing a rebuild first.
    raw_review_gap = float(brain.get("top_review_gap", 0.25) or 0.25)
    review_gap = min(max(raw_review_gap, 0.0), 0.25)
    if gap >= review_gap:
        return f"Learned Top Family Conflict: learned_top={best_top} predicted_top={predicted_top} gap={gap:.3f} >= {review_gap:.3f}"
    return ""


def prediction_stage_diagnostics(brain: dict, fingerprint: Sequence[float], predicted_label: str) -> Dict[str, str]:
    """Brain-only diagnostics for staged prediction review.

    These fields are report data, not hand-written routing rules.  They expose
    how the learned top-family head, label head, and structure head agree or
    disagree so confusion pairs can be fixed with better training/features.
    """
    diag: Dict[str, str] = {
        "learned_top_1": "",
        "learned_top_1_score": "",
        "learned_top_2": "",
        "learned_top_2_score": "",
        "learned_top_gap": "",
        "learned_top_gap_ratio": "",
        "learned_structure": "",
        "learned_structure_score": "",
        "label_structure": "",
        "label_structure_score": "",
        "structure_gap_raw": "",
        "structure_gap_normalized": "",
    }

    _allowed, top_scores = top_prefilter_allowed_tops(
        brain, fingerprint, top_n=max(3, int(brain.get("top_prefilter_n", 3) or 3))
    )
    if top_scores:
        diag["learned_top_1"] = str(top_scores[0][0])
        diag["learned_top_1_score"] = f"{float(top_scores[0][1]):.6f}"
    if len(top_scores) > 1:
        gap = float(top_scores[1][1]) - float(top_scores[0][1])
        ratio = gap / max(0.1, float(top_scores[0][1]))
        diag["learned_top_2"] = str(top_scores[1][0])
        diag["learned_top_2_score"] = f"{float(top_scores[1][1]):.6f}"
        diag["learned_top_gap"] = f"{gap:.6f}"
        diag["learned_top_gap_ratio"] = f"{ratio:.6f}"

    structure_scores = structure_distance_scores(brain, fingerprint)
    label_structure = brain.get("structure_by_label", {}).get(predicted_label, label_default_structure(predicted_label))
    diag["label_structure"] = str(label_structure)
    if structure_scores:
        diag["learned_structure"] = str(structure_scores[0][0])
        diag["learned_structure_score"] = f"{float(structure_scores[0][1]):.6f}"
        label_score = None
        for structure, score in structure_scores:
            if structure == label_structure:
                label_score = float(score)
                break
        if label_score is not None:
            raw_gap = max(0.0, label_score - float(structure_scores[0][1]))
            typical = 1.0
            typical_by_structure = brain.get("typical_structure_dist", {})
            if isinstance(typical_by_structure, dict):
                typical = float(typical_by_structure.get(label_structure, 1.0) or 1.0)
            diag["label_structure_score"] = f"{label_score:.6f}"
            diag["structure_gap_raw"] = f"{raw_gap:.6f}"
            diag["structure_gap_normalized"] = f"{(raw_gap / max(0.1, typical)):.6f}"
    return diag
