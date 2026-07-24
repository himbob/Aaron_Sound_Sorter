# Auto-split from Aaron_Sound_Sorter.py.
# This is a component module, not a legacy wrapper.
from __future__ import annotations

import contextlib

from .core import *
from .two_voter import (
    TWO_VOTER_MANIFEST_FIELDS,
    run_two_voter_consensus,
    two_voter_error_manifest_row,
    two_voter_manifest_row,
)


def write_runtime_pointer(
    project_dir: Path,
    filename: str,
    target_path: Path,
) -> Path:
    """Write a portable project runtime pointer.

    Args:
        project_dir: Repository root.
        filename: Lowercase pointer filename.
        target_path: File or folder referenced by the pointer.

    Returns:
        Written pointer path.

    Side Effects:
        Creates ``config/runtime`` and writes one text file.
    """
    pointer_path = project_dir / "config" / "runtime" / filename
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_project = project_dir.expanduser().resolve()
    resolved_target = target_path.expanduser().resolve()
    try:
        pointer_value = str(resolved_target.relative_to(resolved_project))
    except ValueError:
        pointer_value = str(resolved_target)
    pointer_path.write_text(pointer_value + "\n", encoding="utf-8")
    return pointer_path


def include_tiny_files() -> bool:
    """Tiny/wavetable inclusion is off by default.

    Aaron explicitly asked that tiny files stay out of normal Phase 3 runs unless
    INCLUDE_TINY=1 is set.  This keeps AKWF/wavetable fragments and near-zero
    duration files from polluting blind preview evidence.
    """
    return str(os.environ.get("INCLUDE_TINY", "0")).strip() == "1"


def is_apple_or_hidden_junk_path(path: Path) -> bool:
    """Reject macOS metadata and hidden sidecar files before audio decoding.

    The v0.4.2 overnight run wasted thousands of rows on files like
    ._Kick.wav.  Those are AppleDouble resource forks, not real audio.
    """
    for part in path.parts:
        low = part.lower()
        if low == "__macosx":
            return True
        if low in APPLE_JUNK_NAMES:
            return True
        if part.startswith("._"):
            return True
        if part.startswith(".") and low not in {".", ".."}:
            return True
    return False


def is_akwf_or_wavetable_path(path: Path) -> bool:
    """Skip AKWF/wavetable material by default in blind preview.

    Those files are often intentionally tiny single-cycle waveforms, not normal
    samples for Phase 3 sorting evidence.
    """
    text = "/".join(path.parts).lower()
    return "akwf" in text or "adventure kid" in text or "wavetable" in text or "wave table" in text


def should_skip_real_preview_path(path: Path) -> bool:
    """Return True for path-level junk that is never useful preview input.

    Training-root exclusion is intentionally not handled here. The caller must
    decide whether it is doing a blind real-library preview, where training
    overlap must be excluded, or explicit sort mode, where user-provided files
    must not be silently dropped just because they are symlinks into the
    training tree.
    """
    if is_apple_or_hidden_junk_path(path):
        return True
    return bool(not include_tiny_files() and is_akwf_or_wavetable_path(path))


def should_skip_tiny_after_read(duration_sec: float, source_path: Path) -> bool:
    if include_tiny_files():
        return False
    try:
        return float(duration_sec) <= TINY_DURATION_SECONDS_DEFAULT
    except Exception:
        return False


def normalized_resolved_path(path: Path) -> str:
    try:
        return str(Path(path).expanduser().resolve()).rstrip("/")
    except Exception:
        return str(Path(path).expanduser()).rstrip("/")


def path_is_under(path: Path, root: Path) -> bool:
    p = normalized_resolved_path(path)
    r = normalized_resolved_path(root)
    return bool(r) and (p == r or p.startswith(r + "/"))


def default_training_roots_to_exclude() -> List[Path]:
    # Blind preview must never test against the curated training pool.
    return [Path(DEFAULT_ROOT)]


def collect_resolved_training_audio_files(training_root: Path) -> Set[str]:
    """Return resolved audio targets used by the training tree.

    The locked corpus is symlink-first. Excluding only the training folder is
    not enough because a blind preview over an external sample library can
    still hit the original target file. Exact resolved target exclusion keeps
    the real preview honest without hiding the rest of that source pack.
    """
    root = Path(training_root).expanduser()
    if not root.exists() and not root.is_symlink():
        return set()
    out: Set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames if not name.startswith(".") and name.lower() not in {"_manifests", "__macosx"}
        ]
        here = Path(dirpath)
        for filename in filenames:
            path = here / filename
            if path.suffix.lower() not in AUDIO_EXTS or is_apple_or_hidden_junk_path(path):
                continue
            out.add(normalized_resolved_path(path))
    return out


def is_real_preview_excluded(
    path: Path,
    extra_exclude_roots: Sequence[Path] = (),
    include_default_training_roots: bool = True,
) -> bool:
    roots = (list(default_training_roots_to_exclude()) if include_default_training_roots else []) + list(
        extra_exclude_roots or []
    )
    for root in roots:
        try:
            if path_is_under(path, root):
                return True
        except Exception:
            pass
    text_parts = [str(part).lower().replace("_", " ").replace("-", " ") for part in path.parts]
    for part in text_parts:
        if part in REAL_PREVIEW_EXCLUDE_PARTS:
            return True
        if any(fragment in part for fragment in REAL_PREVIEW_EXCLUDE_FILE_FRAGMENTS):
            return True
    return False


def iter_real_preview_audio_files(
    root: Path,
    max_total: int,
    per_folder: int,
    random_seed: int,
    extra_exclude_roots: Sequence[Path] = (),
    exclude_resolved_files: Optional[Set[str]] = None,
    include_default_training_roots: bool = True,
) -> List[Path]:
    """Pick real-library files for blind preview, excluding training/report folders.

    v0.4.5 hardens the selector so the blind test does not include AppleDouble
    resource forks, hidden files, generated reports, AKWF/wavetable folders, or
    training/reference ZIP material.  This fixes the v0.4.2 read-error explosion.
    """
    root = root.expanduser().resolve()
    if not root.exists():
        return []
    selected: List[Path] = []
    max_total = max(0, int(max_total or 0))
    per_folder = max(1, int(per_folder or 1))
    excluded_files = exclude_resolved_files or set()
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        # Prune known training/report/output/hidden directories before descent.
        keep_dirnames = []
        for name in dirnames:
            candidate = here / name
            if not should_skip_real_preview_path(candidate) and not is_real_preview_excluded(
                candidate, extra_exclude_roots, include_default_training_roots=include_default_training_roots
            ):
                keep_dirnames.append(name)
        dirnames[:] = keep_dirnames
        if should_skip_real_preview_path(here) or is_real_preview_excluded(
            here, extra_exclude_roots, include_default_training_roots=include_default_training_roots
        ):
            dirnames[:] = []
            continue
        audio_files = []
        for filename in filenames:
            p = here / filename
            if p.suffix.lower() in AUDIO_EXTS and normalized_resolved_path(p) in excluded_files:
                continue
            if (
                p.suffix.lower() in AUDIO_EXTS
                and not should_skip_real_preview_path(p)
                and not is_real_preview_excluded(
                    p, extra_exclude_roots, include_default_training_roots=include_default_training_roots
                )
            ):
                audio_files.append(p)
        if not audio_files:
            continue
        chosen = sorted(audio_files, key=lambda p: stable_random_key(str(p), "real_preview", seed=random_seed))[
            :per_folder
        ]
        selected.extend(chosen)
        if max_total and len(selected) >= max_total:
            return selected[:max_total]
    return selected[:max_total] if max_total else selected


def choose_broad_family_candidate_top(
    predicted_top: str,
    learned_top: str,
    learned_gap: float,
    allowed_top: Set[str] = DEFAULT_ALLOWED_TOP,
    min_gap: float = 0.0,
    conflict_override_gap: float = DEFAULT_BROAD_FAMILY_CONFLICT_OVERRIDE_GAP,
    review_reason: str = "",
) -> Tuple[str, str]:
    """Choose the review-only broad-family queue without contradicting weak evidence.

    The raw exact-label head and learned top-family head are both brain evidence.
    When they disagree, the broad-family queue should not blindly trust the top
    head unless its gap is strong.  This is generic across Drums/Instruments/FX/
    Textures; it is not a filename route and it never auto-places a file.
    """
    pred = str(predicted_top or "").strip()
    learned = str(learned_top or "").strip()
    if learned not in allowed_top:
        return "", "no_learned_top_candidate"
    if float(learned_gap) < float(min_gap):
        return "", f"learned_top_gap_below_min {float(learned_gap):.3f} < {float(min_gap):.3f}"
    if not pred or pred == "_TO_REVIEW" or pred not in allowed_top:
        return learned, "learned_top_no_valid_raw_top"
    if learned == pred:
        return learned, "learned_top_agrees_with_raw_top"
    reason_text = str(review_reason or "").lower()
    # If the final review reason explicitly says the raw Drums path was rejected
    # by measured non-drum physics, do not keep putting the file in a Possible
    # Drum review queue.  Use the learned broad-family head as the review queue.
    # This does not auto-place the file; it only prevents misleading review folders.
    if (
        pred == "Drums"
        and learned in allowed_top
        and learned != pred
        and (
            "unsupported_drum" in reason_text
            or "drum_loop_rejected_by_non_drum_physics" in reason_text
            or "unsupported drum loop" in reason_text
        )
    ):
        return learned, (
            f"learned_top_used_after_raw_drum_rejected_by_physics raw_top={pred} "
            f"learned_top={learned} gap={float(learned_gap):.3f}"
        )
    if float(learned_gap) >= max(float(conflict_override_gap), float(min_gap)):
        return learned, (
            f"learned_top_strong_override raw_top={pred} learned_top={learned} "
            f"gap={float(learned_gap):.3f} >= {float(conflict_override_gap):.3f}"
        )
    return pred, (
        f"fallback_to_raw_top_on_weak_broad_conflict raw_top={pred} learned_top={learned} "
        f"gap={float(learned_gap):.3f} < {float(conflict_override_gap):.3f}"
    )


def copy_real_preview_file(
    source_path: str,
    sorted_root: Path,
    final_top: str,
    final_label: str,
    index: int,
    predicted_label: str,
    review_reason: str = "",
) -> str:
    src = Path(source_path)
    if not src.exists() and not src.is_symlink():
        return ""
    if final_top == "_TO_REVIEW":
        final_public = public_label(final_label)
        reason_text = str(review_reason or "").lower()
        if (
            final_public.startswith("_TO_REVIEW/")
            or "unsupported_drum" in reason_text
            or "drum_loop_rejected_by_non_drum_physics" in reason_text
        ):
            # Review folders should describe the conflict, not keep advertising a
            # raw drum guess that the physical guard already rejected.  The raw
            # brain guess remains in the manifest and raw prediction preview.
            parts = [safe_folder_name(p) for p in final_public.split("/") if p.strip()]
            dest_dir = sorted_root.joinpath(*parts) if parts else sorted_root / "_TO_REVIEW" / "Conflicting Evidence"
        else:
            guess = public_label(predicted_label)
            dest_dir = sorted_root / "_TO_REVIEW" / f"guess_{safe_folder_name(guess)}__{safe_folder_name(final_public)}"
    else:
        parts = folder_parts_for_public_label(final_label)
        dest_dir = sorted_root.joinpath(*parts) if parts else sorted_root / safe_folder_name(final_top)
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() or ".wav"
    digest = hashlib.sha1(str(src).encode("utf-8", errors="replace")).hexdigest()[:10]
    dest = dest_dir / f"real_{index:05d}__{digest}__{safe_folder_name(src.stem)[:48]}{suffix}"
    i = 2
    base = dest
    while dest.exists():
        dest = base.with_name(base.stem + f"__{i}" + base.suffix)
        i += 1
    try:
        return materialize_preview_audio(src, dest)
    except Exception:
        return ""


def copy_raw_prediction_preview_file(source_path: str, raw_root: Path, predicted_label: str, index: int) -> str:
    """Copy every real-preview file under the brain's raw top prediction.

    This is Phase 3 diagnostic evidence only. It does not rescue or alter the
    gated placement. It lets Aaron inspect many samples per predicted label even
    when the strict gate correctly keeps the official preview in _TO_REVIEW.
    """
    src = Path(source_path)
    if not src.exists() and not src.is_symlink():
        return ""
    parts = folder_parts_for_public_label(public_label(predicted_label))
    dest_dir = raw_root.joinpath(*parts) if parts else raw_root / safe_folder_name(public_label(predicted_label))
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() or ".wav"
    digest = hashlib.sha1(("raw:" + str(src)).encode("utf-8", errors="replace")).hexdigest()[:10]
    dest = dest_dir / f"raw_{index:05d}__{digest}__{safe_folder_name(src.stem)[:48]}{suffix}"
    i = 2
    base = dest
    while dest.exists():
        dest = base.with_name(base.stem + f"__{i}" + base.suffix)
        i += 1
    try:
        return materialize_preview_audio(src, dest)
    except Exception:
        return ""


def copy_forced_guess_preview_file(source_path: str, forced_root: Path, predicted_label: str, index: int) -> str:
    """Symlink a real-preview file into the brain's forced best-guess tree.

    This is the no-baby-steps output: every readable file gets the brain's best
    exact label, regardless of confidence, gate review, source hint conflict, or
    margin. It is intentionally separated from official output so overnight
    tests can expose the brain's real behavior without pretending it is safe.
    """
    src = Path(source_path)
    if not src.exists() and not src.is_symlink():
        return ""
    parts = folder_parts_for_public_label(public_label(predicted_label))
    dest_dir = forced_root.joinpath(*parts) if parts else forced_root / safe_folder_name(public_label(predicted_label))
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() or ".wav"
    digest = hashlib.sha1(("forced:" + str(src) + "|" + predicted_label).encode("utf-8", errors="replace")).hexdigest()[
        :10
    ]
    dest = dest_dir / f"forced_{index:06d}__{digest}__{safe_folder_name(src.stem)[:56]}{suffix}"
    i = 2
    base = dest
    while dest.exists():
        dest = base.with_name(base.stem + f"__{i}" + base.suffix)
        i += 1
    try:
        return materialize_preview_audio(src, dest)
    except Exception:
        return ""


def copy_broad_family_review_file(
    source_path: str, broad_root: Path, broad_top: str, predicted_label: str, review_reason: str, index: int
) -> str:
    """Copy reviewed files into a broad-family candidate tree.

    This is deliberately not the official sorted output.  It is a review aid
    for Phase 3: when the strict exact-label gate says "review", the brain's
    separate top-family head can still group candidates for listening without
    pretending the exact folder is solved.
    """
    src = Path(source_path)
    if not src.exists() and not src.is_symlink():
        return ""
    broad = safe_folder_name(broad_top or "Unknown")
    guess = safe_folder_name(public_label(predicted_label))
    reason = safe_folder_name(review_reason.split(":", 1)[0] or "Review")
    dest_dir = broad_root / broad / f"guess_{guess}" / reason
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() or ".wav"
    digest = hashlib.sha1(("broad:" + str(src)).encode("utf-8", errors="replace")).hexdigest()[:10]
    dest = dest_dir / f"broad_{index:05d}__{digest}__{safe_folder_name(src.stem)[:48]}{suffix}"
    i = 2
    base = dest
    while dest.exists():
        dest = base.with_name(base.stem + f"__{i}" + base.suffix)
        i += 1
    try:
        return materialize_preview_audio(src, dest)
    except Exception:
        return ""


def copy_physical_role_review_file(
    source_path: str, role_root: Path, candidate_top: str, candidate_label: str, action: str, index: int
) -> str:
    """Copy/symlink reviewed files into physics-role candidate folders."""
    src = Path(source_path)
    if not src.exists() and not src.is_symlink():
        return ""
    top = safe_folder_name(candidate_top or "Unknown")
    parts = folder_parts_for_public_label(candidate_label)
    if parts:
        # Drop duplicate top if folder_parts already starts with the top.
        dest_dir = role_root.joinpath(*parts)
    else:
        dest_dir = role_root / top / safe_folder_name(candidate_label or "Unknown")
    if action:
        dest_dir = dest_dir / safe_folder_name(action)
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() or ".wav"
    digest = hashlib.sha1(("role:" + str(src) + "|" + candidate_label).encode("utf-8", errors="replace")).hexdigest()[
        :10
    ]
    dest = dest_dir / f"role_{index:05d}__{digest}__{safe_folder_name(src.stem)[:48]}{suffix}"
    i = 2
    base = dest
    while dest.exists():
        dest = base.with_name(base.stem + f"__{i}" + base.suffix)
        i += 1
    try:
        return materialize_preview_audio(src, dest)
    except Exception:
        return ""


def copy_risky_sorted_preview_file(
    source_path: str, risky_root: Path, risky_top: str, predicted_label: str, index: int
) -> str:
    """Copy/symlink a reviewed file into a risky sorted workbench tree.

    This is intentionally separate from the official gated output. It lets us
    inspect many more useful "probably sorted" examples without claiming the
    strict production gate has accepted them.
    """
    src = Path(source_path)
    if not src.exists() and not src.is_symlink():
        return ""
    predicted_top = top_for_public_label(predicted_label)
    if predicted_top == risky_top:
        parts = folder_parts_for_public_label(public_label(predicted_label))
        dest_dir = risky_root.joinpath(*parts) if parts else risky_root / safe_folder_name(risky_top)
    else:
        dest_dir = (
            risky_root
            / safe_folder_name(risky_top or "Unknown")
            / "_BROAD_FAMILY_ONLY"
            / f"guess_{safe_folder_name(public_label(predicted_label))}"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() or ".wav"
    digest = hashlib.sha1(
        ("risky:" + str(src) + "|" + risky_top + "|" + predicted_label).encode("utf-8", errors="replace")
    ).hexdigest()[:10]
    dest = dest_dir / f"risky_{index:05d}__{digest}__{safe_folder_name(src.stem)[:48]}{suffix}"
    i = 2
    base = dest
    while dest.exists():
        dest = base.with_name(base.stem + f"__{i}" + base.suffix)
        i += 1
    try:
        return materialize_preview_audio(src, dest)
    except Exception:
        return ""


def write_tree_file(root: Path, out_path: Path) -> None:
    lines: List[str] = []
    if root.exists():
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            depth = len(rel.parts) - 1
            indent = "  " * depth
            suffix = "/" if p.is_dir() else ""
            lines.append(f"{indent}{p.name}{suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_real_sort_label_no_show_report(brain: dict, rows: List[Dict[str, str]], reports_dir: Path) -> Path:
    """Explain why trained labels did or did not show up in the official gated preview."""
    labels = [str(x) for x in brain.get("labels", [])]
    rel = (
        brain.get("label_reliability_by_label", {})
        if isinstance(brain.get("label_reliability_by_label", {}), dict)
        else {}
    )
    out_rows: List[Dict[str, str]] = []
    for label in labels:
        label_rows = [r for r in rows if r.get("raw_predicted_internal_label") == label]
        placed_rows = [r for r in label_rows if r.get("final_top") != "_TO_REVIEW"]
        reviewed_rows = [r for r in label_rows if r.get("final_top") == "_TO_REVIEW"]
        sims: List[float] = []
        margins: List[float] = []
        for r in label_rows:
            with contextlib.suppress(Exception):
                sims.append(float(r.get("similarity", "") or 0.0))
            with contextlib.suppress(Exception):
                margins.append(float(r.get("margin_distance_gap", "") or 0.0))
        reason_counts = Counter(str(r.get("review_reason", "")) for r in reviewed_rows)
        physics_counter = Counter()
        for r in label_rows:
            for tag in str(r.get("physics_tags", "")).split(";"):
                tag = tag.strip()
                if tag:
                    physics_counter[tag] += 1
        if not label_rows:
            dominant_reason = "no raw predictions in real preview sample"
            no_show_class = "not_predicted"
        elif not placed_rows:
            dominant_reason = reason_counts.most_common(1)[0][0] if reason_counts else "all raw predictions reviewed"
            no_show_class = "predicted_but_gate_blocked"
        else:
            dominant_reason = "placed_some"
            no_show_class = "showed_up"
        info = rel.get(label, {}) if isinstance(rel, dict) else {}
        if not isinstance(info, dict):
            info = {}

        def fmt_stat(vals: List[float], which: str) -> str:
            if not vals:
                return ""
            vals = sorted(vals)
            if which == "min":
                return f"{vals[0]:.6f}"
            if which == "max":
                return f"{vals[-1]:.6f}"
            return f"{vals[len(vals) // 2]:.6f}"

        out_rows.append(
            {
                "label": label,
                "public_label": public_label(label),
                "support_tier": str(info.get("support_tier", "")),
                "training_count": str(info.get("training_count", "")),
                "raw_predicted_count": str(len(label_rows)),
                "official_placed_count": str(len(placed_rows)),
                "reviewed_count": str(len(reviewed_rows)),
                "no_show_class": no_show_class,
                "dominant_block_reason": dominant_reason,
                "similarity_min": fmt_stat(sims, "min"),
                "similarity_median": fmt_stat(sims, "median"),
                "similarity_max": fmt_stat(sims, "max"),
                "margin_min": fmt_stat(margins, "min"),
                "margin_median": fmt_stat(margins, "median"),
                "margin_max": fmt_stat(margins, "max"),
                "top_review_reasons": " || ".join(f"{k}: {v}" for k, v in reason_counts.most_common(5)),
                "top_physics_tags_report_only": " || ".join(f"{k}: {v}" for k, v in physics_counter.most_common(10)),
            }
        )
    fields = [
        "label",
        "public_label",
        "support_tier",
        "training_count",
        "raw_predicted_count",
        "official_placed_count",
        "reviewed_count",
        "no_show_class",
        "dominant_block_reason",
        "similarity_min",
        "similarity_median",
        "similarity_max",
        "margin_min",
        "margin_median",
        "margin_max",
        "top_review_reasons",
        "top_physics_tags_report_only",
    ]
    out_path = reports_dir / "real_sort_label_no_show_reasons.csv"
    write_csv(out_path, out_rows, fields)
    return out_path


def run_real_sort_preview(brain: dict, args: argparse.Namespace, run_dir: Path, reports_dir: Path) -> dict:
    """Blind sort real library files after training.

    This is the test Aaron asked for: do not test on the training folders under
    the sample root.  It uses the pure brain prediction, learned structure/top
    heads, and measured audio physics only. Source filenames and folder names are
    never used as placement evidence for blind sorting.
    """
    if getattr(args, "no_real_sort_preview", False):
        summary = {"status": "disabled"}
        write_json(reports_dir / "real_sort_preview_summary.json", summary)
        return summary
    root = Path(getattr(args, "real_preview_root", "") or "").expanduser()
    training_root = Path(getattr(args, "root", "") or DEFAULT_ROOT)
    excluded_training_files = collect_resolved_training_audio_files(training_root)
    files = iter_real_preview_audio_files(
        root,
        max_total=int(getattr(args, "real_preview_max_files", 0) or 0),
        per_folder=int(getattr(args, "real_preview_per_folder", 5) or 5),
        random_seed=int(getattr(args, "random_seed", 20260503) or 20260503),
        extra_exclude_roots=[training_root],
        exclude_resolved_files=excluded_training_files,
        include_default_training_roots=bool(getattr(args, "real_preview_exclude_default_training_roots", True)),
    )
    sorted_root = run_dir / "real_sorted_preview"
    forced_guess_root = run_dir / "real_forced_guess_preview"
    risky_sorted_root = run_dir / "DIAGNOSTIC_RISKY_GUESSES_NOT_A_SORT"
    raw_prediction_root = run_dir / "real_raw_brain_prediction_preview"
    broad_family_root = run_dir / "real_broad_family_review_preview"
    physical_role_root = run_dir / "real_physical_role_review_preview"
    rows: List[Dict[str, str]] = []
    final_top_counts = Counter()
    broad_family_counts = Counter()
    official_copy_counts = Counter()
    forced_guess_copy_counts = Counter()
    raw_copy_counts = Counter()
    risky_copy_counts = Counter()
    broad_copy_counts = Counter()
    physical_role_counts = Counter()
    official_copy_skipped_after_limit = 0
    forced_guess_copy_skipped_after_limit = 0
    raw_copy_skipped_after_limit = 0
    risky_copy_skipped_after_limit = 0
    broad_copy_skipped_after_limit = 0
    physical_role_copy_skipped_after_limit = 0
    official_copy_limit = int(
        getattr(args, "real_preview_official_copy_per_label", DEFAULT_REAL_PREVIEW_OFFICIAL_COPY_PER_LABEL)
        if getattr(args, "real_preview_official_copy_per_label", None) is not None
        else DEFAULT_REAL_PREVIEW_OFFICIAL_COPY_PER_LABEL
    )
    raw_copy_limit = int(
        getattr(args, "real_preview_raw_copy_per_label", DEFAULT_REAL_PREVIEW_RAW_COPY_PER_LABEL)
        if getattr(args, "real_preview_raw_copy_per_label", None) is not None
        else DEFAULT_REAL_PREVIEW_RAW_COPY_PER_LABEL
    )
    force_guess_enabled = bool(getattr(args, "real_preview_force_guess", False))
    forced_guess_copy_limit = int(
        getattr(args, "real_preview_force_guess_copy_per_label", DEFAULT_REAL_PREVIEW_FORCE_GUESS_COPY_PER_LABEL)
        if getattr(args, "real_preview_force_guess_copy_per_label", None) is not None
        else DEFAULT_REAL_PREVIEW_FORCE_GUESS_COPY_PER_LABEL
    )
    risky_copy_limit = int(
        getattr(args, "real_preview_risky_copy_per_label", DEFAULT_REAL_PREVIEW_RISKY_COPY_PER_LABEL)
        if getattr(args, "real_preview_risky_copy_per_label", None) is not None
        else DEFAULT_REAL_PREVIEW_RISKY_COPY_PER_LABEL
    )
    risky_min_similarity = float(
        getattr(args, "real_preview_risky_min_similarity", DEFAULT_REAL_PREVIEW_RISKY_MIN_SIMILARITY)
        if getattr(args, "real_preview_risky_min_similarity", None) is not None
        else DEFAULT_REAL_PREVIEW_RISKY_MIN_SIMILARITY
    )
    broad_copy_limit = int(
        getattr(args, "real_preview_broad_copy_per_label", DEFAULT_REAL_PREVIEW_BROAD_COPY_PER_LABEL)
        if getattr(args, "real_preview_broad_copy_per_label", None) is not None
        else DEFAULT_REAL_PREVIEW_BROAD_COPY_PER_LABEL
    )
    physical_role_copy_limit = int(
        getattr(args, "real_preview_physical_role_copy_per_label", DEFAULT_REAL_PREVIEW_BROAD_COPY_PER_LABEL)
        if getattr(args, "real_preview_physical_role_copy_per_label", None) is not None
        else DEFAULT_REAL_PREVIEW_BROAD_COPY_PER_LABEL
    )
    broad_min_gap = float(
        getattr(args, "real_preview_broad_family_min_gap", DEFAULT_REAL_PREVIEW_BROAD_FAMILY_MIN_GAP) or 0.0
    )
    broad_conflict_override_gap = float(
        getattr(
            args,
            "real_preview_broad_family_conflict_override_gap",
            DEFAULT_BROAD_FAMILY_CONFLICT_OVERRIDE_GAP,
        )
        or 0.0
    )
    skipped_after_read = Counter()
    kept_index = 0
    for original_index, path in enumerate(files, 1):
        fp, duration, read_status = make_fingerprint_safe(path)
        if read_status == "ok" and should_skip_tiny_after_read(duration, path):
            skipped_after_read["tiny_or_wavetable_duration_skipped"] += 1
            continue
        kept_index += 1
        idx = kept_index
        if read_status == "ok":
            pred_label, pred_top, sim, margin, top5 = predict(brain, fp.tolist(), "", duration)
            fact_meta = brain.get("_last_fact_meta", {}) if isinstance(brain.get("_last_fact_meta", {}), dict) else {}
        else:
            pred_label, pred_top, sim, margin, top5 = "review_unreadable", "_TO_REVIEW", 0.0, 0.0, []
            fact_meta = {}
        final_top, final_label, confidence_status, review_reason = gate_prediction(
            pred_label,
            pred_top,
            sim,
            margin,
            read_status,
            duration,
            args.min_similarity,
            args.min_margin,
            args.weak_similarity_floor,
            args.strong_margin,
            bool(getattr(args, "allow_low_similarity_clear_margin", False)),
        )
        final_top, final_label, confidence_status, review_reason = apply_learned_conflict_gates(
            brain,
            fp.tolist() if hasattr(fp, "tolist") else fp,
            pred_label,
            pred_top,
            final_top,
            final_label,
            confidence_status,
            review_reason,
            sim,
            margin,
            duration,
            top5,
        )
        fp_seq = fp.tolist() if hasattr(fp, "tolist") else fp
        try:
            two_voter_manifest = two_voter_manifest_row(
                run_two_voter_consensus(brain, fp_seq, duration, read_status=read_status, top_n=20)
            )
        except Exception as exc:
            two_voter_manifest = two_voter_error_manifest_row(exc)
        stage_diag = prediction_stage_diagnostics(brain, fp_seq, pred_label)
        rel = brain.get("label_reliability_by_label", {})
        rel_info = rel.get(pred_label, {}) if isinstance(rel, dict) else {}
        if not isinstance(rel_info, dict):
            rel_info = {}
        nearest_predicted_examples = format_nearest_training_examples(
            nearest_training_examples_for_label(brain, pred_label, fp_seq, limit=3)
        )
        current_physics = {
            "physics_tags": fingerprint_physics_tags(fp_seq, duration),
            "physics_summary": fingerprint_physics_summary(fp_seq, duration),
            **fingerprint_physics_dict(fp_seq, duration),
        }
        official_copy_key = (
            f"{final_top}::{final_label}" if final_top != "_TO_REVIEW" else f"_TO_REVIEW::{pred_label}::{final_label}"
        )
        raw_copy_key = pred_label
        broad_top = str(stage_diag.get("learned_top_1", "") or "")
        try:
            broad_gap = float(stage_diag.get("learned_top_gap", "") or 0.0)
        except Exception:
            broad_gap = 0.0
        broad_family_candidate_top = ""
        broad_family_candidate_reason = ""
        broad_family_copy_path = ""
        forced_guess_copy_path = ""
        risky_sorted_copy_path = ""
        risky_sorted_top = ""
        risky_sorted_label = ""
        risky_sorted_reason = ""
        role_rec = physical_role_recommendation(brain, fp_seq, duration, pred_label, pred_top, top5)
        physical_role_candidate_top = role_rec.get("physical_role_candidate_top", "")
        physical_role_candidate_label = role_rec.get("physical_role_candidate_label", "")
        physical_role_candidate_reason = role_rec.get("physical_role_candidate_reason", "")
        physical_role_action = role_rec.get("physical_role_action", "")
        physical_role_copy_path = ""
        if (
            final_top == "_TO_REVIEW"
            and read_status == "ok"
            and broad_top in DEFAULT_ALLOWED_TOP
            and broad_gap >= broad_min_gap
        ):
            broad_family_candidate_top, broad_family_candidate_reason = choose_broad_family_candidate_top(
                pred_top,
                broad_top,
                broad_gap,
                allowed_top=DEFAULT_ALLOWED_TOP,
                min_gap=broad_min_gap,
                conflict_override_gap=broad_conflict_override_gap,
                review_reason=review_reason,
            )
            if broad_family_candidate_top:
                broad_family_counts[broad_family_candidate_top] += 1
        if official_copy_limit < 0 or official_copy_counts[official_copy_key] < official_copy_limit:
            official_copy_counts[official_copy_key] += 1
            preview_copy = copy_real_preview_file(
                str(path), sorted_root, final_top, final_label, idx, pred_label, review_reason
            )
        else:
            official_copy_skipped_after_limit += 1
            preview_copy = "COPY_SKIPPED_PER_LABEL_LIMIT"
        if raw_copy_limit < 0 or raw_copy_counts[raw_copy_key] < raw_copy_limit:
            raw_copy_counts[raw_copy_key] += 1
            raw_prediction_copy = copy_raw_prediction_preview_file(str(path), raw_prediction_root, pred_label, idx)
        else:
            raw_copy_skipped_after_limit += 1
            raw_prediction_copy = "COPY_SKIPPED_PER_LABEL_LIMIT"
        if force_guess_enabled and read_status == "ok":
            forced_copy_key = pred_label
            if forced_guess_copy_limit < 0 or forced_guess_copy_counts[forced_copy_key] < forced_guess_copy_limit:
                forced_guess_copy_counts[forced_copy_key] += 1
                forced_guess_copy_path = copy_forced_guess_preview_file(str(path), forced_guess_root, pred_label, idx)
            else:
                forced_guess_copy_skipped_after_limit += 1
                forced_guess_copy_path = "COPY_SKIPPED_PER_LABEL_LIMIT"
        if final_top != "_TO_REVIEW":
            risky_sorted_top = final_top
            risky_sorted_label = final_label
            risky_sorted_reason = "official_auto_place"
        elif broad_family_candidate_top and sim >= risky_min_similarity:
            risky_sorted_top = broad_family_candidate_top
            risky_sorted_label = pred_label
            risky_sorted_reason = "broad_family_risky"
        elif broad_family_candidate_top:
            risky_sorted_reason = f"blocked_similarity_{sim:.3f}_below_{risky_min_similarity:.3f}"
        if risky_sorted_top and risky_sorted_label:
            risky_copy_key = f"{risky_sorted_top}::{risky_sorted_label}"
            if risky_copy_limit < 0 or risky_copy_counts[risky_copy_key] < risky_copy_limit:
                risky_copy_counts[risky_copy_key] += 1
                risky_sorted_copy_path = copy_risky_sorted_preview_file(
                    str(path), risky_sorted_root, risky_sorted_top, risky_sorted_label, idx
                )
            else:
                risky_copy_skipped_after_limit += 1
                risky_sorted_copy_path = "COPY_SKIPPED_PER_LABEL_LIMIT"
        if broad_family_candidate_top:
            broad_copy_key = f"{broad_family_candidate_top}::{pred_label}"
            if broad_copy_limit < 0 or broad_copy_counts[broad_copy_key] < broad_copy_limit:
                broad_copy_counts[broad_copy_key] += 1
                broad_family_copy_path = copy_broad_family_review_file(
                    str(path), broad_family_root, broad_family_candidate_top, pred_label, review_reason, idx
                )
            else:
                broad_copy_skipped_after_limit += 1
                broad_family_copy_path = "COPY_SKIPPED_PER_LABEL_LIMIT"
        if physical_role_candidate_top and physical_role_candidate_label:
            role_copy_key = f"{physical_role_candidate_top}::{physical_role_candidate_label}"
            if physical_role_copy_limit < 0 or physical_role_counts[role_copy_key] < physical_role_copy_limit:
                physical_role_counts[role_copy_key] += 1
                physical_role_copy_path = copy_physical_role_review_file(
                    str(path),
                    physical_role_root,
                    physical_role_candidate_top,
                    physical_role_candidate_label,
                    physical_role_action,
                    idx,
                )
            else:
                physical_role_copy_skipped_after_limit += 1
                physical_role_copy_path = "COPY_SKIPPED_PER_LABEL_LIMIT"
        sanity_warnings = fingerprint_sanity_warnings(pred_label, fp.tolist() if hasattr(fp, "tolist") else fp)
        final_top_counts[final_top] += 1
        matched_key_facts = fact_meta.get("matched_key_facts", []) if isinstance(fact_meta, dict) else []
        failed_key_facts = fact_meta.get("failed_key_facts", []) if isinstance(fact_meta, dict) else []
        if not isinstance(matched_key_facts, (list, tuple)):
            matched_key_facts = [str(matched_key_facts)] if str(matched_key_facts) else []
        if not isinstance(failed_key_facts, (list, tuple)):
            failed_key_facts = [str(failed_key_facts)] if str(failed_key_facts) else []
        row = {
            "index": str(idx),
            "original_selection_index": str(original_index) if "original_index" in locals() else str(idx),
            "source_path": str(path),
            "preview_copy_path": preview_copy,
            "risky_sorted_copy_path": risky_sorted_copy_path,
            "forced_guess_copy_path": forced_guess_copy_path,
            "forced_guess_top": pred_top if read_status == "ok" else "",
            "forced_guess_internal_label": pred_label if read_status == "ok" else "",
            "forced_guess_label": public_label(pred_label) if read_status == "ok" else "",
            "risky_sorted_top": risky_sorted_top,
            "risky_sorted_internal_label": risky_sorted_label,
            "risky_sorted_label": public_label(risky_sorted_label) if risky_sorted_label else "",
            "risky_sorted_reason": risky_sorted_reason,
            "raw_prediction_copy_path": raw_prediction_copy,
            "broad_family_review_copy_path": broad_family_copy_path,
            "broad_family_candidate_top": broad_family_candidate_top,
            "broad_family_candidate_reason": broad_family_candidate_reason,
            "physical_role_review_copy_path": physical_role_copy_path,
            "physical_role_candidate_top": physical_role_candidate_top,
            "physical_role_candidate_label": physical_role_candidate_label,
            "physical_role_action": physical_role_action,
            "physical_role_candidate_reason": physical_role_candidate_reason,
            "final_top": final_top,
            "final_internal_label": final_label,
            "final_label": public_label(final_label),
            "raw_predicted_top": pred_top,
            "raw_predicted_internal_label": pred_label,
            "raw_predicted_label": public_label(pred_label),
            "confidence_status": confidence_status,
            "review_reason": review_reason,
            **two_voter_manifest,
            "sanity_warnings_report_only": sanity_warnings,
            "predicted_label_training_count": str(rel_info.get("training_count", "")),
            "predicted_label_support_tier": str(rel_info.get("support_tier", "")),
            "fact_score": f"{float(fact_meta.get('fact_score', 0.0) or 0.0):.6f}"
            if isinstance(fact_meta, dict)
            else "0.000000",
            "fact_penalty": f"{float(fact_meta.get('fact_penalty', 0.0) or 0.0):.6f}"
            if isinstance(fact_meta, dict)
            else "0.000000",
            "coherent_group_penalty": f"{float(fact_meta.get('coherent_group_penalty', 0.0) or 0.0):.6f}"
            if isinstance(fact_meta, dict)
            else "0.000000",
            "coherent_group_reason": str(
                fact_meta.get("coherent_group_reason", "") if isinstance(fact_meta, dict) else ""
            ),
            "rival_contrast_score": f"{float(fact_meta.get('rival_contrast_score', 0.0) or 0.0):.6f}"
            if isinstance(fact_meta, dict)
            else "0.000000",
            "top_rival_internal_label": str(
                fact_meta.get("top_rival_label", "") if isinstance(fact_meta, dict) else ""
            ),
            "top_rival_label": public_label(
                str(fact_meta.get("top_rival_label", "") if isinstance(fact_meta, dict) else "")
            ),
            "matched_key_facts": "; ".join(str(x) for x in matched_key_facts if str(x)),
            "failed_key_facts": "; ".join(str(x) for x in failed_key_facts if str(x)),
            "fact_profile_strength": str(
                fact_meta.get("fact_profile_strength", "") if isinstance(fact_meta, dict) else ""
            ),
            "rival_second_pass_swapped": str(
                fact_meta.get("rival_second_pass_swapped", "") if isinstance(fact_meta, dict) else ""
            ),
            "rival_second_pass_from_label": str(
                fact_meta.get("rival_second_pass_from_label", "") if isinstance(fact_meta, dict) else ""
            ),
            "rival_second_pass_to_label": str(
                fact_meta.get("rival_second_pass_to_label", "") if isinstance(fact_meta, dict) else ""
            ),
            "rival_second_pass_reason": str(
                fact_meta.get("rival_second_pass_reason", "") if isinstance(fact_meta, dict) else ""
            ),
            "rival_second_pass_score_gap": f"{float(fact_meta.get('rival_second_pass_score_gap', 0.0) or 0.0):.6f}"
            if isinstance(fact_meta, dict)
            else "0.000000",
            "weak_profile_rival_swapped": str(
                fact_meta.get("weak_profile_rival_swapped", "") if isinstance(fact_meta, dict) else ""
            ),
            "weak_profile_rival_from_label": str(
                fact_meta.get("weak_profile_rival_from_label", "") if isinstance(fact_meta, dict) else ""
            ),
            "weak_profile_rival_to_label": str(
                fact_meta.get("weak_profile_rival_to_label", "") if isinstance(fact_meta, dict) else ""
            ),
            "weak_profile_rival_reason": str(
                fact_meta.get("weak_profile_rival_reason", "") if isinstance(fact_meta, dict) else ""
            ),
            "weak_profile_rival_score_gap": f"{float(fact_meta.get('weak_profile_rival_score_gap', 0.0) or 0.0):.6f}"
            if isinstance(fact_meta, dict)
            else "0.000000",
            "committee_agreement_score": f"{float(fact_meta.get('committee_agreement_score', 0.0) or 0.0):.6f}"
            if isinstance(fact_meta, dict)
            else "0.000000",
            "committee_agreement_switched": str(
                fact_meta.get("committee_agreement_switched", False) if isinstance(fact_meta, dict) else False
            ),
            "committee_agreement_from_label": public_label(
                str(fact_meta.get("committee_agreement_from_label", "") if isinstance(fact_meta, dict) else "")
            ),
            "committee_agreement_to_label": public_label(
                str(fact_meta.get("committee_agreement_to_label", "") if isinstance(fact_meta, dict) else "")
            ),
            "committee_agreement_reason": str(
                fact_meta.get("committee_agreement_reason", "") if isinstance(fact_meta, dict) else ""
            ),
            "committee_agreement_advantage": f"{float(fact_meta.get('committee_agreement_advantage', 0.0) or 0.0):.6f}"
            if isinstance(fact_meta, dict)
            else "0.000000",
            "committee_votes": str(fact_meta.get("committee_votes", "") if isinstance(fact_meta, dict) else ""),
            "committee_top_candidates": str(
                fact_meta.get("committee_top_candidates", "") if isinstance(fact_meta, dict) else ""
            ),
            "measured_structure": str(fact_meta.get("measured_structure", "") if isinstance(fact_meta, dict) else ""),
            "measured_structure_confidence": str(
                fact_meta.get("measured_structure_confidence", "") if isinstance(fact_meta, dict) else ""
            ),
            "measured_structure_reason": str(
                fact_meta.get("measured_structure_reason", "") if isinstance(fact_meta, dict) else ""
            ),
            "similarity": f"{sim:.6f}",
            "margin_distance_gap": f"{margin:.6f}",
            **stage_diag,
            "duration_sec": f"{duration:.6f}",
            **current_physics,
            "nearest_predicted_training_examples": nearest_predicted_examples,
            "read_status": read_status,
            "top_match_1": public_label(top5[0][0]) if len(top5) > 0 else "",
            "top_match_1_internal": top5[0][0] if len(top5) > 0 else "",
            "top_match_1_dist": f"{top5[0][1]:.6f}" if len(top5) > 0 else "",
            "top_match_2": public_label(top5[1][0]) if len(top5) > 1 else "",
            "top_match_2_internal": top5[1][0] if len(top5) > 1 else "",
            "top_match_2_dist": f"{top5[1][1]:.6f}" if len(top5) > 1 else "",
            "top_match_3": public_label(top5[2][0]) if len(top5) > 2 else "",
            "top_match_3_internal": top5[2][0] if len(top5) > 2 else "",
            "top_match_3_dist": f"{top5[2][1]:.6f}" if len(top5) > 2 else "",
            "top_match_4": public_label(top5[3][0]) if len(top5) > 3 else "",
            "top_match_4_internal": top5[3][0] if len(top5) > 3 else "",
            "top_match_4_dist": f"{top5[3][1]:.6f}" if len(top5) > 3 else "",
            "top_match_5": public_label(top5[4][0]) if len(top5) > 4 else "",
            "top_match_5_internal": top5[4][0] if len(top5) > 4 else "",
            "top_match_5_dist": f"{top5[4][1]:.6f}" if len(top5) > 4 else "",
        }
        rows.append(row)
        if idx % 25 == 0:
            print(f"Real blind preview: sorted {idx} kept / {len(files)} selected before post-read skip", flush=True)
    fields = [
        "index",
        "original_selection_index",
        "source_path",
        "preview_copy_path",
        "forced_guess_copy_path",
        "forced_guess_top",
        "forced_guess_internal_label",
        "forced_guess_label",
        "risky_sorted_copy_path",
        "risky_sorted_top",
        "risky_sorted_internal_label",
        "risky_sorted_label",
        "risky_sorted_reason",
        "raw_prediction_copy_path",
        "broad_family_review_copy_path",
        "broad_family_candidate_top",
        "broad_family_candidate_reason",
        "physical_role_review_copy_path",
        "physical_role_candidate_top",
        "physical_role_candidate_label",
        "physical_role_action",
        "physical_role_candidate_reason",
        "final_top",
        "final_internal_label",
        "final_label",
        "raw_predicted_top",
        "raw_predicted_internal_label",
        "raw_predicted_label",
        "confidence_status",
        "review_reason",
        *TWO_VOTER_MANIFEST_FIELDS,
        "sanity_warnings_report_only",
        "predicted_label_training_count",
        "predicted_label_support_tier",
        "fact_score",
        "fact_penalty",
        "coherent_group_penalty",
        "coherent_group_reason",
        "rival_contrast_score",
        "top_rival_internal_label",
        "top_rival_label",
        "matched_key_facts",
        "failed_key_facts",
        "fact_profile_strength",
        "rival_second_pass_swapped",
        "rival_second_pass_from_label",
        "rival_second_pass_to_label",
        "rival_second_pass_reason",
        "rival_second_pass_score_gap",
        "weak_profile_rival_swapped",
        "weak_profile_rival_from_label",
        "weak_profile_rival_to_label",
        "weak_profile_rival_reason",
        "weak_profile_rival_score_gap",
        "committee_agreement_score",
        "committee_agreement_switched",
        "committee_agreement_from_label",
        "committee_agreement_to_label",
        "committee_agreement_reason",
        "committee_agreement_advantage",
        "committee_votes",
        "committee_top_candidates",
        "measured_structure",
        "measured_structure_confidence",
        "measured_structure_reason",
        "similarity",
        "margin_distance_gap",
        "learned_top_1",
        "learned_top_1_score",
        "learned_top_2",
        "learned_top_2_score",
        "learned_top_gap",
        "learned_top_gap_ratio",
        "learned_structure",
        "learned_structure_score",
        "label_structure",
        "label_structure_score",
        "structure_gap_raw",
        "structure_gap_normalized",
        "duration_sec",
        *PHYSICS_REPORT_FIELDS,
        "nearest_predicted_training_examples",
        "read_status",
        "top_match_1",
        "top_match_1_internal",
        "top_match_1_dist",
        "top_match_2",
        "top_match_2_internal",
        "top_match_2_dist",
        "top_match_3",
        "top_match_3_internal",
        "top_match_3_dist",
        "top_match_4",
        "top_match_4_internal",
        "top_match_4_dist",
        "top_match_5",
        "top_match_5_internal",
        "top_match_5_dist",
    ]
    write_csv(reports_dir / "real_sort_preview_manifest.csv", rows, fields)
    write_committee_agreement_report(rows, reports_dir)
    # Write separate real-preview trees without overwriting sorted_preview_tree.txt.
    write_tree_file(sorted_root, reports_dir / "real_sorted_preview_tree.txt")
    write_tree_file(forced_guess_root, reports_dir / "real_forced_guess_preview_tree.txt")
    write_tree_file(risky_sorted_root, reports_dir / "DIAGNOSTIC_RISKY_GUESSES_NOT_A_SORT_tree.txt")
    write_tree_file(raw_prediction_root, reports_dir / "real_raw_brain_prediction_preview_tree.txt")
    write_tree_file(broad_family_root, reports_dir / "real_broad_family_review_preview_tree.txt")
    write_tree_file(physical_role_root, reports_dir / "real_physical_role_review_preview_tree.txt")
    no_show_report = write_real_sort_label_no_show_report(brain, rows, reports_dir)
    summary = {
        "status": "ok" if files else "no_files_or_root_missing",
        "root": str(root),
        "files_selected_before_post_read_skip": len(files),
        "files_selected": len(rows),
        "post_read_skip_counts": dict(sorted(skipped_after_read.items())),
        "max_files": int(getattr(args, "real_preview_max_files", 0) or 0),
        "per_folder": int(getattr(args, "real_preview_per_folder", 5) or 5),
        "official_audio_copy_limit_per_label": official_copy_limit,
        "force_guess_enabled": force_guess_enabled,
        "force_guess_copy_limit_per_label": forced_guess_copy_limit,
        "raw_audio_copy_limit_per_label": raw_copy_limit,
        "risky_audio_copy_limit_per_label": risky_copy_limit,
        "risky_min_similarity": risky_min_similarity,
        "broad_family_review_copy_limit_per_label": broad_copy_limit,
        "broad_family_min_learned_top_gap": broad_min_gap,
        "official_audio_copy_attempts": int(sum(official_copy_counts.values())),
        "force_guess_copy_attempts": int(sum(forced_guess_copy_counts.values())),
        "raw_audio_copy_attempts": int(sum(raw_copy_counts.values())),
        "risky_audio_copy_attempts": int(sum(risky_copy_counts.values())),
        "broad_family_review_copy_attempts": int(sum(broad_copy_counts.values())),
        "physical_role_review_copy_attempts": int(sum(physical_role_counts.values())),
        "official_audio_copy_skipped_after_limit": int(official_copy_skipped_after_limit),
        "force_guess_copy_skipped_after_limit": int(forced_guess_copy_skipped_after_limit),
        "raw_audio_copy_skipped_after_limit": int(raw_copy_skipped_after_limit),
        "risky_audio_copy_skipped_after_limit": int(risky_copy_skipped_after_limit),
        "broad_family_review_copy_skipped_after_limit": int(broad_copy_skipped_after_limit),
        "physical_role_review_copy_skipped_after_limit": int(physical_role_copy_skipped_after_limit),
        "excluded_training_folders": sorted(REAL_PREVIEW_EXCLUDE_PARTS),
        "excluded_resolved_training_audio_files": len(excluded_training_files),
        "final_top_counts": dict(sorted(final_top_counts.items())),
        "broad_family_review_candidate_counts": dict(sorted(broad_family_counts.items())),
        "broad_family_conflict_override_gap": broad_conflict_override_gap,
        "manifest": str(reports_dir / "real_sort_preview_manifest.csv"),
        "preview_folder": str(sorted_root),
        "forced_guess_preview_folder": str(forced_guess_root),
        "forced_guess_preview_tree": str(reports_dir / "real_forced_guess_preview_tree.txt"),
        "diagnostic_risky_guesses_not_a_sort_folder": str(risky_sorted_root),
        "risky_sorted_preview_tree": str(reports_dir / "DIAGNOSTIC_RISKY_GUESSES_NOT_A_SORT_tree.txt"),
        "raw_prediction_preview_folder": str(raw_prediction_root),
        "raw_prediction_preview_tree": str(reports_dir / "real_raw_brain_prediction_preview_tree.txt"),
        "broad_family_review_preview_folder": str(broad_family_root),
        "broad_family_review_preview_tree": str(reports_dir / "real_broad_family_review_preview_tree.txt"),
        "physical_role_review_preview_folder": str(physical_role_root),
        "physical_role_review_preview_tree": str(reports_dir / "real_physical_role_review_preview_tree.txt"),
        "label_no_show_report": str(no_show_report),
    }
    write_json(reports_dir / "real_sort_preview_summary.json", summary)
    return summary


def run_build_eval(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else project_dir / f"Phase3_Pure_Brain_Lab_Run_{stamp}"
    )
    reports_dir = run_dir / "reports"
    brain_dir = run_dir / "brain"
    logs_dir = run_dir / "logs"
    upload_dir = run_dir / "upload"
    for d in [reports_dir, brain_dir, logs_dir, upload_dir]:
        d.mkdir(parents=True, exist_ok=True)

    source_mode = str(getattr(args, "source_mode", "tree") or "tree").strip().lower()
    if args.report_dir:
        report_dir = Path(args.report_dir).expanduser().resolve()
    elif source_mode == "manifest":
        report_dir = find_latest_physics_report(root)
    else:
        report_dir = root / "_MANIFESTS"
    clean_csv = report_dir / "candidate_clean_training_set.csv"
    supplemental_roots: List[Path] = []
    if source_mode == "manifest":
        clean_rows = read_csv(clean_csv)
        training_source_note = f"manifest CSV: {clean_csv}"
    else:
        clean_rows = scan_training_tree_rows(root)
        training_source_note = f"live trusted folder tree: {root}"
        if not bool(getattr(args, "no_supplemental_training", False)):
            for item in list(getattr(args, "supplemental_training_root", []) or []):
                p = Path(str(item)).expanduser()
                if p.exists():
                    supplemental_roots.append(p)
                    rows = scan_training_tree_rows(p, source_name="supplemental_training")
                    clean_rows.extend(rows)
        if supplemental_roots:
            training_source_note += " + supplemental training roots: " + ", ".join(str(p) for p in supplemental_roots)
    allowed_top = parse_allowed(args.allowed_top)

    print(f"Run folder: {run_dir}")
    print(f"Training source: {training_source_note}")
    if source_mode != "manifest":
        print(f"Reference manifest for comparison only: {clean_csv}")
    print(f"Allowed top folders: {', '.join(sorted(allowed_top))}")
    print("Old sorter runtime: NOT USED")
    print()

    train_rows, eval_rows, skipped_rows = select_rows(
        clean_rows,
        allowed_top=allowed_top,
        max_train_per_group=args.max_train_per_group,
        max_eval_per_label=args.max_eval_per_label,
        max_eval_total=args.max_eval_total,
        random_seed=args.random_seed,
        min_train_after_holdout=args.min_active_train_per_label,
    )

    # Report terminal structure labels and possible structure dirt.
    remap_rows = []
    conflict_rows = []
    for row in train_rows + eval_rows:
        group_key = str(row.get("group_key", "")).replace("\\", "/").strip()
        dur = row_duration_sec(row)
        label = row.get("_expected_label", "")
        structure = row.get("_structure", "") or structure_lane_for_row(row, group_key)
        if structure == "loop":
            remap_rows.append(
                {
                    "source_path": str(row.get("source_path", "")),
                    "group_key": group_key,
                    "duration_sec": f"{dur:.6f}",
                    "structure": structure,
                    "mapped_label": label,
                    "reason": "loop_terminal_leaf_assigned_from_training_hierarchy",
                }
            )
        warning = str(row.get("_structure_conflict_warning", ""))
        if warning:
            conflict_rows.append(
                {
                    "source_path": str(row.get("source_path", "")),
                    "group_key": group_key,
                    "duration_sec": f"{dur:.6f}",
                    "structure": structure,
                    "expected_label": label,
                    "warning": warning,
                }
            )

    write_csv(
        reports_dir / "selection_structure_lane_enforcement.csv",
        remap_rows,
        ["source_path", "group_key", "duration_sec", "structure", "mapped_label", "reason"],
    )
    write_csv(
        reports_dir / "selection_structure_conflict_warnings.csv",
        conflict_rows,
        ["source_path", "group_key", "duration_sec", "structure", "expected_label", "warning"],
    )

    write_csv(reports_dir / "selection_skipped_rows.csv", skipped_rows, ["source_path", "group_key", "reason"])

    structure_remap_rows = []
    for row in train_rows + eval_rows:
        if str(row.get("_structure_remap_reason", "")):
            structure_remap_rows.append(
                {
                    "source_path": str(row.get("source_path", "")),
                    "group_key": str(row.get("group_key", "")),
                    "structure_before_remap": str(row.get("_structure_before_remap", "")),
                    "structure_after_remap": str(row.get("_structure", "")),
                    "expected_label": str(row.get("_expected_label", "")),
                    "remap_reason": str(row.get("_structure_remap_reason", "")),
                }
            )
    write_csv(
        reports_dir / "selection_structure_remap_two_lane.csv",
        structure_remap_rows,
        [
            "source_path",
            "group_key",
            "structure_before_remap",
            "structure_after_remap",
            "expected_label",
            "remap_reason",
        ],
    )

    selection_label_rows = []
    label_keys = sorted(set(str(r.get("_expected_label", "")) for r in train_rows + eval_rows))
    for label in label_keys:
        train_for_label = [r for r in train_rows if str(r.get("_expected_label", "")) == label]
        eval_for_label = [r for r in eval_rows if str(r.get("_expected_label", "")) == label]
        public = public_label(label)
        selection_label_rows.append(
            {
                "internal_label": label,
                "public_label": public,
                "top": top_for_public_label(public),
                "structure": label_default_structure(label),
                "train_selected": str(len(train_for_label)),
                "eval_selected": str(len(eval_for_label)),
                "eval_reused_training_source_count": str(
                    sum(1 for r in eval_for_label if str(r.get("_eval_reused_training_source", "0")) == "1")
                ),
                "sampling_pool_size": str(
                    max([int(str(r.get("_sampling_pool_size", "0") or "0")) for r in train_for_label] + [0])
                ),
            }
        )
    write_csv(
        reports_dir / "selection_label_counts.csv",
        selection_label_rows,
        [
            "internal_label",
            "public_label",
            "top",
            "structure",
            "train_selected",
            "eval_selected",
            "eval_reused_training_source_count",
            "sampling_pool_size",
        ],
    )

    selection_summary = {
        "clean_csv": str(clean_csv),
        "training_source_mode": source_mode,
        "training_source_note": training_source_note,
        "supplemental_training_roots": [str(p) for p in supplemental_roots],
        "supplemental_training_enabled": not bool(getattr(args, "no_supplemental_training", False)),
        "raw_rows_loaded_before_selection": len(clean_rows),
        "allowed_top": sorted(allowed_top),
        "raw_train_candidates_selected_before_curation": len(train_rows),
        "eval_selected_rows": len(eval_rows),
        "skipped_rows": len(skipped_rows),
        "max_train_per_group": args.max_train_per_group,
        "source_cap_per_label": args.source_cap_per_label,
        "dry_core_augment_policy": str(getattr(args, "dry_core_augment_policy", "") or ""),
        "dry_core_augment_scope": "trusted training rows only; voice/vocal and saxophone labels only; eval rows are not augmented",
        "dry_core_augment_timeout_sec": float(getattr(args, "dry_core_augment_timeout_sec", 45.0) or 45.0),
        "min_active_train_per_label": args.min_active_train_per_label,
        "min_source_groups_per_label": args.min_source_groups_per_label,
        "random_seed": args.random_seed,
        "max_eval_per_label": args.max_eval_per_label,
        "max_eval_total": args.max_eval_total,
        "curate_max_keep_per_label": args.curate_max_keep_per_label,
        "curate_fx_max_keep_per_label": args.curate_fx_max_keep_per_label,
        "curate_central_pool_fraction": args.curate_central_pool_fraction,
        "curate_fx_central_pool_fraction": args.curate_fx_central_pool_fraction,
        "structure_leaf_policy": "One Shots / Loops are terminal child folders inside each category, not top-level labels",
        "prediction_policy": "pure brain global match; expected structure is not passed to predict(); each label is scored with that label structure scaler",
        "real_sort_preview_policy": "blind preview excludes Sorted samples, training zips/folders, reports, and generated preview folders; source filenames/folder names are never used as placement evidence",
        "train_top_counts": dict(sorted(Counter(str(r.get("_expected_top", "")) for r in train_rows).items())),
        "eval_top_counts": dict(sorted(Counter(str(r.get("_expected_top", "")) for r in eval_rows).items())),
        "train_structure_counts": dict(sorted(Counter(str(r.get("_structure", "")) for r in train_rows).items())),
        "eval_structure_counts": dict(sorted(Counter(str(r.get("_structure", "")) for r in eval_rows).items())),
        "eval_reused_training_source_count": sum(
            1 for r in eval_rows if str(r.get("_eval_reused_training_source", "0")) == "1"
        ),
        "selection_policy": "locked training is read-only; train rows come from folder truth; max_train_per_group<=0 trains every readable row; eval rows in train-all mode are marked as diagnostic reused training recall, not holdout accuracy",
        "long_running_policy": "v20260512: _LONG_FX and _LONG_RUNNING are structure markers. They are stripped from the middle of public labels but preserved as terminal /Long FX or /Long Running labels in the brain; physics conflicts are report-only.",
        "mixed_loop_policy": "mixed loop folder names are trusted when Aaron placed them; no filename/source-name rerouting is allowed",
        "loop_component_spy_policy": "loop_component_spy.csv compares loop teachers to one-shot teacher fingerprints as a first-pass check; it is not event segmentation yet",
        "feature_policy": "FP_SIZE 50: original MFCC/spectral/envelope features plus sub/bass/mid/presence/air ratios, ZCR mean, temporal centroid, onset regularity, attack rise time, onset span ratio, event rate, tail energy ratio, spectral flux variance, and pitch confidence",
        "centroid_policy": "adaptive per-folder model: tiny folders use exemplars; uniform folders use one centroid; messy folders use deterministic multi-centroids plus exemplar anchors. Model choice is based on physics spread, not folder names.",
        "normalization_policy": "per-structure label-side scalers plus per-top distance normalization are used during global prediction; no expected structure is passed to eval",
        "top_prefilter_policy": "learned top-3 super-category prefilter before label scoring; not filename or rule routing",
        "gate_policy": "v0.4.57 locked-training gate: scan the live trusted folder tree, keep every readable teacher, reserve only true holdout eval, strip structure marker folders from labels, and keep filename text out of routing",
        "sanity_warning_policy": "fingerprint sanity warnings are report-only and never change placement in Phase 3",
    }
    write_json(reports_dir / "selection_summary.json", selection_summary)

    print(f"Training rows selected: {len(train_rows)}")
    print(f"Eval rows selected: {len(eval_rows)}")
    print()

    dry_core_augment_policy = (
        "voice_sax"
        if bool(getattr(args, "augment_dry_core_voice_sax", False))
        else str(getattr(args, "dry_core_augment_policy", "") or "")
    )
    dry_core_augment_timeout_sec = float(getattr(args, "dry_core_augment_timeout_sec", 45.0) or 45.0)
    if dry_core_augment_policy:
        print(
            "Dry-core training augmentation active for trusted Voice/Vocal and Saxophone labels only. "
            "Original audio files are not modified.",
            flush=True,
        )
    print("Extracting training features...")
    train_features = extract_feature_rows(
        train_rows,
        reports_dir,
        "train",
        dry_core_augment_policy=dry_core_augment_policy,
        dry_core_augment_timeout_sec=dry_core_augment_timeout_sec,
    )
    print("Extracting eval features...")
    eval_features = extract_feature_rows(eval_rows, reports_dir, "eval")

    eval_features = remove_eval_training_overlaps(train_features, eval_features, reports_dir)
    train_features = ensure_eval_labels_have_training_support(train_features, eval_features, reports_dir)

    # Drop unreadable rows for training. Keep unreadable eval rows as failures.
    readable_train = [r for r in train_features if r.read_status == "ok"]
    print(f"Readable training rows before curation: {len(readable_train)}")
    copy_training_preview(readable_train, run_dir, reports_dir, max_per_label=args.training_preview_per_label)

    curated_train = curate_training_features(
        readable_train,
        reports_dir,
        run_dir,
        min_group_size=args.curate_min_group_size,
        max_keep_per_label=args.curate_max_keep_per_label,
        fx_max_keep_per_label=args.curate_fx_max_keep_per_label,
        fx_central_pool_fraction=args.curate_fx_central_pool_fraction,
        max_reject_fraction=args.curate_max_reject_fraction,
        preview_keep_per_label=args.curated_preview_per_label,
        preview_reject_per_label=args.curated_preview_per_label,
        central_pool_fraction=args.curate_central_pool_fraction,
        random_seed=args.random_seed,
        source_cap_per_label=args.source_cap_per_label,
        min_active_train_per_label=args.min_active_train_per_label,
        min_source_groups_per_label=args.min_source_groups_per_label,
        copy_gold_workspace=not args.no_gold_workspace,
    )
    print(f"Readable training rows after curation: {len(curated_train)}", flush=True)

    skip_expensive_diagnostics = bool(getattr(args, "skip_expensive_training_diagnostics", False))
    print("Building folder brain: centroids, exemplars, structure head, and support metadata...", flush=True)
    brain = build_brain(curated_train, max_centroids=args.max_centroids)
    raw_brain_rows = sum(int(v) for v in (brain.get("raw_counts", {}) or {}).values())
    active_math_rows = sum(int(v) for v in (brain.get("effective_counts", {}) or {}).values())
    excluded_math_rows = 0
    try:
        for _lbl, meta in (brain.get("effective_training_balance_by_label", {}) or {}).items():
            if isinstance(meta, dict):
                excluded_math_rows += int(meta.get("excluded_from_math_count", 0) or 0)
    except Exception:
        excluded_math_rows = 0
    print(
        f"Brain math pool: raw_readable={raw_brain_rows} active_effective={active_math_rows} excluded_from_math={excluded_math_rows}",
        flush=True,
    )
    brain["run_build_eval_math_pool_summary"] = {
        "raw_readable_training_rows": int(raw_brain_rows),
        "active_effective_training_rows": int(active_math_rows),
        "excluded_from_math_rows": int(excluded_math_rows),
        "policy": "Trusted folder tree remains intact; high-confidence structure contradictions are excluded only from scaler/centroid/router math.",
    }
    brain["expensive_training_diagnostics_skipped"] = bool(skip_expensive_diagnostics)
    brain["expensive_training_diagnostics_policy"] = (
        "Skipped for train-brain/product workflow; run build-eval without skip_expensive_training_diagnostics "
        "when deep developer-only cross-boundary reports are needed."
        if skip_expensive_diagnostics
        else "Full developer-only training diagnostics were computed."
    )
    print("Writing balanced effective training report...", flush=True)
    write_effective_training_balance_report(brain, reports_dir)

    if skip_expensive_diagnostics:
        print("Skipping expensive developer-only training diagnostics for product brain build.", flush=True)
        print("Writing lightweight training label spread report...", flush=True)
        write_training_label_spread_report(brain, reports_dir)
    else:
        print("Computing label isolation diagnostics...", flush=True)
        populate_label_isolation_metrics(brain, curated_train)
        print("Writing training label spread report...", flush=True)
        write_training_label_spread_report(brain, reports_dir)
        print("Writing cross-boundary training diagnostics...", flush=True)
        write_training_cross_boundary_report(brain, curated_train, reports_dir)
        print("Writing training recall contract diagnostics...", flush=True)
        write_training_recall_contract_report(brain, curated_train, reports_dir)

    brain_path = brain_dir / "phase3_pure_scratch_brain.json"
    print(f"Writing brain JSON: {brain_path}", flush=True)
    write_json(brain_path, brain)
    print("Writing eval activation audit...", flush=True)
    write_eval_activation_audit(brain, eval_features, reports_dir)

    # Sanity: all brain labels must belong to enabled production top folders.
    forbidden = []
    forbidden_top = [x for x in brain["labels"] if brain["top_by_label"].get(x) not in allowed_top]
    validation = {
        "status": "PASS" if not forbidden and not forbidden_top else "FAIL",
        "forbidden_prefix_labels": forbidden,
        "forbidden_top_labels": forbidden_top,
        "label_count": len(brain["labels"]),
        "training_count": sum(int(v) for v in brain["counts"].values()),
    }
    write_json(reports_dir / "brain_validation.json", validation)
    if validation["status"] != "PASS":
        print("Brain validation failed. See reports/brain_validation.json")
        return 2

    sorted_root = run_dir / "sorted_preview"
    summary = evaluate(
        brain,
        eval_features,
        reports_dir,
        sorted_root,
        args.min_similarity,
        args.min_margin,
        args.weak_similarity_floor,
        args.strong_margin,
        bool(getattr(args, "allow_low_similarity_clear_margin", False)),
    )
    real_preview_summary = run_real_sort_preview(brain, args, run_dir, reports_dir)
    if args.copy_listen_pack:
        copy_audio_samples_for_listening(eval_features, reports_dir)

    counts_by_label = Counter(r.label for r in curated_train)
    with (reports_dir / "brain_training_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Aaron Sound Sorter Stage 4 Brain Training Summary\n\n")
        f.write("This run did not call Aaron_Sound_Sorter.py.\n")
        f.write(f"Brain path: {brain_path}\n")
        f.write(f"Labels: {len(brain['labels'])}\n")
        f.write(f"Readable training rows before curation: {len(readable_train)}\n")
        f.write(f"Readable training rows after curation: {len(curated_train)}\n\n")
        f.write("Training counts by label, raw -> effective competitive rows:\n")
        balance_info = brain.get("effective_training_balance_by_label", {})
        for label, count in sorted(counts_by_label.items()):
            meta = balance_info.get(label, {}) if isinstance(balance_info, dict) else {}
            eff = (
                int(meta.get("effective_count", brain.get("effective_counts", {}).get(label, count)) or count)
                if isinstance(meta, dict)
                else count
            )
            cap = str(meta.get("cap_applied", "")) if isinstance(meta, dict) else ""
            f.write(f"  {count:5d} -> {eff:5d}  {label}  cap_applied={cap}\n")

    readme = run_dir / "README_RUN_RESULT.txt"
    readme.write_text(
        "Aaron Sound Sorter Stage 4 v0.5.7 COMMITTEE_PHYSICS_GAP_LOCKS\n\n"
        "This run did not call Aaron_Sound_Sorter.py.\n"
        "Old sorter judge/router logic was not used.\n\n"
        f"Brain: {brain_path}\n"
        f"Reports: {reports_dir}\n"
        f"Raw eval label accuracy: {summary['raw_label_accuracy']:.2%}\n"
        f"Raw eval top accuracy: {summary['raw_top_accuracy']:.2%}\n"
        f"Auto-placed: {summary['auto_placed']}\n"
        f"Sent to review: {summary['sent_to_review']}\n"
        f"Auto-place label accuracy: {summary['auto_place_label_accuracy']:.2%}\n"
        f"Auto-place top accuracy: {summary['auto_place_top_accuracy']:.2%}\n"
        f"Sorted preview folder: {sorted_root}\n"
        f"Real blind preview status: {real_preview_summary.get('status', '')}\n"
        f"Real blind preview files: {real_preview_summary.get('files_selected', 0)}\n"
        f"Real blind preview folder: {real_preview_summary.get('preview_folder', '')}\n"
        f"Review-only broad family candidates: {real_preview_summary.get('broad_family_review_preview_folder', '')}\n",
        encoding="utf-8",
    )

    write_runtime_pointer(project_dir, "phase3_latest_pure_brain_run.txt", run_dir)
    write_runtime_pointer(project_dir, "phase3_latest_pure_brain_path.txt", brain_path)

    print()
    print(f"Brain: {brain_path}")
    print(f"Eval summary: {reports_dir / 'pure_brain_eval_summary.txt'}")
    print(f"Real preview summary: {reports_dir / 'real_sort_preview_summary.json'}")
    if (
        bool(getattr(args, "suppress_upload_zip", False))
        or os.environ.get("AARON_SOUND_SORTER_SUPPRESS_LAB_UPLOAD_ZIP", "0") == "1"
    ):
        print("Upload ZIP: suppressed; runner will create a small logs-only ZIP if needed")
        return 0
    zip_path = zip_run(run_dir)
    print(f"Upload ZIP: {zip_path}")
    return 0
