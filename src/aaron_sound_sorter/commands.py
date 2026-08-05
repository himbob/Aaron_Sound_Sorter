# Auto-split from Aaron_Sound_Sorter.py.
# This is a component module, not a legacy wrapper.
from __future__ import annotations

import contextlib
import subprocess
import tempfile
from dataclasses import dataclass

from aaron_audio_intelligence.shape_memory_brain import SHAPE_STARTER_MEMORY_BRAIN_NAME

from .brain_family_training import AnchorSelectionConfig, build_brain_family_training_trees
from .core import *
from .shape_memory_training import (
    DEFAULT_STARTER_SHAPE_MEMORY_EXAMPLES_PER_LABEL,
    DEFAULT_STARTER_SHAPE_MEMORY_WEIGHT,
    train_shape_memory_starter_from_tree,
)

# Visible product/developer policy values. Keep defaults here instead of hiding
# mystery numbers inside command methods. Command methods should read like a
# sequence of steps, not a pile of knobs.
UNLIMITED_PREVIEW_COPY = 999999
PRODUCT_SORT_OFFICIAL_COPY_PER_LABEL = -1
PRODUCT_SORT_WEAK_SIMILARITY_FLOOR = 0.20
PRODUCT_SORT_STRONG_MARGIN = 1.50
PRODUCT_SORT_RISKY_MIN_SIMILARITY = 0.80
PRODUCT_SORT_BROAD_CONFLICT_OVERRIDE_GAP = 0.80
TRAIN_BRAIN_WEAK_SIMILARITY_FLOOR = 0.20
TRAIN_BRAIN_STRONG_MARGIN = 1.50
TRAIN_BRAIN_RISKY_MIN_SIMILARITY = 0.80
TRAIN_BRAIN_BROAD_CONFLICT_OVERRIDE_GAP = 0.80
RUNTIME_CONFIG_RELATIVE = Path("config") / "runtime"


def runtime_pointer_path(project_dir: Path, filename: str) -> Path:
    """Return a runtime-pointer path and ensure its parent exists.

    Args:
        project_dir: Repository root.
        filename: Lowercase pointer filename.

    Returns:
        Path under ``config/runtime``.

    Side Effects:
        Creates the runtime configuration folder when missing.
    """
    pointer_path = project_dir / RUNTIME_CONFIG_RELATIVE / filename
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    return pointer_path


def portable_project_path(project_dir: Path, target_path: Path) -> str:
    """Serialize a path relative to the project when possible.

    Args:
        project_dir: Repository root used as the relative-path anchor.
        target_path: Local file or folder path.

    Returns:
        A portable relative path for project-owned targets, otherwise an
        absolute path.
    """
    resolved_project = project_dir.expanduser().resolve()
    resolved_target = target_path.expanduser().resolve()
    try:
        return str(resolved_target.relative_to(resolved_project))
    except ValueError:
        return str(resolved_target)


def resolve_project_pointer(project_dir: Path, pointer_value: str) -> Path:
    """Resolve an absolute or project-relative runtime pointer.

    Args:
        project_dir: Repository root used for relative pointer values.
        pointer_value: Text read from a runtime pointer.

    Returns:
        An absolute resolved path.
    """
    candidate = Path(pointer_value).expanduser()
    if not candidate.is_absolute():
        candidate = project_dir / candidate
    return candidate.resolve()


def shape_memory_starter_output_path(base_brain_path: Path, requested_path: str | None = None) -> Path:
    """Return the starter ShapeVoter memory path for a training command.

    Args:
        base_brain_path: Full brain path whose sibling is used by default.
        requested_path: Optional explicit output path from CLI arguments.

    Returns:
        Absolute starter memory brain path.

    Side Effects:
        None.
    """
    raw = str(requested_path or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(base_brain_path).expanduser().resolve().with_name(SHAPE_STARTER_MEMORY_BRAIN_NAME)


def maybe_train_shape_memory_starter_for_cli(
    *,
    args: argparse.Namespace,
    training_root: Path,
    base_brain_path: Path,
    report_dir: Path,
) -> int:
    """Train the optional low-trust starter ShapeVoter memory brain.

    Args:
        args: Parsed training command arguments.
        training_root: Trusted training tree used by the folder brain.
        base_brain_path: Full brain JSON providing feature scaler metadata.
        report_dir: Report folder for starter-memory audit files.

    Returns:
        Zero on success or when disabled; nonzero when requested training fails.

    Side Effects:
        May write ``stage4_shape_memory_starter_brain.json`` and reports.
    """
    if not bool(getattr(args, "train_shape_memory", True)):
        print("ShapeVoter starter memory training disabled.", flush=True)
        return 0
    base_brain_path = Path(base_brain_path).expanduser().resolve()
    if not base_brain_path.exists():
        print(f"WARNING: shape-memory starter skipped; base brain missing: {base_brain_path}", flush=True)
        return 0
    output_path = shape_memory_starter_output_path(
        base_brain_path,
        str(getattr(args, "save_shape_memory_brain", "") or ""),
    )
    print(f"Training low-trust ShapeVoter starter memory: {output_path}", flush=True)
    try:
        summary = train_shape_memory_starter_from_tree(
            training_root=training_root,
            base_brain_path=base_brain_path,
            output_brain_path=output_path,
            report_dir=report_dir,
            allowed_top=str(getattr(args, "allowed_top", "Drums,Instruments,FX,Textures")),
            max_examples_per_label=int(
                getattr(args, "shape_memory_train_per_label", DEFAULT_STARTER_SHAPE_MEMORY_EXAMPLES_PER_LABEL)
            ),
            evidence_weight=int(getattr(args, "shape_memory_evidence_weight", DEFAULT_STARTER_SHAPE_MEMORY_WEIGHT)),
            random_seed=int(getattr(args, "random_seed", 20260503)),
            apply=True,
        )
    except Exception as exc:
        print(f"ShapeVoter starter memory training failed: {exc}", flush=True)
        return 1
    print(f"Saved low-trust ShapeVoter starter memory: {summary.output_brain_path}", flush=True)
    print(f"ShapeVoter starter report: {summary.report_dir}", flush=True)
    return 0


@dataclass(frozen=True)
class TrainBrainBuildRequest:
    """Explicit configuration passed to the old build/eval trainer."""

    root: str
    project_dir: str
    report_dir: str
    run_dir: str
    allowed_top: str
    source_mode: str = "tree"
    supplemental_training_root: list[str] = None
    no_supplemental_training: bool = True
    max_train_per_group: int = 0
    max_eval_per_label: int = 0
    max_eval_total: int = 0
    max_centroids: int = 6
    min_similarity: float = 0.0
    min_margin: float = 0.0
    weak_similarity_floor: float = TRAIN_BRAIN_WEAK_SIMILARITY_FLOOR
    strong_margin: float = TRAIN_BRAIN_STRONG_MARGIN
    allow_low_similarity_clear_margin: bool = False
    training_preview_per_label: int = 0
    random_seed: int = 17
    curate_min_group_size: int = 1
    curate_max_keep_per_label: int = UNLIMITED_PREVIEW_COPY
    curate_fx_max_keep_per_label: int = UNLIMITED_PREVIEW_COPY
    curate_fx_central_pool_fraction: float = 1.0
    curate_max_reject_fraction: float = 1.0
    curate_central_pool_fraction: float = 1.0
    curated_preview_per_label: int = 0
    source_cap_per_label: int = UNLIMITED_PREVIEW_COPY
    min_active_train_per_label: int = 1
    min_source_groups_per_label: int = 1
    no_gold_workspace: bool = True
    copy_listen_pack: bool = False
    real_preview_root: str = ""
    real_preview_max_files: int = 0
    real_preview_per_folder: int = 0
    real_preview_official_copy_per_label: int = 0
    real_preview_force_guess: bool = False
    real_preview_force_guess_copy_per_label: int = 0
    real_preview_raw_copy_per_label: int = 0
    real_preview_risky_copy_per_label: int = 0
    real_preview_risky_min_similarity: float = TRAIN_BRAIN_RISKY_MIN_SIMILARITY
    real_preview_broad_family_min_gap: float = 0.0
    real_preview_broad_family_conflict_override_gap: float = TRAIN_BRAIN_BROAD_CONFLICT_OVERRIDE_GAP
    real_preview_broad_copy_per_label: int = 0
    real_preview_physical_role_copy_per_label: int = 0
    no_real_sort_preview: bool = True
    skip_expensive_training_diagnostics: bool = True
    dry_core_augment_policy: str = ""
    dry_core_augment_timeout_sec: float = 45.0
    suppress_upload_zip: bool = True

    def __post_init__(self) -> None:
        if self.supplemental_training_root is None:
            object.__setattr__(self, "supplemental_training_root", [])


@dataclass(frozen=True)
class SortPreviewRequest:
    """Explicit product-sort request passed to the preview sorter engine."""

    root: str
    real_preview_root: str
    min_similarity: float
    min_margin: float
    random_seed: int
    real_preview_max_files: int = 0
    real_preview_per_folder: int = UNLIMITED_PREVIEW_COPY
    real_preview_official_copy_per_label: int = PRODUCT_SORT_OFFICIAL_COPY_PER_LABEL
    real_preview_force_guess: bool = False
    real_preview_force_guess_copy_per_label: int = 0
    real_preview_raw_copy_per_label: int = 0
    real_preview_risky_copy_per_label: int = 0
    real_preview_risky_min_similarity: float = PRODUCT_SORT_RISKY_MIN_SIMILARITY
    real_preview_broad_family_min_gap: float = 0.0
    real_preview_broad_family_conflict_override_gap: float = PRODUCT_SORT_BROAD_CONFLICT_OVERRIDE_GAP
    real_preview_broad_copy_per_label: int = 0
    real_preview_physical_role_copy_per_label: int = 0
    no_real_sort_preview: bool = False
    real_preview_exclude_default_training_roots: bool = False
    weak_similarity_floor: float = PRODUCT_SORT_WEAK_SIMILARITY_FLOOR
    strong_margin: float = PRODUCT_SORT_STRONG_MARGIN
    allow_low_similarity_clear_margin: bool = False


def safe_audio_parts(raw_name: str) -> list[str]:
    """Return safe relative audio path parts or an empty list."""
    name = str(raw_name).replace("\\", "/")
    parts = [part for part in name.split("/") if part]
    if not parts or any(part in {"..", ""} for part in parts):
        return []
    if any(part == "__MACOSX" or part.startswith("._") or part.startswith(".") for part in parts):
        return []
    if Path(parts[-1]).suffix.lower() not in AUDIO_EXTS:
        return []
    return list(parts)


def link_audio_tree(source_root: Path, destination_root: Path) -> int:
    """Symlink all safe audio files from a folder into a staging tree."""
    count = 0
    for source_file in sorted(source_root.rglob("*")):
        if not source_file.is_file():
            continue
        parts = safe_audio_parts(str(source_file.relative_to(source_root)))
        if not parts:
            continue
        target = destination_root.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            target.unlink()
        target.symlink_to(source_file.resolve())
        count += 1
    return count


def extract_safe_audio_zip(input_path: Path, destination_root: Path) -> int:
    """Extract only safe audio members from a ZIP into a temporary tree."""
    extracted_count = 0
    with zipfile.ZipFile(input_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = safe_audio_parts(info.filename)
            if not parts:
                continue
            target = destination_root.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            extracted_count += 1
    return extracted_count


def resolve_brain_path(project_dir: Path, requested_brain: str) -> Path:
    """Resolve the explicitly requested brain or the latest trained brain pointer."""
    brain_path = resolve_project_pointer(project_dir, requested_brain or DEFAULT_FOLDER_BRAIN)
    if brain_path.exists():
        return brain_path
    latest = runtime_pointer_path(project_dir, "phase4_latest_folder_brain_path.txt")
    if latest.exists():
        candidate = resolve_project_pointer(project_dir, latest.read_text(encoding="utf-8").strip())
        if candidate.exists():
            return candidate
    return brain_path


def product_sort_training_exclusion_root(args: argparse.Namespace, project_dir: Path) -> Path:
    """Return the training root to exclude during product sort staging."""
    if getattr(args, "training_root", ""):
        return Path(args.training_root).expanduser().resolve()
    return project_dir / ".phase4_sort_no_training_exclusion"


def create_zip_temp_root() -> Path:
    """Create the temp folder used to hold extracted ZIP audio symlink targets."""
    temp_parent_raw = os.environ.get("AARON_SOUND_SORTER_TEMP_DIR", "").strip()
    temp_parent = Path(temp_parent_raw).expanduser().resolve() if temp_parent_raw else None
    if temp_parent:
        temp_parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix="aaron_sound_sorter_zip_", dir=str(temp_parent) if temp_parent else None)
    ).resolve()


def stage_sort_input(input_path: Path, staged_root: Path, output_dir: Path) -> tuple[Path, bool, Optional[Path]]:
    """Stage one product sort input as symlinks and return the preview root."""
    if input_path.is_dir():
        linked_count = link_audio_tree(input_path, staged_root)
        if linked_count <= 0:
            raise SystemExit(f"No readable audio files found under folder input: {input_path}")
        return staged_root, False, None

    if input_path.is_file() and input_path.suffix.lower() in AUDIO_EXTS:
        target = staged_root / input_path.name
        with contextlib.suppress(FileNotFoundError):
            target.unlink()
        target.symlink_to(input_path.resolve())
        return staged_root, False, None

    if input_path.suffix.lower() == ".zip":
        zip_temp_root = create_zip_temp_root()
        extracted_count = extract_safe_audio_zip(input_path, zip_temp_root)
        if extracted_count <= 0:
            raise SystemExit(f"No readable audio files found inside ZIP input: {input_path}")
        linked_count = link_audio_tree(zip_temp_root, staged_root)
        if linked_count <= 0:
            raise SystemExit(f"No staged audio files were linked from ZIP input: {input_path}")
        (output_dir / "ZIP_TEMP_CACHE_LOCATION.txt").write_text(str(zip_temp_root) + "\n", encoding="utf-8")
        return staged_root, True, zip_temp_root

    raise SystemExit(f"Expected audio file, ZIP, or folder input: {input_path}")


def zip_run(run_dir: Path) -> Path:
    upload_dir = run_dir / "upload"
    upload_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    zip_path = upload_dir / f"PHASE3_PURE_BRAIN_LAB_UPLOAD_BACK_{stamp}.zip"
    parent = run_dir.parent
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in run_dir.rglob("*"):
            if p == zip_path:
                continue
            if "upload" in p.relative_to(run_dir).parts:
                continue
            if p.name == ".DS_Store" or "__MACOSX" in p.parts:
                continue
            rel_parts = p.relative_to(run_dir).parts
            include_sorted_audio = os.environ.get("INCLUDE_SORTED_AUDIO_IN_ZIP", "0") == "1"
            if (
                (not include_sorted_audio)
                and rel_parts
                and rel_parts[0]
                in {
                    "sorted_preview",
                    "training_preview",
                    "training_curated_keep_preview",
                    "training_curated_reject_preview",
                    "real_sorted_preview",
                    "real_raw_brain_prediction_preview",
                    "real_broad_family_review_preview",
                    "real_physical_role_review_preview",
                    "listen_review_pack",
                    "gold_training_candidate_workspace",
                }
            ):
                # Keep upload small. The local preview folders stay visible on Aaron's Mac.
                continue
            # Include manifests, reports, logs, and brain by default.
            z.write(p, p.relative_to(parent))
    if os.environ.get("PHASE3_REVEAL_UPLOAD_ZIP", "1") != "0":
        with contextlib.suppress(FileNotFoundError, OSError):
            subprocess.run(
                ["open", "-R", str(zip_path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    return zip_path


def copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst, symlinks=True)


def write_phase4_summary(
    output_dir: Path, manifest_rows: List[Dict[str, str]], brain_path: Path, training_note: str = ""
) -> None:
    counts = Counter(str(r.get("final_top", "") or "_TO_REVIEW") for r in manifest_rows)
    review_reasons = Counter(
        str(r.get("review_reason", "") or "") for r in manifest_rows if str(r.get("final_top", "")) == "_TO_REVIEW"
    )
    lines = [
        "Aaron Sound Sorter Phase 4 Folder-Brain Summary",
        "",
        f"Brain: {brain_path}",
    ]
    if training_note:
        lines.append(f"Training mode: {training_note}")
    lines.extend(
        [
            f"Total files processed: {len(manifest_rows)}",
            "",
            "Top folder counts:",
        ]
    )
    for name, count in sorted(counts.items()):
        lines.append(f"  {name}: {count}")
    if review_reasons:
        lines.extend(["", "Most common review reasons:"])
        for reason, count in review_reasons.most_common(20):
            lines.append(f"  {count}: {reason}")
    (output_dir / "Aaron_Sorted_Sounds_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def zip_phase4_output(output_dir: Path) -> Path:
    root = output_dir / "Aaron_Sorted_Sounds"
    zip_path = output_dir / "Aaron_Sorted_Sounds.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if root.exists():
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    # Do not follow symlinked audio into the ZIP unless explicitly requested.
                    # This keeps report/test zips small and prevents accidental disk/network blowups.
                    if p.is_symlink() and os.environ.get("AARON_SOUND_SORTER_ZIP_SYMLINK_AUDIO", "0") != "1":
                        continue
                    z.write(p, p.relative_to(output_dir))
        for name in ["Aaron_Sorted_Sounds_manifest.csv", "Aaron_Sorted_Sounds_summary.txt"]:
            p = output_dir / name
            if p.exists():
                z.write(p, p.name)
    return zip_path


def run_train_brain_command(args: argparse.Namespace) -> int:
    """Train and save a folder-supervised brain from a folder tree.

    Mode 1: trusted-tree. Aaron's curated training folder tree is truth.
    Mode 2: approved-sort. A reviewed Aaron_Sorted_Sounds output is truth;
    _TO_REVIEW is skipped by the scanner.
    """
    training_root = Path(args.training_root).expanduser().resolve()
    if not training_root.exists():
        raise SystemExit(f"Training root not found: {training_root}")
    save_brain = Path(args.save_brain or DEFAULT_FOLDER_BRAIN).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else project_dir / "stage4_folder_brain_training" / time.strftime("run_%Y%m%d_%H%M%S")
    )
    build_request = TrainBrainBuildRequest(
        root=str(training_root),
        project_dir=str(project_dir),
        report_dir="",
        run_dir=str(run_dir),
        allowed_top=str(args.allowed_top),
        max_eval_per_label=int(args.max_eval_per_label),
        max_eval_total=int(args.max_eval_total),
        max_centroids=int(args.max_centroids),
        min_similarity=float(args.min_similarity),
        min_margin=float(args.min_margin),
        training_preview_per_label=int(args.training_preview_per_label),
        random_seed=int(args.random_seed),
        min_active_train_per_label=int(args.min_active_train_per_label),
        real_preview_root=str(training_root),
        skip_expensive_training_diagnostics=not bool(getattr(args, "full_diagnostics", False)),
        dry_core_augment_policy=(
            "voice_sax"
            if bool(getattr(args, "augment_dry_core_voice_sax", False))
            else str(getattr(args, "dry_core_augment_policy", "") or "")
        ),
        dry_core_augment_timeout_sec=float(getattr(args, "dry_core_augment_timeout_sec", 45.0) or 45.0),
    )
    code = run_build_eval(build_request)
    built = run_dir / "brain" / "phase3_pure_scratch_brain.json"
    if code != 0 or not built.exists():
        print(f"Brain training failed or no brain was written: {built}")
        return int(code or 1)
    brain = read_json(built)
    brain["training_mode"] = str(args.mode)
    brain["training_root"] = str(training_root)
    brain["brain_type"] = "stage4_folder_supervised_brain"
    brain["folder_supervised_policy"] = (
        "Folder paths in the training tree are the labels. No fixed terminal category vocabulary is used."
    )
    print(f"Saving reusable folder-supervised brain: {save_brain}", flush=True)
    write_json(save_brain, brain)
    latest = runtime_pointer_path(project_dir, "phase4_latest_folder_brain_path.txt")
    latest.write_text(portable_project_path(project_dir, save_brain) + "\n", encoding="utf-8")
    print(f"Saved folder-supervised brain: {save_brain}", flush=True)
    print(f"Latest brain pointer: {latest}", flush=True)
    print(f"Training reports: {run_dir / 'reports'}", flush=True)
    starter_code = maybe_train_shape_memory_starter_for_cli(
        args=args,
        training_root=training_root,
        base_brain_path=save_brain,
        report_dir=run_dir / "reports" / "shape_memory_starter",
    )
    if starter_code != 0:
        return starter_code
    return 0


def _train_one_family_brain(
    *,
    training_root: Path,
    save_brain: Path,
    project_dir: Path,
    run_dir: Path,
    allowed_top: str,
    max_centroids: int,
    min_active_train_per_label: int,
    training_preview_per_label: int,
    random_seed: int,
    brain_family_role: str,
    dry_core_augment_policy: str = "",
    dry_core_augment_timeout_sec: float = 45.0,
) -> int:
    """Train one brain from one prepared training tree."""
    ns = argparse.Namespace(
        training_root=str(training_root),
        mode="trusted-tree",
        save_brain=str(save_brain),
        project_dir=str(project_dir),
        run_dir=str(run_dir),
        allowed_top=str(allowed_top),
        max_centroids=int(max_centroids),
        min_similarity=0.52,
        min_margin=1.00,
        random_seed=int(random_seed),
        max_eval_per_label=0,
        max_eval_total=0,
        training_preview_per_label=int(training_preview_per_label),
        min_active_train_per_label=int(min_active_train_per_label),
        full_diagnostics=False,
        dry_core_augment_policy=str(dry_core_augment_policy or ""),
        dry_core_augment_timeout_sec=float(dry_core_augment_timeout_sec or 45.0),
        augment_dry_core_voice_sax=bool(str(dry_core_augment_policy or "").strip()),
    )
    code = run_train_brain_command(ns)
    if code == 0 and save_brain.exists():
        try:
            brain = read_json(save_brain)
            brain["brain_family_role"] = brain_family_role
            brain["brain_family_policy"] = {
                "full": "all usable folder-truth data; stability lane",
                "core_baby": "central low-deviation anchors; precision/slam-dunk lane",
                "spread_baby": "diverse clean anchors; balanced recall lane",
                "outlier_baby": "plausible edge anchors; weird-but-possible recall only",
            }.get(brain_family_role, brain_family_role)
            write_json(save_brain, brain)
        except Exception as exc:
            print(f"WARNING: could not annotate brain_family_role on {save_brain}: {exc}", flush=True)
    return int(code)


def run_train_brain_family_command(args: argparse.Namespace) -> int:
    """Build full/core/spread/outlier brain files from one trusted folder tree.

    This is the multi-brain trainer. It first selects baby-brain anchor samples
    using measured fingerprints, then trains each brain with the existing
    folder-supervised trainer so the brain JSON schema stays compatible.
    """
    training_root = Path(args.training_root).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()
    stamp = time.strftime("run_%Y%m%d_%H%M%S")
    run_root = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else project_dir / "stage4_brain_family_training" / stamp
    )
    run_root.mkdir(parents=True, exist_ok=True)
    config = AnchorSelectionConfig(
        core_anchors=int(args.core_anchors),
        spread_anchors=int(args.spread_anchors),
        outlier_anchors=int(args.outlier_anchors),
        central_pool_fraction=float(args.central_pool_fraction),
        spread_pool_fraction=float(args.spread_pool_fraction),
        max_files_per_label_to_scan=int(args.max_files_per_label_to_scan),
        fingerprint_timeout_sec=float(args.fingerprint_timeout_sec),
    )
    dry_core_policy = (
        "voice_sax"
        if bool(getattr(args, "augment_dry_core_voice_sax", False))
        else str(getattr(args, "dry_core_augment_policy", "") or "")
    )
    dry_core_timeout = float(getattr(args, "dry_core_augment_timeout_sec", 45.0) or 45.0)
    if dry_core_policy:
        print(
            f"Dry-core training augmentation enabled: {dry_core_policy} (same folder-truth labels; audio files unchanged)",
            flush=True,
        )
    print("Building measured baby-brain anchor trees...", flush=True)
    trees = build_brain_family_training_trees(
        training_root=training_root,
        output_root=run_root / "selected_anchor_trees",
        config=config,
    )
    print(f"Anchor selection manifest: {trees.manifest_path}", flush=True)
    print(f"Anchor selection summary: {trees.summary_path}", flush=True)

    save_full = Path(args.save_full).expanduser().resolve()
    save_core = Path(args.save_core_baby).expanduser().resolve()
    save_spread = Path(args.save_spread_baby).expanduser().resolve()
    save_outlier = Path(args.save_outlier_baby).expanduser().resolve()

    if not bool(args.skip_full):
        print("\nTraining full brain from all usable training data...", flush=True)
        code = _train_one_family_brain(
            training_root=trees.full_root,
            save_brain=save_full,
            project_dir=project_dir,
            run_dir=run_root / "full_brain",
            allowed_top=str(args.allowed_top),
            max_centroids=int(args.full_max_centroids),
            min_active_train_per_label=int(args.min_active_train_per_label),
            training_preview_per_label=int(args.training_preview_per_label),
            random_seed=int(args.random_seed),
            brain_family_role="full",
            dry_core_augment_policy=dry_core_policy,
            dry_core_augment_timeout_sec=dry_core_timeout,
        )
        if code != 0:
            return code

    print("\nTraining core baby brain from central low-deviation anchors...", flush=True)
    code = _train_one_family_brain(
        training_root=trees.core_root,
        save_brain=save_core,
        project_dir=project_dir,
        run_dir=run_root / "core_baby_brain",
        allowed_top=str(args.allowed_top),
        max_centroids=int(args.baby_max_centroids),
        min_active_train_per_label=1,
        training_preview_per_label=int(args.core_anchors),
        random_seed=int(args.random_seed),
        brain_family_role="core_baby",
        dry_core_augment_policy=dry_core_policy,
        dry_core_augment_timeout_sec=dry_core_timeout,
    )
    if code != 0:
        return code

    print("\nTraining spread baby brain from diverse clean anchors...", flush=True)
    code = _train_one_family_brain(
        training_root=trees.spread_root,
        save_brain=save_spread,
        project_dir=project_dir,
        run_dir=run_root / "spread_baby_brain",
        allowed_top=str(args.allowed_top),
        max_centroids=int(args.baby_max_centroids),
        min_active_train_per_label=1,
        training_preview_per_label=int(args.spread_anchors),
        random_seed=int(args.random_seed),
        brain_family_role="spread_baby",
        dry_core_augment_policy=dry_core_policy,
        dry_core_augment_timeout_sec=dry_core_timeout,
    )
    if code != 0:
        return code

    print("\nTraining outlier baby brain from plausible edge anchors...", flush=True)
    code = _train_one_family_brain(
        training_root=trees.outlier_root,
        save_brain=save_outlier,
        project_dir=project_dir,
        run_dir=run_root / "outlier_baby_brain",
        allowed_top=str(args.allowed_top),
        max_centroids=int(args.baby_max_centroids),
        min_active_train_per_label=1,
        training_preview_per_label=int(args.outlier_anchors),
        random_seed=int(args.random_seed),
        brain_family_role="outlier_baby",
        dry_core_augment_policy=dry_core_policy,
        dry_core_augment_timeout_sec=dry_core_timeout,
    )
    if code != 0:
        return code

    latest = runtime_pointer_path(project_dir, "phase4_latest_brain_family_paths.txt")
    shape_starter_pointer = (
        str(shape_memory_starter_output_path(save_full, str(getattr(args, "save_shape_memory_brain", "") or "")))
        if bool(getattr(args, "train_shape_memory", True))
        else "disabled"
    )
    latest.write_text(
        "\n".join(
            [
                f"full={portable_project_path(project_dir, save_full)}",
                f"core_baby={portable_project_path(project_dir, save_core)}",
                f"spread_baby={portable_project_path(project_dir, save_spread)}",
                f"outlier_baby={portable_project_path(project_dir, save_outlier)}",
                (
                    "shape_starter_memory="
                    + (
                        portable_project_path(project_dir, Path(shape_starter_pointer))
                        if shape_starter_pointer != "disabled"
                        else "disabled"
                    )
                ),
                f"run_root={portable_project_path(project_dir, run_root)}",
                f"anchor_manifest={portable_project_path(project_dir, trees.manifest_path)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    starter_code = maybe_train_shape_memory_starter_for_cli(
        args=args,
        training_root=training_root,
        base_brain_path=save_full,
        report_dir=run_root / "shape_memory_starter",
    )
    if starter_code != 0:
        return starter_code
    print("\nDONE training brain family.", flush=True)
    print(f"Full brain: {save_full}", flush=True)
    print(f"Core baby brain: {save_core}", flush=True)
    print(f"Spread baby brain: {save_spread}", flush=True)
    print(f"Outlier baby brain: {save_outlier}", flush=True)
    print(f"ShapeVoter starter memory: {shape_starter_pointer}", flush=True)
    print(f"Reports: {run_root}", flush=True)
    print(f"Latest brain family pointer: {latest}", flush=True)
    return 0


def run_sort_command(args: argparse.Namespace) -> int:
    """Human-facing Phase 4 sort command using a saved folder brain.

    This does not rebuild the brain during sorting. It loads the brain produced
    by train-brain, then places new audio into learned training folders or review.
    """
    project_dir = Path(args.project_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    brain_path = resolve_brain_path(project_dir, str(args.brain or ""))
    if not brain_path.exists():
        raise SystemExit(
            f"No trained folder brain found: {brain_path}\n"
            "Build one first with: python3 Aaron_Sound_Sorter.py train-brain TRAINING_FOLDER --save-brain stage4_folder_brain.json"
        )
    brain = read_json(brain_path)
    validate_v052_brain_or_die(brain, brain_path)

    staged_root = output_dir / "_staged_input"
    if staged_root.exists() or staged_root.is_symlink():
        shutil.rmtree(staged_root, ignore_errors=True)
    staged_root.mkdir(parents=True, exist_ok=True)

    real_preview_root, input_is_zip, zip_temp_root = stage_sort_input(input_path, staged_root, output_dir)

    run_dir = output_dir / "stage4_sort_run"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    # In Phase 4 sort mode, never use the output folder as the training root.
    # The staged ZIP input lives under output_dir/_staged_input; using output_dir
    # as the training root made the sorter exclude every staged file as "training"
    # and produced empty manifests. Only exclude a training root when the caller
    # explicitly supplies one.
    training_root_for_exclusion = product_sort_training_exclusion_root(args, project_dir)
    sort_request = SortPreviewRequest(
        root=str(training_root_for_exclusion),
        real_preview_root=str(real_preview_root),
        min_similarity=float(args.min_similarity),
        min_margin=float(args.min_margin),
        random_seed=int(args.random_seed),
    )
    run_real_sort_preview(brain, sort_request, run_dir, reports_dir)

    final_root = output_dir / "Aaron_Sorted_Sounds"
    preview_root = run_dir / "real_sorted_preview"
    copytree_replace(preview_root, final_root)

    manifest_src = reports_dir / "real_sort_preview_manifest.csv"
    manifest_dst = output_dir / "Aaron_Sorted_Sounds_manifest.csv"
    manifest_rows: List[Dict[str, str]] = []
    if manifest_src.exists():
        manifest_rows = read_csv(manifest_src)

        # Phase 4 product output should show only one sorted tree: Aaron_Sorted_Sounds.
        # The real-preview tree is an internal staging artifact.  Rewrite manifest paths
        # so the user-facing manifest points at Aaron_Sorted_Sounds, then remove the
        # duplicate staging tree below.
        preview_prefix = str(preview_root.resolve())
        final_prefix = str(final_root.resolve())
        for row in manifest_rows:
            preview_path = str(row.get("preview_copy_path", "") or "")
            if preview_path:
                try:
                    resolved_preview = str(Path(preview_path).resolve())
                except Exception:
                    resolved_preview = preview_path
                if resolved_preview == preview_prefix or resolved_preview.startswith(preview_prefix + os.sep):
                    rel_path = os.path.relpath(resolved_preview, preview_prefix)
                    row["preview_copy_path"] = str(Path(final_prefix) / rel_path)
        fields = (
            list(manifest_rows[0].keys())
            if manifest_rows
            else ["source_path", "final_top", "final_label", "confidence_status", "review_reason"]
        )
        write_csv(manifest_dst, manifest_rows, fields)
    else:
        write_csv(manifest_dst, [], ["source_path", "final_top", "final_label", "confidence_status", "review_reason"])

    # Remove the internal duplicate sorted tree from normal Phase 4 sort output.
    # Keep reports under stage4_sort_run/reports, but do not leave both
    # stage4_sort_run/real_sorted_preview and Aaron_Sorted_Sounds for users to compare.
    if preview_root.exists() or preview_root.is_symlink():
        shutil.rmtree(preview_root, ignore_errors=True)

    write_phase4_summary(
        output_dir, manifest_rows, brain_path, training_note=str(brain.get("training_mode", "folder-supervised"))
    )
    if bool(getattr(args, "make_zip", True)):
        zip_phase4_output(output_dir)
    # _staged_input is intentionally symlinks only.  For ZIP input the symlink
    # targets live in a Python-created temp folder outside the project/output tree;
    # keep that temp cache while Aaron inspects the run so sorted symlinks stay valid.
    if input_is_zip:
        print(f"ZIP temp cache kept for symlink targets: {zip_temp_root}")
        print(f"Temp cache note: {output_dir / 'ZIP_TEMP_CACHE_LOCATION.txt'}")
    else:
        print(f"Staged folder input uses symlinks only: {staged_root}")
    print(f"Sorted output: {final_root}")
    print(f"Manifest: {manifest_dst}")
    print(f"Summary: {output_dir / 'Aaron_Sorted_Sounds_summary.txt'}")
    return 0


# CLI argument parsing lives in Aaron_Sound_Sorter.py.
# This module intentionally exposes command handler functions only.


def normalize_cli_args(argv: Optional[List[str]] = None) -> List[str]:
    # Human shortcut: python3 Aaron_Sound_Sorter.py input.zip output_folder
    # Developer form remains: python3 Aaron_Sound_Sorter.py build-eval ...
    cli_args = list(sys.argv[1:] if argv is None else argv)
    if (
        len(cli_args) >= 2
        and not str(cli_args[0]).startswith("-")
        and cli_args[0] not in {"build-eval", "self-test", "sort", "train-brain", "train-brain-family"}
    ):
        cli_args.insert(0, "sort")
    return cli_args


def add_shape_memory_training_args(parser: argparse.ArgumentParser) -> None:
    """Add shared ShapeVoter starter-memory training flags.

    Args:
        parser: Training subparser to extend.

    Returns:
        None.

    Side Effects:
        Mutates ``parser`` by registering CLI arguments.
    """
    parser.add_argument(
        "--train-shape-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Also train the low-trust ShapeVoter starter memory brain from the trusted training tree. Default: on."),
    )
    parser.add_argument(
        "--save-shape-memory-brain",
        default="",
        help=f"Starter ShapeVoter memory JSON to write. Default: sibling {SHAPE_STARTER_MEMORY_BRAIN_NAME}.",
    )
    parser.add_argument(
        "--shape-memory-train-per-label",
        type=int,
        default=DEFAULT_STARTER_SHAPE_MEMORY_EXAMPLES_PER_LABEL,
        help="Low-trust starter ShapeVoter examples per label. Default: 3.",
    )
    parser.add_argument(
        "--shape-memory-evidence-weight",
        type=int,
        default=DEFAULT_STARTER_SHAPE_MEMORY_WEIGHT,
        help="Low-trust support weight per starter shape example. GUI corrections remain stronger.",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Aaron Sound Sorter Stage 4")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sortp = sub.add_parser("sort", help="Sort a ZIP or folder using a saved folder-supervised Stage 4 brain.")
    sortp.add_argument("input")
    sortp.add_argument("output")
    sortp.add_argument(
        "--brain",
        default="",
        help=(
            "Saved folder-supervised brain JSON. Defaults to stage4_folder_brain.json "
            "or config/runtime/phase4_latest_folder_brain_path.txt."
        ),
    )
    sortp.add_argument(
        "--training-root",
        default="",
        help="Optional training root to exclude from diagnostics. Sorting does not rebuild from it.",
    )
    sortp.add_argument(
        "--make-zip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write Aaron_Sorted_Sounds.zip. Default: on.",
    )
    sortp.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    sortp.add_argument("--allowed-top", default="Drums,Instruments,FX,Textures")
    sortp.add_argument("--max-centroids", type=int, default=7)
    sortp.add_argument("--min-similarity", type=float, default=0.52)
    sortp.add_argument("--min-margin", type=float, default=1.00)
    sortp.add_argument("--random-seed", type=int, default=20260503)
    sortp.set_defaults(func=run_sort_command)

    trainp = sub.add_parser("train-brain", help="Train a saved folder-supervised brain from folder paths.")
    trainp.add_argument("training_root", help="Trusted training folder tree or approved Aaron_Sorted_Sounds folder.")
    trainp.add_argument(
        "--mode",
        choices=["trusted-tree", "approved-sort"],
        default="trusted-tree",
        help="trusted-tree builds from Aaron's curated training tree; approved-sort learns an approved sorted output and skips _TO_REVIEW.",
    )
    trainp.add_argument(
        "--save-brain", default=DEFAULT_FOLDER_BRAIN, help="Brain JSON to write. Default: stage4_folder_brain.json"
    )
    trainp.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    trainp.add_argument("--run-dir", default="")
    trainp.add_argument("--allowed-top", default="Drums,Instruments,FX,Textures")
    trainp.add_argument("--max-centroids", type=int, default=7)
    trainp.add_argument("--min-similarity", type=float, default=0.52)
    trainp.add_argument("--min-margin", type=float, default=1.00)
    trainp.add_argument("--random-seed", type=int, default=20260503)
    trainp.add_argument("--max-eval-per-label", type=int, default=0)
    trainp.add_argument("--max-eval-total", type=int, default=0)
    trainp.add_argument("--training-preview-per-label", type=int, default=0)
    trainp.add_argument("--min-active-train-per-label", type=int, default=1)
    trainp.add_argument(
        "--full-diagnostics",
        action="store_true",
        help="Also compute slow developer-only O(rows x labels) training diagnostics. Off by default for product brain builds.",
    )
    trainp.add_argument(
        "--augment-dry-core-voice-sax",
        action="store_true",
        help="Experimental: add harmonic-core dry-view training rows only for trusted Voice/Vocal and Saxophone training folders. Default off.",
    )
    trainp.add_argument(
        "--dry-core-augment-timeout-sec",
        type=float,
        default=45.0,
        help="Per-file timeout for experimental dry-core training augmentation.",
    )
    add_shape_memory_training_args(trainp)
    trainp.set_defaults(func=run_train_brain_command)

    familyp = sub.add_parser(
        "train-brain-family", help="Train full, core baby, spread baby, and outlier baby brains from one trusted tree."
    )
    familyp.add_argument("training_root", help="Trusted training folder tree.")
    familyp.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    familyp.add_argument("--run-dir", default="")
    familyp.add_argument("--allowed-top", default="Drums,Instruments,FX,Textures")
    familyp.add_argument("--save-full", default=DEFAULT_FOLDER_BRAIN)
    familyp.add_argument("--save-core-baby", default="stage4_folder_brain_core_baby.json")
    familyp.add_argument("--save-spread-baby", default="stage4_folder_brain_spread_baby.json")
    familyp.add_argument("--save-outlier-baby", default="stage4_folder_brain_outlier_baby.json")
    familyp.add_argument(
        "--skip-full", action="store_true", help="Only rebuild the baby brains; keep the existing full brain."
    )
    familyp.add_argument("--core-anchors", type=int, default=3, help="Central low-deviation anchors per label.")
    familyp.add_argument("--spread-anchors", type=int, default=3, help="Diverse clean anchors per label.")
    familyp.add_argument("--outlier-anchors", type=int, default=3, help="Plausible edge anchors per label.")
    familyp.add_argument(
        "--central-pool-fraction",
        type=float,
        default=0.70,
        help="Outlier lane starts near this quantile; core always uses closest samples.",
    )
    familyp.add_argument(
        "--spread-pool-fraction",
        type=float,
        default=0.85,
        help="Spread lane diversity is chosen from this clean central fraction.",
    )
    familyp.add_argument(
        "--max-files-per-label-to-scan",
        type=int,
        default=0,
        help="0 means scan every direct audio file in each label folder.",
    )
    familyp.add_argument("--fingerprint-timeout-sec", type=float, default=45.0)
    familyp.add_argument(
        "--augment-dry-core-voice-sax",
        action="store_true",
        help="Experimental: train Voice/Vocal and Saxophone labels with both full-audio and dry-core harmonic views. Default off.",
    )
    familyp.add_argument(
        "--dry-core-augment-timeout-sec",
        type=float,
        default=45.0,
        help="Per-file timeout for experimental dry-core training augmentation.",
    )
    familyp.add_argument("--full-max-centroids", type=int, default=6)
    familyp.add_argument("--baby-max-centroids", type=int, default=3)
    familyp.add_argument("--training-preview-per-label", type=int, default=3)
    familyp.add_argument("--min-active-train-per-label", type=int, default=1)
    familyp.add_argument("--random-seed", type=int, default=20260513)
    add_shape_memory_training_args(familyp)
    familyp.set_defaults(func=run_train_brain_family_command)

    run = sub.add_parser("build-eval", help="Build a pure scratch brain and evaluate it.")
    run.add_argument("--root", default=DEFAULT_ROOT)
    run.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    run.add_argument("--report-dir", default="")
    run.add_argument("--run-dir", default="")
    run.add_argument("--allowed-top", default="Drums,Instruments,FX,Textures")
    run.add_argument(
        "--source-mode",
        choices=["tree", "manifest"],
        default="tree",
        help="tree scans the live trusted folder tree and preserves all current folders. manifest uses the latest candidate_clean_training_set.csv for comparison/debug only.",
    )
    run.add_argument(
        "--supplemental-training-root",
        action="append",
        default=[],
        help="Extra training tree to merge into tree-mode builds. May be repeated.",
    )
    run.add_argument(
        "--no-supplemental-training",
        action="store_true",
        help="Disable any supplemental training roots supplied with --supplemental-training-root.",
    )
    run.add_argument(
        "--max-train-per-group",
        type=int,
        default=0,
        help="Max candidate rows selected per label before curation. 0 means use every prepared clean row.",
    )
    run.add_argument("--max-eval-per-label", type=int, default=3)
    run.add_argument(
        "--max-eval-total", type=int, default=0, help="0 means no total cap, so every label can get eval coverage."
    )
    run.add_argument("--max-centroids", type=int, default=7)
    run.add_argument(
        "--min-similarity",
        type=float,
        default=0.52,
        help="Strict auto-place gate. Lower means more placements, more risk.",
    )
    run.add_argument(
        "--min-margin", type=float, default=1.00, help="Strict auto-place gate. Lower means more placements, more risk."
    )
    run.add_argument(
        "--weak-similarity-floor",
        type=float,
        default=0.20,
        help="Lowest similarity that can pass only when margin is very strong.",
    )
    run.add_argument(
        "--strong-margin",
        type=float,
        default=1.50,
        help="Margin needed to auto-place a low-similarity but clear-winner prediction.",
    )
    run.add_argument(
        "--allow-low-similarity-clear-margin",
        action="store_true",
        help="Legacy/experimental: allow low-similarity predictions to auto-place when margin is very strong. Off by default because Phase 3 should prefer review over overconfident guesses.",
    )
    run.add_argument(
        "--training-preview-per-label",
        type=int,
        default=6,
        help="Copy this many raw training examples per internal label into training_preview.",
    )
    run.add_argument(
        "--random-seed",
        type=int,
        default=20260503,
        help="Deterministic random seed for candidate sampling and curated keep choice.",
    )
    run.add_argument("--curate-min-group-size", type=int, default=1)
    run.add_argument("--curate-max-keep-per-label", type=int, default=100)
    run.add_argument("--curate-fx-max-keep-per-label", type=int, default=100)
    run.add_argument(
        "--curate-fx-central-pool-fraction",
        type=float,
        default=1.0,
        help="FX curation is loose: discard only extreme outlier tail before random keep.",
    )
    run.add_argument("--curate-max-reject-fraction", type=float, default=1.00)
    run.add_argument(
        "--curate-central-pool-fraction",
        type=float,
        default=1.0,
        help="Keep examples from this central fraction before random curated selection. 0.70 is intentionally not too strict.",
    )
    run.add_argument("--curated-preview-per-label", type=int, default=6)
    run.add_argument(
        "--source-cap-per-label",
        type=int,
        default=DEFAULT_SOURCE_CAP_PER_LABEL,
        help="Max kept training examples per source pack inside each label.",
    )
    run.add_argument(
        "--min-active-train-per-label",
        type=int,
        default=MIN_ACTIVE_TRAIN_PER_LABEL,
        help="Labels with fewer clean examples stay discovery-only and do not train.",
    )
    run.add_argument(
        "--min-source-groups-per-label",
        type=int,
        default=DEFAULT_MIN_SOURCE_GROUPS_PER_LABEL,
        help="Labels with too few source packs stay discovery-only.",
    )
    run.add_argument(
        "--no-gold-workspace", action="store_true", help="Do not copy the full gold candidate listening workspace."
    )
    run.add_argument(
        "--copy-listen-pack",
        action="store_true",
        help="Copy up to 2 eval samples per label into a small listening pack.",
    )
    run.add_argument(
        "--real-preview-root",
        default=os.environ.get("AARON_SAMPLE_LIBRARY_ROOT", ""),
        help="Blind real-library preview root. Defaults to AARON_SAMPLE_LIBRARY_ROOT.",
    )
    run.add_argument(
        "--real-preview-max-files",
        type=int,
        default=1000,
        help="Max blind real-library files to sort after building the brain.",
    )
    run.add_argument(
        "--real-preview-per-folder",
        type=int,
        default=25,
        help="Max blind real-library preview files per source folder.",
    )
    run.add_argument(
        "--real-preview-official-copy-per-label",
        type=int,
        default=DEFAULT_REAL_PREVIEW_OFFICIAL_COPY_PER_LABEL,
        help="Max audio files copied per official gated preview label. Reports still include every scanned file. Use -1 for unlimited.",
    )
    run.add_argument(
        "--real-preview-force-guess",
        action="store_true",
        help="Create real_forced_guess_preview: every readable real-preview file is symlinked to the brain's raw best label, with no review/gate escape hatch.",
    )
    run.add_argument(
        "--real-preview-force-guess-copy-per-label",
        type=int,
        default=DEFAULT_REAL_PREVIEW_FORCE_GUESS_COPY_PER_LABEL,
        help="Max forced-guess audio links per raw label. Use -1 for unlimited. Only used with --real-preview-force-guess.",
    )
    run.add_argument(
        "--real-preview-raw-copy-per-label",
        type=int,
        default=DEFAULT_REAL_PREVIEW_RAW_COPY_PER_LABEL,
        help="Max audio files copied per raw brain prediction label. Reports still include every scanned file. Use -1 for unlimited.",
    )
    run.add_argument(
        "--real-preview-risky-copy-per-label",
        type=int,
        default=DEFAULT_REAL_PREVIEW_RISKY_COPY_PER_LABEL,
        help="Max audio files copied per risky sorted preview label. This is a workbench, not official output. Use -1 for unlimited.",
    )
    run.add_argument(
        "--real-preview-risky-min-similarity",
        type=float,
        default=DEFAULT_REAL_PREVIEW_RISKY_MIN_SIMILARITY,
        help="Minimum raw similarity for reviewed files to enter the risky sorted preview tree.",
    )
    run.add_argument(
        "--real-preview-broad-family-min-gap",
        type=float,
        default=DEFAULT_REAL_PREVIEW_BROAD_FAMILY_MIN_GAP,
        help="Review-only broad family grouping threshold. Defaults to 0 so reviewed files still get a broad-family listening queue; this never auto-places.",
    )
    run.add_argument(
        "--real-preview-broad-family-conflict-override-gap",
        type=float,
        default=DEFAULT_BROAD_FAMILY_CONFLICT_OVERRIDE_GAP,
        help="When raw exact-label top and learned broad top disagree, use learned broad top only at/above this gap. Otherwise review-queue under raw top.",
    )
    run.add_argument(
        "--real-preview-broad-copy-per-label",
        type=int,
        default=DEFAULT_REAL_PREVIEW_BROAD_COPY_PER_LABEL,
        help="Max reviewed candidate audio copied per broad-family/predicted-label group. Reports still include every scanned file. Use -1 for unlimited.",
    )
    run.add_argument(
        "--real-preview-physical-role-copy-per-label",
        type=int,
        default=DEFAULT_REAL_PREVIEW_BROAD_COPY_PER_LABEL,
        help="Max reviewed candidate audio copied per physics/role recommendation group. Reports still include every scanned file. Use -1 for unlimited.",
    )
    run.add_argument("--no-real-sort-preview", action="store_true", help="Disable blind real-library preview sorting.")
    run.add_argument(
        "--suppress-upload-zip",
        action="store_true",
        help="Do not create the old large Phase 3 lab upload ZIP. Product runners create logs-only ZIPs separately.",
    )
    run.add_argument(
        "--skip-expensive-training-diagnostics",
        action="store_true",
        help="Skip slow developer-only cross-boundary and recall diagnostics for faster smoke tests.",
    )
    run.add_argument(
        "--augment-dry-core-voice-sax",
        action="store_true",
        help="Experimental: add harmonic-core dry-view training rows only for trusted Voice/Vocal and Saxophone training folders. Default off.",
    )
    run.add_argument(
        "--dry-core-augment-timeout-sec",
        type=float,
        default=45.0,
        help="Per-file timeout for experimental dry-core training augmentation.",
    )
    run.set_defaults(func=run_build_eval)

    selftest = sub.add_parser(
        "self-test", help="Run targeted unit tests for file filtering, fingerprints, structure head, and path safety."
    )
    selftest.add_argument("--tmp-root", default="")
    selftest.set_defaults(func=lambda args: run_self_tests(Path(args.tmp_root) if args.tmp_root else None))

    return ap
