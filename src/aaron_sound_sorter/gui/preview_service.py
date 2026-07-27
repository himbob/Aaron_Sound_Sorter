"""GUI-facing preview, export, and correction-evidence services."""

from __future__ import annotations

import csv
import filecmp
import json
import os
import shutil
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from aaron_audio_intelligence.physics_memory_brain import PHYSICS_MEMORY_BRAIN_NAME
from aaron_audio_intelligence.shape_memory_brain import SHAPE_MEMORY_BRAIN_NAME, SHAPE_STARTER_MEMORY_BRAIN_NAME
from aaron_audio_intelligence.user_memory_brain import USER_MEMORY_BRAIN_NAME
from aaron_audio_intelligence.voter_memory_brain import VOTER_MEMORY_BRAIN_NAME
from aaron_sound_sorter.domain.models import SortFileResult, SortRequest
from aaron_sound_sorter.domain.policies import BrainVoterPolicy, ConsensusPolicy, PhysicsVoterPolicy, ShapeVoterPolicy
from aaron_sound_sorter.engine.consensus import ConsensusRunner
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.placement_resolver import PlacementResolver
from aaron_sound_sorter.engine.sorter import SortSamplesUseCase, voter_result_digest
from aaron_sound_sorter.gui.calibration_feedback import (
    calibration_feedback_from_preview_row,
    record_approved_session_feedback,
)
from aaron_sound_sorter.gui.models import (
    ExportMode,
    ExportSummary,
    PreviewRow,
    SortPreviewSession,
    TrainingImportSummary,
)
from aaron_sound_sorter.infrastructure.audio_repository import AudioInputRepository, InputPreparationCancelled
from aaron_sound_sorter.infrastructure.brain_repository import BrainRepository
from aaron_sound_sorter.infrastructure.report_writer import (
    compact_top_guess,
    disable_macos_metadata_sidecars,
    safe_folder_path,
    unique_path,
)
from aaron_sound_sorter.neural_audio.authority import (
    apply_neural_prediction_to_result,
    has_decisive_loop_structure,
    load_optional_confidence_calibration,
    neural_conflict_families,
    neural_defers_to_exact_human_teacher,
    neural_disagreement_requires_review,
    neural_panns_contradicts,
    neural_semantic_compatibility,
    neural_structure_conflicts,
)
from aaron_sound_sorter.neural_audio.authority_promotion import enabled_authority_groups
from aaron_sound_sorter.neural_audio.calibration import ConfidenceCalibrationBundle, ReviewFeedbackStore
from aaron_sound_sorter.neural_audio.gui_training import NeuralTrainingInbox
from aaron_sound_sorter.neural_audio.runtime import (
    NeuralRuntimeBatch,
    NeuralRuntimePrediction,
    run_configured_neural_predictions,
)
from aaron_sound_sorter.taxonomy_contracts import (
    STRUCTURE_TERMINALS,
    TOP_LEVEL_TAXONOMY_FAMILIES,
    canonicalize_taxonomy_label,
    is_valid_taxonomy_contract_label,
    normalize_taxonomy_path,
    taxonomy_label_contract,
)
from aaron_sound_sorter.taxonomy_registry import TaxonomyRegistry
from aaron_sound_sorter.voters.base import Voter
from aaron_sound_sorter.voters.brain_recall import BalancedRecallBrainVoter, FullBrainVoter
from aaron_sound_sorter.voters.physics_voter import PhysicsVoter
from aaron_sound_sorter.voters.shape_voter import ShapeVoter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BRAIN = "stage4_folder_brain.json"
DEFAULT_GUI_BRAIN_CONFIG = Path("config/gui_brains.yaml")
DEFAULT_GUI_TAXONOMY_CATALOG = Path("config/gui_taxonomy_catalog.json")
DEFAULT_CANONICAL_TAXONOMY = Path("config/canonical_taxonomy.json")
DEFAULT_TAXONOMY_ALIASES = Path("config/taxonomy_aliases.json")
DEFAULT_MASTER_TAXONOMY_LEDGER = Path("Aaron_Master_Attribute_Classification_Ledger_v1_0.json")
DEFAULT_TRAINING_TAXONOMY_ROOT = Path("training/locked_curated_v1")
SORTED_ROOT_NAME = "Aaron_Sorted_Sounds"
PreviewProgressCallback = Callable[[int, int, str], None]
PreviewRowCallback = Callable[[PreviewRow], None]
CancelRequestedCallback = Callable[[], bool]


class PreviewCancelled(Exception):
    """Raised when a GUI preview run is cancelled by the user."""


REVIEW_LABELS = [
    "_TO_REVIEW/Needs Human Review",
    "_TO_REVIEW/Measured Role Conflict",
    "_TO_REVIEW/No Strong Voter Consensus",
]
TRAINING_STRUCTURE_FOLDER_NAMES = {
    "_ONE_SHOTS": "One Shots",
    "_LOOPS": "Loops",
    "_LONG_FX": "Long FX",
    "_LONG_RUNNING": "Long Running",
}
AUDIO_LABEL_SUFFIXES = {".wav", ".wave", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".ogg"}


@dataclass(frozen=True)
class BrainFamilyConfig:
    """Brain-lane configuration used by GUI preview sorting.

    Args:
        config_path: YAML file that declared the brain family.
        full_brain_path: Main stage-four brain JSON path.
        baby_brain_path: Legacy baby brain path, when configured.
        core_baby_brain_path: Clean-center baby brain path, when configured.
        spread_baby_brain_path: Diversity baby brain path, when configured.
        outlier_baby_brain_path: Outlier recall baby brain path, when
            configured.
        user_memory_brain_path: GUI correction memory brain path, when
            configured.
        physics_memory_brain_path: GUI correction PhysicsVoter memory brain
            path, when configured.
        voter_memory_brain_path: GUI correction low-level voter memory brain
            path, when configured.
        shape_starter_memory_brain_path: Low-trust batch starter ShapeVoter
            memory brain path, when configured.
        shape_memory_brain_path: ShapeVoter correction memory brain path, when
            configured.
        harmonic_core_baby_brain_path: Harmonic-core baby brain path, when
            configured.
        harmonic_spread_baby_brain_path: Harmonic-spread baby brain path, when
            configured.
        harmonic_outlier_baby_brain_path: Harmonic-outlier baby brain path,
            when configured.
        use_baby_brains_in_sort: Whether normal baby lanes participate.
        use_harmonic_brains_in_sort: Whether harmonic baby lanes participate.

    Side Effects:
        None.

    Important Constraints:
        These paths configure trained reference data only. They must not be
        interpreted as evidence about an input sample.
    """

    config_path: Path
    full_brain_path: Path
    baby_brain_path: Path | None
    core_baby_brain_path: Path | None
    spread_baby_brain_path: Path | None
    outlier_baby_brain_path: Path | None
    user_memory_brain_path: Path | None
    physics_memory_brain_path: Path | None
    voter_memory_brain_path: Path | None
    shape_starter_memory_brain_path: Path | None
    shape_memory_brain_path: Path | None
    harmonic_core_baby_brain_path: Path | None
    harmonic_spread_baby_brain_path: Path | None
    harmonic_outlier_baby_brain_path: Path | None
    use_baby_brains_in_sort: bool
    use_harmonic_brains_in_sort: bool

    def display_summary(self) -> str:
        """Return a compact human-facing summary for the GUI."""
        normal_lanes = [
            self.baby_brain_path,
            self.core_baby_brain_path,
            self.spread_baby_brain_path,
            self.outlier_baby_brain_path,
            self.user_memory_brain_path,
            self.physics_memory_brain_path,
            self.voter_memory_brain_path,
            self.shape_starter_memory_brain_path,
            self.shape_memory_brain_path,
        ]
        harmonic_lanes = [
            self.harmonic_core_baby_brain_path,
            self.harmonic_spread_baby_brain_path,
            self.harmonic_outlier_baby_brain_path,
        ]
        configured_normal = sum(1 for path in normal_lanes if path is not None)
        configured_harmonic = sum(1 for path in harmonic_lanes if path is not None)
        normal_state = "on" if self.use_baby_brains_in_sort else "diagnostic/off"
        harmonic_state = "on" if self.use_harmonic_brains_in_sort else "diagnostic/off"
        return (
            f"full={self.full_brain_path.name}; "
            f"baby={configured_normal} lanes ({normal_state}); "
            f"harmonic={configured_harmonic} lanes ({harmonic_state})"
        )


class SortPreviewService:
    """Classify user-selected audio into editable GUI preview rows.

    Args:
        project_root: Repository root. Defaults to the installed project root.

    Side Effects:
        Creates a preview run folder under ``_reports/gui_preview`` and may
        extract ZIP inputs into that run folder. It does not place final sorted
        files, train brains, or alter classifier behavior.

    Important Constraints:
        This service does not use filenames or source folders as classification
        evidence. Paths are used only for input preparation, display, and export.
    """

    def __init__(self, project_root: Path | None = None, brain_config_path: Path | None = None) -> None:
        self.project_root = Path(project_root or PROJECT_ROOT).expanduser().resolve()
        self.brain_config_path = self.resolve_brain_config_path(brain_config_path)

    def classify_input(
        self,
        input_path: Path,
        *,
        brain_path: Path | None = None,
        candidate_count: int = 100,
        sort_workers: int = 1,
        progress_callback: PreviewProgressCallback | None = None,
        row_callback: PreviewRowCallback | None = None,
        cancel_requested: CancelRequestedCallback | None = None,
    ) -> SortPreviewSession:
        """Return an editable preview session for the selected input.

        Args:
            input_path: Folder, ZIP, or audio file selected by the user.
            brain_path: Optional active brain path. Relative paths resolve
                against the project root.
            candidate_count: Number of candidates requested from each voter.
            sort_workers: Per-file classification worker count.
            progress_callback: Optional callback receiving completed file count,
                total file count, and the latest display file name.
            row_callback: Optional callback receiving each finalized preview row.
            Rows are not published while neural evidence is still pending.
            cancel_requested: Optional callback returning true when the browser
                has requested cooperative cancellation.

        Returns:
            A preview session containing one row per classified audio file.

        Raises:
            ValueError: If input preparation finds no supported audio files.
            FileNotFoundError: If the selected brain does not exist.
            PreviewCancelled: If the browser cancels the preview run.
        """
        brain_config = self.load_brain_family_config(full_brain_override=brain_path)
        if not brain_config.full_brain_path.exists():
            raise FileNotFoundError(f"Brain file not found: {brain_config.full_brain_path}")
        run_dir = self.new_preview_run_dir()
        workspace_dir = run_dir / "preview_workspace"
        request = SortRequest(
            input_path=Path(input_path).expanduser(),
            output_dir=workspace_dir,
            brain_path=brain_config.full_brain_path,
            baby_brain_path=brain_config.baby_brain_path,
            core_baby_brain_path=brain_config.core_baby_brain_path,
            spread_baby_brain_path=brain_config.spread_baby_brain_path,
            outlier_baby_brain_path=brain_config.outlier_baby_brain_path,
            user_memory_brain_path=brain_config.user_memory_brain_path,
            physics_memory_brain_path=brain_config.physics_memory_brain_path,
            voter_memory_brain_path=brain_config.voter_memory_brain_path,
            shape_starter_memory_brain_path=brain_config.shape_starter_memory_brain_path,
            shape_memory_brain_path=brain_config.shape_memory_brain_path,
            harmonic_core_baby_brain_path=brain_config.harmonic_core_baby_brain_path,
            harmonic_spread_baby_brain_path=brain_config.harmonic_spread_baby_brain_path,
            harmonic_outlier_baby_brain_path=brain_config.harmonic_outlier_baby_brain_path,
            use_baby_brains_in_sort=brain_config.use_baby_brains_in_sort,
            use_harmonic_brains_in_sort=brain_config.use_harmonic_brains_in_sort,
            write_zip=False,
            candidate_count=int(candidate_count),
            sort_workers=max(1, int(sort_workers)),
            use_persistent_analysis_cache=gui_analysis_cache_enabled(),
            analysis_cache_dir=gui_analysis_cache_dir(self.project_root),
        )
        try:
            _raise_if_preview_cancelled(cancel_requested)
            sorter = build_product_sorter(candidate_count=request.candidate_count)
            sorter.configure_analysis_cache(request)
            sorter.analysis_cache.reset()
            brain = sorter.brain_repository.load(request.brain_path)
            baby_brains = sorter.load_baby_brains_if_available(request)
            harmonic_baby_brains = sorter.load_harmonic_baby_brains_if_available(request)
            sorter.attach_memory_brains(brain, baby_brains)
            prepared_input = sorter.audio_repository.prepare(
                request.input_path,
                request.output_dir,
                cancel_requested=cancel_requested,
            )
            total_files = len(prepared_input.audio_files)
            if progress_callback is not None:
                progress_callback(0, total_files, "Prepared audio input")
            _raise_if_preview_cancelled(cancel_requested)
            results = classify_audio_files_with_progress(
                sorter,
                prepared_input.audio_files,
                brain,
                baby_brains,
                harmonic_baby_brains,
                use_baby_brains_in_sort=request.use_baby_brains_in_sort,
                use_harmonic_brains_in_sort=request.use_harmonic_brains_in_sort,
                max_workers=request.sort_workers,
                progress_callback=progress_callback,
                # Do not publish the base-sorter candidate. A GUI row is only
                # complete after memory, CLAP, PANNs, and neural authority have
                # finished the final category decision.
                row_callback=None,
                cancel_requested=cancel_requested,
            )
            _raise_if_preview_cancelled(cancel_requested)
            labels = load_available_labels(brain_config.full_brain_path, project_root=self.project_root)
            rows = [preview_row_from_result(index, result) for index, result in enumerate(results, start=1)]
            neural_batch = run_configured_neural_predictions(
                self.project_root,
                [(row.row_id, row.source_path) for row in rows],
                run_dir,
            )
            apply_neural_runtime_authority(rows, neural_batch, project_root=self.project_root)
            if row_callback is not None:
                for row in rows:
                    row_callback(row)
            session = SortPreviewSession(
                run_dir=run_dir,
                input_path=Path(input_path).expanduser(),
                brain_path=brain_config.full_brain_path,
                available_labels=labels,
                rows=rows,
            )
            write_preview_manifest(run_dir / "Aaron_GUI_Preview.csv", session)
            return session
        except InputPreparationCancelled as exc:
            raise PreviewCancelled("Preview cancelled while preparing input.") from exc
        except PreviewCancelled:
            raise
        except Exception as exc:
            error_path = write_preview_error_report(run_dir, request.input_path, exc)
            message = friendly_preview_error_message(exc)
            raise RuntimeError(f"{message} Error report: {error_path}") from exc

    def resolve_brain_path(self, brain_path: Path | None = None) -> Path:
        """Resolve an optional brain path against the project root."""
        raw = Path(brain_path or DEFAULT_BRAIN).expanduser()
        return raw if raw.is_absolute() else self.project_root / raw

    def resolve_brain_config_path(self, brain_config_path: Path | None = None) -> Path:
        """Resolve the GUI brain-family config path against the project root."""
        configured = brain_config_path or os.environ.get("AARON_GUI_BRAIN_CONFIG") or DEFAULT_GUI_BRAIN_CONFIG
        raw = Path(configured).expanduser()
        return raw if raw.is_absolute() else self.project_root / raw

    def load_brain_family_config(self, *, full_brain_override: Path | None = None) -> BrainFamilyConfig:
        """Load the GUI brain-family config from YAML.

        Args:
            full_brain_override: Optional explicit full brain path for tests or
                internal callers. The GUI does not expose this to normal users.

        Returns:
            Resolved brain-family paths and lane participation flags.

        Raises:
            FileNotFoundError: If the YAML config is missing and no explicit
                full brain override is provided.
            ValueError: If the YAML file is malformed for the supported schema.
        """
        if self.brain_config_path.exists():
            config = load_simple_yaml_mapping(self.brain_config_path)
        elif full_brain_override is None:
            raise FileNotFoundError(f"GUI brain config not found: {self.brain_config_path}")
        else:
            config = {}
        brains = nested_mapping(config, "brains")
        baby = nested_mapping(brains, "baby")
        harmonic_baby = nested_mapping(brains, "harmonic_baby")
        options = nested_mapping(config, "options")
        full_brain = full_brain_override or scalar_path_value(brains.get("full"), DEFAULT_BRAIN)
        return BrainFamilyConfig(
            config_path=self.brain_config_path,
            full_brain_path=self.resolve_project_path(full_brain),
            baby_brain_path=self.resolve_optional_project_path(baby.get("legacy")),
            core_baby_brain_path=self.resolve_optional_project_path(baby.get("core")),
            spread_baby_brain_path=self.resolve_optional_project_path(baby.get("spread")),
            outlier_baby_brain_path=self.resolve_optional_project_path(baby.get("outlier")),
            user_memory_brain_path=self.resolve_optional_project_path(
                brains.get("user_memory") or USER_MEMORY_BRAIN_NAME
            ),
            physics_memory_brain_path=self.resolve_optional_project_path(
                brains.get("physics_memory") or PHYSICS_MEMORY_BRAIN_NAME
            ),
            voter_memory_brain_path=self.resolve_optional_project_path(
                brains.get("voter_memory") or VOTER_MEMORY_BRAIN_NAME
            ),
            shape_starter_memory_brain_path=self.resolve_optional_project_path(
                brains.get("shape_starter_memory") or SHAPE_STARTER_MEMORY_BRAIN_NAME
            ),
            shape_memory_brain_path=self.resolve_optional_project_path(
                brains.get("shape_memory") or SHAPE_MEMORY_BRAIN_NAME
            ),
            harmonic_core_baby_brain_path=self.resolve_optional_project_path(harmonic_baby.get("core")),
            harmonic_spread_baby_brain_path=self.resolve_optional_project_path(harmonic_baby.get("spread")),
            harmonic_outlier_baby_brain_path=self.resolve_optional_project_path(harmonic_baby.get("outlier")),
            use_baby_brains_in_sort=bool_option(options.get("use_baby_brains_in_sort"), default=True),
            use_harmonic_brains_in_sort=bool_option(options.get("use_harmonic_brains_in_sort"), default=False),
        )

    def resolve_project_path(self, path_value: Path | str) -> Path:
        """Resolve a path-like value against the project root."""
        raw = Path(path_value).expanduser()
        return raw if raw.is_absolute() else self.project_root / raw

    def resolve_optional_project_path(self, path_value: object) -> Path | None:
        """Resolve an optional path-like config value against the project root."""
        if path_value is None:
            return None
        text = str(path_value).strip()
        if not text:
            return None
        return self.resolve_project_path(text)

    def new_preview_run_dir(self) -> Path:
        """Create and return a new preview report folder."""
        run_dir = unique_path(self.project_root / "_reports" / "gui_preview" / f"run_{timestamp()}")
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir


def load_simple_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load the small indentation-based YAML subset used by GUI config.

    Args:
        path: YAML file to read.

    Returns:
        Nested dictionaries containing string and boolean scalar values.

    Raises:
        ValueError: If the file contains a line outside the supported
            ``key: value`` / ``key:`` mapping subset.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        stripped = content.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line {line_number} in {path}: {raw_line}")
        key, scalar = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Missing YAML key on line {line_number} in {path}")
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        scalar = scalar.strip()
        if scalar:
            parent[key] = parse_simple_yaml_scalar(scalar)
            continue
        child: dict[str, Any] = {}
        parent[key] = child
        stack.append((indent, child))
    return root


def parse_simple_yaml_scalar(raw_value: str) -> str | bool:
    """Parse a scalar from the GUI YAML subset."""
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


def nested_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a nested mapping from ``mapping`` or an empty mapping."""
    value = mapping.get(key)
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise ValueError(f"Expected '{key}' to be a mapping in GUI brain config")


def scalar_path_value(value: object, default: str) -> str:
    """Return a scalar path value from config with a default fallback."""
    if value is None:
        return default
    if isinstance(value, dict):
        raise ValueError("Expected brain path to be a scalar string")
    text = str(value).strip()
    return text or default


def bool_option(value: object, *, default: bool) -> bool:
    """Return a boolean option from YAML config with a default fallback."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def friendly_preview_error_message(exc: Exception) -> str:
    """Return a human-readable GUI preview error message.

    Args:
        exc: Exception raised while preparing or classifying preview audio.

    Returns:
        Error text suitable for the browser status area.

    Side Effects:
        None.
    """
    text = str(exc).strip()
    if text.startswith("No audio files found"):
        return (
            f"{text}. Choose a folder, ZIP, or file containing supported audio "
            f"({', '.join(sorted(AUDIO_LABEL_SUFFIXES))})."
        )
    if text:
        return text
    return f"{type(exc).__name__}: preview failed before a detailed error was reported."


def write_preview_error_report(run_dir: Path, input_path: Path, exc: Exception) -> Path:
    """Write a local error report for a failed GUI preview.

    Args:
        run_dir: Existing preview run directory.
        input_path: User-selected input path.
        exc: Exception that stopped preview creation.

    Returns:
        Path to the written text report.

    Side Effects:
        Writes ``Aaron_GUI_Preview_Error.txt`` under ``run_dir``.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    error_path = run_dir / "Aaron_GUI_Preview_Error.txt"
    lines = [
        "Aaron Sound Sorter GUI preview failed",
        f"Time: {datetime.now().isoformat(timespec='seconds')}",
        f"Input path: {Path(input_path).expanduser()}",
        f"Error type: {type(exc).__name__}",
        f"Error: {friendly_preview_error_message(exc)}",
        "",
        "Traceback:",
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    ]
    error_path.write_text("\n".join(lines), encoding="utf-8")
    return error_path


class SortPlanExporter:
    """Write a user-approved GUI sort plan to disk.

    Args:
        project_root: Repository root used for optional correction evidence
            packs under ``_reports/gui_corrections``.

    Side Effects:
        Writes sorted files and CSV/JSONL reports to the selected destination
        and optional correction report folder.
    """

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = Path(project_root or PROJECT_ROOT).expanduser().resolve()

    def export(
        self,
        session: SortPreviewSession,
        destination_root: Path,
        *,
        mode: ExportMode,
        write_correction_evidence: bool = False,
    ) -> ExportSummary:
        """Export approved rows by copy, move, or symlink.

        Args:
            session: Preview session with user-approved folders.
            destination_root: Destination folder selected by the user.
            mode: Export operation: ``copy``, ``move``, or ``symlink``.
            write_correction_evidence: When true, write a correction evidence
                pack under ``_reports/gui_corrections`` if corrections exist.

        Returns:
            Export summary with output paths and non-fatal errors.

        Raises:
            ValueError: If ``mode`` is not supported.
        """
        if mode not in {"copy", "move", "symlink"}:
            raise ValueError(f"Unsupported export mode: {mode}")
        record_approved_session_feedback(self.project_root, session)
        disable_macos_metadata_sidecars()
        destination = Path(destination_root).expanduser().resolve()
        sorted_root = destination / SORTED_ROOT_NAME
        sorted_root.mkdir(parents=True, exist_ok=True)
        exported_paths: dict[str, Path] = {}
        errors: list[str] = []
        for row in session.rows:
            try:
                exported_paths[row.row_id] = self.export_one_row(row, sorted_root=sorted_root, mode=mode)
            except Exception as exc:
                errors.append(f"{row.row_id}: {row.display_name}: {exc}")
        approved_plan_path = destination / "Aaron_GUI_Approved_Sort_Plan.csv"
        corrections_path = destination / "Aaron_GUI_Corrections.csv"
        write_approved_plan(approved_plan_path, session, exported_paths, mode=mode)
        write_corrections_csv(corrections_path, session)
        evidence_path = None
        if write_correction_evidence and session.corrected_count:
            evidence_path = self.write_correction_evidence_package(session)
        return ExportSummary(
            destination_root=destination,
            sorted_root=sorted_root,
            approved_plan_path=approved_plan_path,
            corrections_path=corrections_path,
            correction_evidence_path=evidence_path,
            exported_count=len(exported_paths),
            corrected_count=session.corrected_count,
            errors=errors,
        )

    @staticmethod
    def export_one_row(row: PreviewRow, *, sorted_root: Path, mode: ExportMode) -> Path:
        """Export one approved row into the sorted tree."""
        target_folder = sorted_root / safe_folder_path(row.approved_folder)
        target_folder.mkdir(parents=True, exist_ok=True)
        target_path = unique_path(target_folder / row.source_path.name)
        if mode == "copy":
            shutil.copy2(row.source_path, target_path)
        elif mode == "move":
            shutil.move(str(row.source_path), str(target_path))
        elif mode == "symlink":
            target_path.symlink_to(row.source_path.resolve())
        else:
            raise ValueError(f"Unsupported export mode: {mode}")
        return target_path

    def write_correction_evidence_package(
        self, session: SortPreviewSession, correction_dir: Path | None = None
    ) -> Path:
        """Write measured-evidence JSONL for corrected rows.

        The pack is intentionally a staging artifact. It does not mutate the
        training tree or rebuild brains.
        """
        output_dir = correction_dir or self.project_root / "_reports" / "gui_corrections" / f"run_{timestamp()}"
        output_dir.mkdir(parents=True, exist_ok=correction_dir is not None)
        evidence_path = output_dir / "Aaron_GUI_Correction_Evidence.jsonl"
        with evidence_path.open("w", encoding="utf-8") as handle:
            for row in session.rows:
                if row.is_corrected:
                    handle.write(json.dumps(correction_evidence(row), sort_keys=True) + "\n")
        write_corrections_csv(output_dir / "Aaron_GUI_Corrections.csv", session)
        return evidence_path


class TrainingCorrectionImporter:
    """Import GUI-corrected rows into the trusted training tree.

    Args:
        project_root: Project root that owns reports and the default training
            tree.
        training_root: Optional curated training root. Defaults to
            ``training/locked_curated_v1`` under the project root.

    Side Effects:
        Copies corrected audio files into training label slots and writes audit
        reports under ``_reports/gui_training_imports``.

    Important Constraints:
        This class stages human corrections as training data. It does not
        classify audio, inspect filenames for evidence, or rebuild brain JSON
        files by itself.
    """

    def __init__(self, project_root: Path | None = None, training_root: Path | None = None) -> None:
        self.project_root = Path(project_root or PROJECT_ROOT).expanduser().resolve()
        configured_training_root = training_root or self.project_root / DEFAULT_TRAINING_TAXONOMY_ROOT
        self.training_root = Path(configured_training_root).expanduser().resolve()
        self.exporter = SortPlanExporter(project_root=self.project_root)

    def import_session(self, session: SortPreviewSession) -> TrainingImportSummary:
        """Copy corrected session rows into training slots.

        Args:
            session: Preview session with approved folder corrections.

        Returns:
            Import summary with report paths and staged training files.

        Raises:
            OSError: If report or training folders cannot be created.
        """
        report_dir = unique_path(self.project_root / "_reports" / "gui_training_imports" / f"run_{timestamp()}")
        report_dir.mkdir(parents=True, exist_ok=False)
        correction_evidence_path = self.exporter.write_correction_evidence_package(session, correction_dir=report_dir)
        manifest_rows: list[dict[str, str]] = []
        staged_paths: list[Path] = []
        reused_existing_paths: list[Path] = []
        errors: list[str] = []
        skipped_count = 0
        calibration_store = ReviewFeedbackStore(
            self.project_root / "neural_artifacts" / "calibration_feedback" / "review_feedback.jsonl"
        )
        for row in session.rows:
            if not row.is_corrected:
                continue
            row_record = self.import_row(row, conflict_archive_dir=report_dir / "superseded_training_conflicts")
            manifest_rows.append(row_record)
            if row_record["status"] == "staged":
                staged_paths.append(Path(row_record["staged_path"]))
                feedback = calibration_feedback_from_preview_row(
                    row,
                    duplicate_conflict=int(row_record.get("superseded_conflicting_training_count", "0") or 0) > 0,
                    brain_version=str(session.brain_path),
                )
                if feedback is not None:
                    calibration_store.append(feedback)
                continue
            if row_record["status"] == "duplicate_existing":
                reused_existing_paths.append(Path(row_record["staged_path"]))
                feedback = calibration_feedback_from_preview_row(
                    row,
                    duplicate_conflict=int(row_record.get("superseded_conflicting_training_count", "0") or 0) > 0,
                    brain_version=str(session.brain_path),
                )
                if feedback is not None:
                    calibration_store.append(feedback)
                continue
            skipped_count += 1
            errors.append(f"{row.row_id}: {row.display_name}: {row_record['message']}")
        manifest_path = report_dir / "Aaron_GUI_Training_Import.csv"
        write_training_import_manifest(manifest_path, manifest_rows)
        neural_intake = NeuralTrainingInbox(self.project_root).queue_import_manifest(manifest_path)
        errors.extend(neural_intake.errors)
        return TrainingImportSummary(
            training_root=self.training_root,
            report_dir=report_dir,
            manifest_path=manifest_path,
            correction_evidence_path=correction_evidence_path,
            staged_count=len(staged_paths),
            reused_existing_count=len(reused_existing_paths),
            skipped_count=skipped_count,
            staged_paths=staged_paths,
            neural_intake_path=neural_intake.current_manifest_path,
            neural_event_log_path=neural_intake.event_log_path,
            neural_queued_count=neural_intake.queued_count,
            neural_reaffirmed_count=neural_intake.reaffirmed_count,
            neural_superseded_count=neural_intake.superseded_count,
            errors=errors,
        )

    def import_row(self, row: PreviewRow, *, conflict_archive_dir: Path | None = None) -> dict[str, str]:
        """Copy one corrected row into its approved training slot."""
        record = training_import_base_record(row)
        try:
            slot_path = training_slot_path(self.training_root, row.approved_folder)
        except ValueError as exc:
            record.update({"status": "skipped", "message": str(exc)})
            return record
        if not row.source_path.exists() or not row.source_path.is_file():
            record.update(
                {"status": "skipped", "training_slot": str(slot_path), "message": "source audio file not found"}
            )
            return record
        superseded_paths = supersede_conflicting_training_audio(
            self.training_root,
            approved_slot_path=slot_path,
            source_path=row.source_path,
            archive_root=conflict_archive_dir,
        )
        slot_path.mkdir(parents=True, exist_ok=True)
        duplicate_path = existing_training_audio_match(slot_path, row.source_path)
        if duplicate_path is not None:
            record.update(
                {
                    "status": "duplicate_existing",
                    "training_slot": str(slot_path),
                    "staged_path": str(duplicate_path),
                    "message": "matching audio is already staged in this training slot",
                    "superseded_conflicting_training_count": str(len(superseded_paths)),
                    "superseded_conflicting_training_paths": json.dumps([str(path) for path in superseded_paths]),
                }
            )
            return record
        target_path = unique_path(slot_path / row.source_path.name)
        shutil.copy2(row.source_path, target_path)
        record.update(
            {
                "status": "staged",
                "training_slot": str(slot_path),
                "staged_path": str(target_path),
                "message": "copied corrected audio into trusted training slot",
                "superseded_conflicting_training_count": str(len(superseded_paths)),
                "superseded_conflicting_training_paths": json.dumps([str(path) for path in superseded_paths]),
            }
        )
        return record


def existing_training_audio_match(slot_path: Path, source_path: Path) -> Path | None:
    """Return an existing slot file that is byte-identical to source audio.

    Args:
        slot_path: Training slot folder to scan.
        source_path: Corrected audio file selected by the user.

    Returns:
        Matching existing audio path, or ``None`` when this correction is new.

    Side Effects:
        Reads candidate file metadata and content for comparison only.
    """
    if not slot_path.exists() or not slot_path.is_dir():
        return None
    for candidate_path in sorted(slot_path.iterdir()):
        if not candidate_path.is_file():
            continue
        if candidate_path.suffix.lower() not in AUDIO_LABEL_SUFFIXES:
            continue
        if filecmp.cmp(candidate_path, source_path, shallow=False):
            return candidate_path
    return None


def supersede_conflicting_training_audio(
    training_root: Path,
    *,
    approved_slot_path: Path,
    source_path: Path,
    archive_root: Path | None,
) -> list[Path]:
    """Archive exact same-audio teachers from non-approved training slots.

    Args:
        training_root: Curated training root to scan.
        approved_slot_path: Slot selected by the latest human correction.
        source_path: Audio file approved by the user.
        archive_root: Report folder where superseded files are moved.

    Returns:
        Original paths removed from conflicting training slots.

    Side Effects:
        Moves exact byte-identical conflicting audio files into ``archive_root``
        when provided.

    Important Constraints:
        This is training-data hygiene, not runtime sorting evidence. It compares
        audio file bytes only and does not use names or folders to classify
        unknown audio.
    """
    root = Path(training_root).expanduser().resolve()
    approved_slot = Path(approved_slot_path).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()
    if not root.exists() or not source.is_file():
        return []
    try:
        source_size = source.stat().st_size
    except OSError:
        return []
    moved: list[Path] = []
    for candidate_path in sorted(root.rglob("*")):
        if not candidate_path.is_file() or candidate_path.suffix.lower() not in AUDIO_LABEL_SUFFIXES:
            continue
        try:
            resolved_candidate = candidate_path.resolve()
            if resolved_candidate == source or approved_slot in resolved_candidate.parents:
                continue
            if candidate_path.stat().st_size != source_size:
                continue
            if not filecmp.cmp(candidate_path, source, shallow=False):
                continue
        except OSError:
            continue
        moved.append(candidate_path)
        if archive_root is not None:
            archive_target = unique_path(Path(archive_root).expanduser().resolve() / candidate_path.relative_to(root))
            archive_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidate_path), str(archive_target))
    return moved


def training_slot_path(training_root: Path, approved_folder: str) -> Path:
    """Return the training slot folder for an approved GUI taxonomy label.

    Args:
        training_root: Root of the curated training tree.
        approved_folder: Human-approved output taxonomy label.

    Returns:
        Folder path ending in a training structure folder such as ``_LOOPS``.

    Raises:
        ValueError: If the approved folder is a review path or lacks a trainable
            terminal structure.

    Side Effects:
        None.
    """
    contract = taxonomy_label_contract(approved_folder)
    normalized = contract.canonical_label
    if not is_valid_taxonomy_label(normalized):
        reason = ", ".join(contract.reasons) if contract.reasons else "invalid_taxonomy_label"
        raise ValueError(f"approved folder is not a valid training label ({reason}): {approved_folder}")
    parts = normalized.split("/")
    if parts[0] == "_TO_REVIEW":
        raise ValueError("review folders are not trainable brain labels")
    structure_folder_by_label = {value: key for key, value in TRAINING_STRUCTURE_FOLDER_NAMES.items()}
    structure_folder = structure_folder_by_label.get(parts[-1])
    if structure_folder is None:
        allowed = ", ".join(sorted(structure_folder_by_label))
        raise ValueError(f"approved folder must end with one of: {allowed}")
    label_parent = safe_folder_path("/".join(parts[:-1]))
    return Path(training_root).expanduser().resolve() / label_parent / structure_folder


def training_import_base_record(row: PreviewRow) -> dict[str, str]:
    """Return base CSV values for one training import row."""
    return {
        "row_id": row.row_id,
        "source_path": str(row.source_path),
        "display_name": row.display_name,
        "proposed_folder": row.proposed_folder,
        "approved_folder": row.approved_folder,
        "training_slot": "",
        "staged_path": "",
        "status": "",
        "message": "",
        "superseded_conflicting_training_count": "0",
        "superseded_conflicting_training_paths": "[]",
    }


def training_import_manifest_fields() -> list[str]:
    """Return CSV fields for GUI training import manifests."""
    return [
        "row_id",
        "source_path",
        "display_name",
        "proposed_folder",
        "approved_folder",
        "training_slot",
        "staged_path",
        "status",
        "message",
        "superseded_conflicting_training_count",
        "superseded_conflicting_training_paths",
    ]


def write_training_import_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the GUI training import manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=training_import_manifest_fields())
        writer.writeheader()
        writer.writerows(rows)


def build_product_sorter(candidate_count: int = 100) -> SortSamplesUseCase:
    """Build the same product sorter dependencies used by the CLI runner."""
    voters = declare_voters(candidate_count=candidate_count)
    return SortSamplesUseCase(
        brain_repository=BrainRepository(),
        audio_repository=AudioInputRepository(),
        consensus_runner=ConsensusRunner(ConsensusPolicy(top_n=candidate_count)),
        voters=voters,
        arbiter=FamilyClaimArbiter(placement_resolver=PlacementResolver()),
    )


def declare_voters(candidate_count: int = 100) -> list[Voter]:
    """Declare product voters for GUI preview classification."""
    return [
        FullBrainVoter(BrainVoterPolicy(top_n=candidate_count)),
        BalancedRecallBrainVoter(
            BrainVoterPolicy(top_n=candidate_count),
            lane_name="core_baby",
            purpose="precision_clean_center",
        ),
        BalancedRecallBrainVoter(
            BrainVoterPolicy(top_n=candidate_count),
            lane_name="spread_baby",
            purpose="balanced_clean_diversity",
        ),
        BalancedRecallBrainVoter(
            BrainVoterPolicy(top_n=candidate_count),
            lane_name="outlier_baby",
            purpose="edge_case_recall_not_final_truth",
        ),
        PhysicsVoter(PhysicsVoterPolicy(top_n=candidate_count)),
        ShapeVoter(ShapeVoterPolicy()),
    ]


def classify_audio_files_with_progress(
    sorter: SortSamplesUseCase,
    audio_files: list[Path],
    brain: dict[str, Any],
    baby_brains: dict[str, dict[str, Any] | None] | None,
    harmonic_baby_brains: dict[str, dict[str, Any] | None] | None,
    *,
    use_baby_brains_in_sort: bool,
    use_harmonic_brains_in_sort: bool,
    max_workers: int,
    progress_callback: PreviewProgressCallback | None,
    row_callback: PreviewRowCallback | None = None,
    cancel_requested: CancelRequestedCallback | None = None,
) -> list[SortFileResult]:
    """Classify audio files and optionally publish GUI progress.

    Args:
        sorter: Product sorter use case.
        audio_files: Prepared audio files in display/export order.
        brain: Loaded full brain.
        baby_brains: Loaded normal baby brains by lane.
        harmonic_baby_brains: Loaded harmonic baby brains by lane.
        use_baby_brains_in_sort: Whether normal baby brains participate.
        use_harmonic_brains_in_sort: Whether harmonic baby brains participate.
        max_workers: Maximum per-file classification workers.
        progress_callback: Optional callback receiving completed count, total
            count, and latest file name.
        row_callback: Optional callback receiving each completed preview row.
        cancel_requested: Optional callback returning true when classification
            should stop scheduling more audio files.

    Returns:
        Sort results in the same order as ``audio_files``.

    Raises:
        PreviewCancelled: If ``cancel_requested`` returns true before or during
            classification.

    Side Effects:
        Calls ``progress_callback`` from the worker collection thread. It does
        not write sorted output or mutate brains.
    """
    if progress_callback is None and row_callback is None and cancel_requested is None:
        return sorter.classify_audio_files(
            audio_files,
            brain,
            baby_brains,
            harmonic_baby_brains,
            use_baby_brains_in_sort=use_baby_brains_in_sort,
            use_harmonic_brains_in_sort=use_harmonic_brains_in_sort,
            max_workers=max_workers,
        )
    total_files = len(audio_files)
    if max_workers <= 1 or total_files <= 1:
        ordered_results = []
        for completed_count, audio_file in enumerate(audio_files, start=1):
            _raise_if_preview_cancelled(cancel_requested)
            result = sorter.classify_one_file(
                audio_file,
                brain,
                baby_brains,
                harmonic_baby_brains,
                use_baby_brains_in_sort=use_baby_brains_in_sort,
                use_harmonic_brains_in_sort=use_harmonic_brains_in_sort,
            )
            ordered_results.append(result)
            if row_callback is not None:
                row_callback(preview_row_from_result(completed_count, result))
            if progress_callback is not None:
                progress_callback(completed_count, total_files, audio_file.name)
            _raise_if_preview_cancelled(cancel_requested)
        return ordered_results
    ordered_results: list[SortFileResult | None] = [None] * total_files
    worker_count = max(1, min(int(max_workers), total_files))
    completed_count = 0
    next_index = 0
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="aaron-gui-preview") as executor:
        futures = {}

        def submit_next_file() -> bool:
            nonlocal next_index
            _raise_if_preview_cancelled(cancel_requested)
            if next_index >= total_files:
                return False
            audio_file = audio_files[next_index]
            future = executor.submit(
                sorter.classify_one_file,
                audio_file,
                brain,
                baby_brains,
                harmonic_baby_brains,
                use_baby_brains_in_sort=use_baby_brains_in_sort,
                use_harmonic_brains_in_sort=use_harmonic_brains_in_sort,
            )
            futures[future] = (next_index, audio_file)
            next_index += 1
            return True

        for _ in range(worker_count):
            if not submit_next_file():
                break
        while futures:
            _raise_if_preview_cancelled(cancel_requested)
            done_futures, _pending_futures = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
            if not done_futures:
                continue
            for future in done_futures:
                index, audio_file = futures.pop(future)
                result = future.result()
                ordered_results[index] = result
                completed_count += 1
                if row_callback is not None:
                    row_callback(preview_row_from_result(index + 1, result))
                if progress_callback is not None:
                    progress_callback(completed_count, total_files, audio_file.name)
                submit_next_file()
    return [result for result in ordered_results if result is not None]


def _raise_if_preview_cancelled(cancel_requested: CancelRequestedCallback | None) -> None:
    """Raise when the browser has requested cancellation."""
    if cancel_requested is not None and cancel_requested():
        raise PreviewCancelled("Preview cancelled by user.")


def preview_row_from_result(index: int, result: SortFileResult) -> PreviewRow:
    """Adapt one classifier result into a GUI preview row."""
    decision = result.decision
    raw_folder = str(decision.folder_path or decision.final_label or "_TO_REVIEW/Unknown")
    folder = normalize_taxonomy_label(raw_folder)
    return PreviewRow(
        row_id=f"{index:05d}",
        source_path=result.source_path,
        display_name=result.source_path.name,
        proposed_folder=folder,
        approved_folder=folder,
        final_top=str(decision.final_top or ""),
        consensus_status=str(decision.consensus_status or ""),
        confidence=decision_confidence(result),
        duration_sec=float(result.physics.duration_sec),
        read_status=str(result.physics.read_status),
        decision_reason=str(decision.reason or ""),
        diagnostic_summary=diagnostic_summary(result),
        candidate_folders=detected_candidate_folders(result),
        neural_decision_state="provisional",
        result=result,
    )


def apply_neural_runtime_authority(
    rows: list[PreviewRow],
    batch: NeuralRuntimeBatch,
    *,
    project_root: Path | None = None,
) -> None:
    """Apply conservative category-wide neural ownership to GUI proposals.

    Exact human-trained content owns the proposal immediately. A supported,
    separated learned neighborhood owns only after the independent safety
    gates accept it. Weak or sparse neural disagreements become Review across
    voice, instruments, drums, and FX.
    """
    predictions = {prediction.row_id: prediction for prediction in batch.predictions}
    calibration = load_optional_confidence_calibration(project_root) if project_root is not None else None
    authority_groups = enabled_authority_groups(project_root) if project_root is not None else None
    for row in rows:
        prediction = predictions.get(row.row_id)
        row.neural_runtime_status = batch.status
        row.neural_runtime_message = batch.message
        row.neural_row_error = str(batch.row_errors.get(row.row_id, ""))
        if prediction is None:
            row.neural_decision_state = "finalized_without_neural"
            if row.neural_row_error:
                row.neural_runtime_status = "row_error"
                row.diagnostic_summary = (
                    f"{row.diagnostic_summary}; neural=row_error ({row.neural_row_error})"
                )
            elif batch.status == "partial_timeout":
                row.neural_runtime_status = "not_completed_timeout"
                row.diagnostic_summary = (
                    f"{row.diagnostic_summary}; neural=not_completed_timeout ({batch.message})"
                )
            elif batch.status in {"error", "unavailable"}:
                row.diagnostic_summary = f"{row.diagnostic_summary}; neural={batch.status} ({batch.message})"
            continue
        row.neural_runtime_status = "predicted"
        row.neural_decision_state = "finalized"
        _apply_neural_prediction(
            row,
            prediction,
            calibration=calibration,
            authority_groups=authority_groups,
        )


def _apply_neural_prediction(
    row: PreviewRow,
    prediction: NeuralRuntimePrediction,
    *,
    calibration: ConfidenceCalibrationBundle | None = None,
    authority_groups: frozenset[str] | None = None,
) -> None:
    """Apply one neural result without consulting source-name text."""
    neural_label = normalize_taxonomy_label(prediction.predicted_label)
    row.neural_folder = neural_label
    row.neural_known_distribution = prediction.known_distribution
    row.neural_ownership_ready = prediction.ownership_ready
    row.neural_ownership_reason = prediction.ownership_block_reason
    row.neural_label_example_count = prediction.label_example_count
    row.neural_exact_training_match = prediction.exact_training_match
    row.neural_similarity = prediction.top_similarity
    row.neural_margin = prediction.margin
    row.neural_radius_ratio = prediction.radius_ratio
    row.neural_semantic_status = prediction.semantic_status
    row.neural_semantic_family = prediction.semantic_family
    row.neural_semantic_second_family = prediction.semantic_second_family
    row.neural_semantic_score = prediction.semantic_top_score
    row.neural_semantic_second_score = prediction.semantic_second_score
    row.neural_semantic_margin = prediction.semantic_margin
    row.neural_semantic_family_scores = dict(prediction.semantic_family_scores)
    row.neural_prompt_status = prediction.prompt_brain_status
    row.neural_prompt_suggestions = [
        {
            "path": suggestion.path,
            "positive_score": suggestion.positive_score,
            "negative_score": suggestion.negative_score,
            "prompt_margin": suggestion.prompt_margin,
            "top_positive_prompt": suggestion.top_positive_prompt,
            "top_positive_similarity": suggestion.top_positive_similarity,
            "top_negative_prompt": suggestion.top_negative_prompt,
            "top_negative_similarity": suggestion.top_negative_similarity,
        }
        for suggestion in prediction.prompt_suggestions
    ]
    row.panns_status = prediction.panns_status
    row.panns_model_id = prediction.panns_model_id
    row.panns_events = [{"label": event.label, "score": event.score} for event in prediction.panns_events]
    row.panns_family_scores = dict(prediction.panns_family_scores)
    row.panns_support_score = prediction.panns_support_score
    row.panns_contradiction_score = prediction.panns_contradiction_score
    row.panns_supporting_events = [
        {"label": event.label, "score": event.score} for event in prediction.panns_supporting_events
    ]
    row.panns_contradicting_events = [
        {"label": event.label, "score": event.score} for event in prediction.panns_contradicting_events
    ]
    if neural_label and neural_label not in row.candidate_folders:
        row.candidate_folders.insert(0, neural_label)

    neural_summary = (
        f"neural={neural_label or 'none'} "
        f"(known={prediction.known_distribution}, "
        f"ownership_ready={prediction.ownership_ready}, "
        f"ownership_reason={prediction.ownership_block_reason or 'unspecified'}, "
        f"label_examples={prediction.label_example_count}, "
        f"exact_training={prediction.exact_training_match}, "
        f"similarity={prediction.top_similarity:.3f}, "
        f"margin={prediction.margin:.3f}, "
        f"radius_ratio={prediction.radius_ratio:.3f}, "
        f"semantic={prediction.semantic_family or 'none'}, "
        f"semantic_score={prediction.semantic_top_score:.3f}, "
        f"semantic_margin={prediction.semantic_margin:.3f}, "
        f"panns_families={dict(prediction.panns_family_scores)})"
    )
    row.diagnostic_summary = f"{row.diagnostic_summary}; {neural_summary}"
    if not is_valid_taxonomy_label(neural_label):
        return

    if row.result is not None and authority_groups is not None:
        updated_result = apply_neural_prediction_to_result(
            row.result,
            prediction,
            calibration=calibration,
            enabled_groups=authority_groups,
        )
        row.result = updated_result
        decision = updated_result.decision
        decided_folder = normalize_taxonomy_label(decision.folder_path or decision.final_label)
        row.proposed_folder = decided_folder
        row.approved_folder = decided_folder
        row.final_top = str(decision.final_top or "")
        row.consensus_status = str(decision.consensus_status or "")
        row.confidence = decision_confidence(updated_result)
        row.decision_reason = str(decision.reason or "")
        runtime_evidence = updated_result.facts.evidence.get("neural_runtime", {})
        if isinstance(runtime_evidence, dict):
            row.neural_ownership_ready = bool(runtime_evidence.get("ownership_ready", False))
            row.neural_ownership_reason = str(runtime_evidence.get("ownership_block_reason", ""))
        return

    legacy_label = row.proposed_folder
    if prediction.ownership_ready:
        if prediction.exact_training_match:
            row.proposed_folder = neural_label
            row.approved_folder = neural_label
            row.final_top = neural_label.split("/", 1)[0]
            row.consensus_status = "neural_known_distribution_owner"
            row.confidence = 1.0
            row.decision_reason = (
                "This exact audio was explicitly taught by the user. Independent neural "
                "witnesses remain visible diagnostics but cannot veto the correction."
            )
            return
        if neural_panns_contradicts(prediction):
            event_names = ", ".join(event.label for event in prediction.panns_contradicting_events[:3])
            row.neural_ownership_ready = False
            row.neural_ownership_reason = "panns_family_contradiction"
            row.proposed_folder = "_TO_REVIEW/Measured Role Conflict"
            row.approved_folder = row.proposed_folder
            row.final_top = "_TO_REVIEW"
            row.consensus_status = "neural_panns_family_conflict_review"
            row.decision_reason = (
                "Learned memory matched, but independent PANNs audio events "
                f"contradicted its source family ({event_names}); forcing human review."
            )
            return
        semantic_assessment = neural_semantic_compatibility(prediction, neural_label)
        if semantic_assessment.contradictory:
            row.neural_ownership_ready = False
            row.neural_ownership_reason = semantic_assessment.reason
            row.proposed_folder = "_TO_REVIEW/Measured Role Conflict"
            row.approved_folder = row.proposed_folder
            row.final_top = "_TO_REVIEW"
            row.consensus_status = "neural_semantic_family_conflict_review"
            row.decision_reason = (
                "Learned memory matched, but independent audio-only semantics "
                f"contradicted its source family ({prediction.semantic_family} vs {neural_label}); "
                "forcing human review."
            )
            return
        if row.result is not None and neural_structure_conflicts(row.result, neural_label):
            row.proposed_folder = "_TO_REVIEW/Measured Role Conflict"
            row.approved_folder = row.proposed_folder
            row.final_top = "_TO_REVIEW"
            row.consensus_status = "neural_measured_structure_conflict_review"
            row.decision_reason = (
                "Neural identity was inside a learned neighborhood, but its structure "
                "contradicted measured audio structure."
            )
            return
        row.proposed_folder = neural_label
        row.approved_folder = neural_label
        row.final_top = neural_label.split("/", 1)[0]
        row.consensus_status = "neural_known_distribution_owner"
        row.confidence = max(0.0, min(1.0, prediction.top_similarity))
        row.decision_reason = (
            "Source-name-blind neural audio had production-ready learned evidence "
            f"and took ownership from the legacy proposal ({legacy_label})."
        )
        return

    if row.result is not None and neural_defers_to_exact_human_teacher(row.result, prediction):
        row.neural_ownership_reason = "defer_to_exact_human_teacher"
        return

    if neural_disagreement_requires_review(
        prediction,
        legacy_label,
        neural_label,
        result=row.result,
    ):
        neural_family, legacy_family = neural_conflict_families(neural_label, legacy_label)
        row.proposed_folder = "_TO_REVIEW/Measured Role Conflict"
        row.approved_folder = row.proposed_folder
        row.final_top = "_TO_REVIEW"
        row.consensus_status = "neural_legacy_owner_conflict_review"
        row.decision_reason = (
            "Neural audio was not ready for production ownership and disagreed with the "
            f"legacy source family ({neural_family} vs {legacy_family}); forcing human review."
        )


def decision_confidence(result: SortFileResult) -> float:
    """Return a compact confidence value for display."""
    if result.decision.combined_rank_score is not None:
        return max(0.0, 1.0 / (1.0 + float(result.decision.combined_rank_score)))
    confidences = [guess.confidence for guess in result.brain_votes.guesses[:1] + result.physics_votes.guesses[:1]]
    return max([float(value) for value in confidences] or [0.0])


def diagnostic_summary(result: SortFileResult) -> str:
    """Build a short display summary from voter and shape evidence."""
    evidence = result.facts.evidence if isinstance(result.facts.evidence, dict) else {}
    shape = evidence.get("shape_vote", {}) if isinstance(evidence.get("shape_vote", {}), dict) else {}
    primary_shape = shape.get("primary_shape", "")
    shape_conf = shape.get("confidence", "")
    brain_top = result.brain_votes.guesses[0].folder_path if result.brain_votes.guesses else ""
    physics_top_guess = compact_top_guess(evidence.get("physics_vote_result", {}))
    physics_top = str(physics_top_guess.get("folder_path", ""))
    if not physics_top:
        physics_top = result.physics_votes.guesses[0].folder_path if result.physics_votes.guesses else ""
    return f"shape={primary_shape} ({shape_conf}); brain={brain_top or 'none'}; physics={physics_top or 'none'}"


def detected_candidate_folders(result: SortFileResult, *, limit: int = 40) -> list[str]:
    """Return voter-detected folder options for GUI correction hints.

    Args:
        result: Completed sorter result for one audio file.
        limit: Maximum number of unique folders returned.

    Returns:
        Unique taxonomy folders proposed by final arbitration, brain lanes,
        physics, and shared candidates.

    Side Effects:
        None.

    Important Constraints:
        This is display-only diagnostic data. It must not feed classification
        decisions or inspect source filenames/folders as audio evidence.
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(raw_value: object) -> None:
        if len(candidates) >= limit:
            return
        label = normalize_taxonomy_label(raw_value)
        if not is_valid_taxonomy_label(label) or label in seen:
            return
        seen.add(label)
        candidates.append(label)

    add_candidate(result.decision.folder_path or result.decision.final_label)
    add_guess_folders(candidates, seen, result.brain_votes.guesses, limit=limit)
    add_guess_folders(candidates, seen, result.physics_votes.guesses, limit=limit)
    for candidate in result.decision.shared_candidates[:limit]:
        if isinstance(candidate, dict):
            add_candidate(candidate.get("folder_path") or candidate.get("label"))
    evidence = result.facts.evidence if isinstance(result.facts.evidence, dict) else {}
    for digest_key in diagnostic_vote_digest_keys():
        digest = evidence.get(digest_key, {})
        if isinstance(digest, dict):
            add_digest_folders(candidates, seen, digest, limit=limit)
    authority_trace = result.decision.authority_trace if isinstance(result.decision.authority_trace, dict) else {}
    for trace_key in ("raw_claim", "winner_after_pick"):
        trace_value = authority_trace.get(trace_key, {})
        if isinstance(trace_value, dict):
            add_candidate(trace_value.get("path") or trace_value.get("folder_path") or trace_value.get("label"))
    if has_decisive_loop_structure(result):
        candidates = [
            candidate
            for candidate in candidates
            if taxonomy_label_contract(candidate).structure_terminal != "One Shots"
        ]
    return candidates



def add_guess_folders(
    candidates: list[str],
    seen: set[str],
    guesses: list[Any],
    *,
    limit: int,
) -> None:
    """Append normalized folder paths from category guesses."""
    for guess in guesses[:limit]:
        if len(candidates) >= limit:
            return
        label = normalize_taxonomy_label(getattr(guess, "folder_path", "") or getattr(guess, "label", ""))
        if not is_valid_taxonomy_label(label) or label in seen:
            continue
        seen.add(label)
        candidates.append(label)


def add_digest_folders(candidates: list[str], seen: set[str], digest: dict[str, Any], *, limit: int) -> None:
    """Append normalized folder paths from a compact voter digest."""
    guesses = digest.get("top_guesses", [])
    if not isinstance(guesses, list):
        return
    for guess in guesses[:limit]:
        if len(candidates) >= limit:
            return
        if not isinstance(guess, dict):
            continue
        label = normalize_taxonomy_label(guess.get("folder_path") or guess.get("label"))
        if not is_valid_taxonomy_label(label) or label in seen:
            continue
        seen.add(label)
        candidates.append(label)


def diagnostic_vote_digest_keys() -> tuple[str, ...]:
    """Return evidence keys that may contain compact voter top candidates."""
    return (
        "physics_vote_result",
        "brain_ensemble_vote_result",
        "full_brain_vote_result",
        "baby_brain_vote_result",
        "core_baby_vote_result",
        "spread_baby_vote_result",
        "outlier_baby_vote_result",
        "harmonic_core_baby_vote_result",
        "harmonic_spread_baby_vote_result",
        "harmonic_outlier_baby_vote_result",
        "dry_core_brain_ensemble_vote_result",
        "dry_core_full_brain_vote_result",
        "dry_core_physics_vote_result",
    )


def load_available_labels(
    brain_path: Path | BrainFamilyConfig,
    *,
    project_root: Path | None = None,
    taxonomy_catalog_path: Path | None = None,
) -> list[str]:
    """Load chooser taxonomy labels for GUI correction workflows.

    Args:
        brain_path: Active trained brain JSON or GUI brain-family config.
            Its labels represent folders the sorter can currently recognize.
        project_root: Optional project root used to merge training-folder slots,
            the master taxonomy ledger, and GUI-only future taxonomy labels.
        taxonomy_catalog_path: Optional catalog path. Relative paths resolve
            against ``project_root``.

    Returns:
        Review labels followed by sorted unique taxonomy labels.

    Side Effects:
        Reads JSON files and scans training taxonomy directories when they
        exist. It does not train, sort, or mutate brain files.

    Important Constraints:
        These labels are used only by the GUI chooser and correction-evidence
        writer. They are not sorting evidence and must not influence voters.
    """
    label_set = set()
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
        registry_path = root / DEFAULT_CANONICAL_TAXONOMY
        if registry_path.is_file():
            registry = TaxonomyRegistry.load(registry_path, root / DEFAULT_TAXONOMY_ALIASES)
            return REVIEW_LABELS + registry.manual_labels()
    if isinstance(brain_path, BrainFamilyConfig):
        label_set.update(load_brain_family_taxonomy_labels(brain_path))
    else:
        label_set.update(load_brain_taxonomy_labels(Path(brain_path)))
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
        label_set.update(load_training_taxonomy_labels(root / DEFAULT_TRAINING_TAXONOMY_ROOT))
        label_set.update(load_json_taxonomy_labels(root / DEFAULT_MASTER_TAXONOMY_LEDGER))
        catalog_path = resolve_taxonomy_catalog_path(root, taxonomy_catalog_path)
        label_set.update(load_json_taxonomy_labels(catalog_path))
    normalized = sorted({label for label in label_set if is_valid_taxonomy_label(label)})
    return REVIEW_LABELS + normalized


def resolve_taxonomy_catalog_path(project_root: Path, taxonomy_catalog_path: Path | None) -> Path:
    """Resolve the GUI taxonomy catalog path against the project root.

    Args:
        project_root: Project root that owns the default config folder.
        taxonomy_catalog_path: Optional explicit catalog path.

    Returns:
        Absolute path to the catalog JSON.

    Side Effects:
        None.
    """
    configured = taxonomy_catalog_path or DEFAULT_GUI_TAXONOMY_CATALOG
    raw_path = Path(configured).expanduser()
    return raw_path if raw_path.is_absolute() else project_root / raw_path


def load_brain_taxonomy_labels(brain_path: Path) -> list[str]:
    """Load trained taxonomy labels from one brain JSON file.

    Args:
        brain_path: Brain JSON path to read.

    Returns:
        Normalized trained labels from the brain's ``labels`` list.

    Side Effects:
        Reads ``brain_path`` when it exists.
    """
    resolved = Path(brain_path).expanduser()
    if not resolved.exists() or not resolved.is_file():
        return []
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            brain = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    labels = brain.get("labels", [])
    if not isinstance(labels, list):
        return []
    return [normalize_taxonomy_label(label) for label in labels]


def load_brain_family_taxonomy_labels(brain_family: BrainFamilyConfig) -> list[str]:
    """Load trained taxonomy labels from every configured GUI brain file.

    Args:
        brain_family: GUI brain-family configuration.

    Returns:
        Normalized labels from every configured brain file.

    Side Effects:
        Reads the configured brain JSON files.
    """
    paths: list[Path] = [brain_family.full_brain_path]
    paths.extend(
        path
        for path in [
            brain_family.baby_brain_path,
            brain_family.core_baby_brain_path,
            brain_family.spread_baby_brain_path,
            brain_family.outlier_baby_brain_path,
            brain_family.user_memory_brain_path,
            brain_family.physics_memory_brain_path,
            brain_family.voter_memory_brain_path,
            brain_family.shape_starter_memory_brain_path,
            brain_family.shape_memory_brain_path,
            brain_family.harmonic_core_baby_brain_path,
            brain_family.harmonic_spread_baby_brain_path,
            brain_family.harmonic_outlier_baby_brain_path,
        ]
        if path is not None
    )
    labels: list[str] = []
    for path in paths:
        try:
            labels.extend(load_brain_taxonomy_labels(path))
        except (OSError, json.JSONDecodeError):
            continue
    return labels


def load_json_taxonomy_labels(path: Path) -> list[str]:
    """Load taxonomy labels from a flexible JSON catalog or ledger.

    Args:
        path: JSON file containing either a ``labels`` list or nested ledger
            values such as ``primary_folder_path``.

    Returns:
        Valid-looking taxonomy labels discovered in the JSON tree.

    Side Effects:
        Reads ``path`` when it exists. Missing files are ignored.
    """
    resolved = Path(path).expanduser()
    if not resolved.exists() or not resolved.is_file():
        return []
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    discovered = discover_taxonomy_label_strings(payload)
    return [normalize_taxonomy_label(label) for label in discovered]


def discover_taxonomy_label_strings(payload: Any) -> list[str]:
    """Return taxonomy-shaped strings from a JSON-compatible value.

    Args:
        payload: JSON-compatible value to inspect.

    Returns:
        Strings that look like internal taxonomy labels.

    Side Effects:
        None.
    """
    if isinstance(payload, str):
        label = normalize_taxonomy_label(payload)
        return [label] if is_valid_taxonomy_label(label) else []
    if isinstance(payload, list):
        labels: list[str] = []
        for element in payload:
            labels.extend(discover_taxonomy_label_strings(element))
        return labels
    if isinstance(payload, dict):
        labels = []
        explicit_labels = payload.get("labels")
        if isinstance(explicit_labels, list):
            labels.extend(discover_taxonomy_label_strings(explicit_labels))
        for value in payload.values():
            if value is explicit_labels:
                continue
            labels.extend(discover_taxonomy_label_strings(value))
        return labels
    return []


def load_training_taxonomy_labels(training_root: Path) -> list[str]:
    """Load taxonomy labels from existing curated training slot folders.

    Args:
        training_root: Root of ``training/locked_curated_v1``.

    Returns:
        Labels converted from slot folders such as ``_ONE_SHOTS`` and
        ``_LOOPS``.

    Side Effects:
        Scans directory names only. It does not inspect audio filenames.
    """
    resolved_root = Path(training_root).expanduser()
    if not resolved_root.exists() or not resolved_root.is_dir():
        return []
    labels = []
    for slot_dir in resolved_root.rglob("*"):
        if not slot_dir.is_dir():
            continue
        structure_label = TRAINING_STRUCTURE_FOLDER_NAMES.get(slot_dir.name)
        if structure_label is None:
            continue
        parent_parts = slot_dir.parent.relative_to(resolved_root).parts
        labels.append(normalize_taxonomy_label("/".join([*parent_parts, structure_label])))
    return labels


def normalize_taxonomy_label(value: object) -> str:
    """Normalize one GUI taxonomy label string.

    Args:
        value: Raw label-like value.

    Returns:
        Slash-separated label with empty path parts removed.

    Side Effects:
        None.
    """
    return canonicalize_taxonomy_label(normalize_taxonomy_path(value))


def is_valid_taxonomy_label(label: str) -> bool:
    """Return whether a label is safe for the GUI taxonomy chooser.

    Args:
        label: Normalized taxonomy label.

    Returns:
        ``True`` when the label belongs to an internal top family and does not
        look like an audio filename or filesystem path.

    Side Effects:
        None.
    """
    normalized = normalize_taxonomy_label(label)
    if not normalized or normalized.startswith("/") or len(normalized) > 220:
        return False
    parts = normalized.split("/")
    if len(parts) < 2 or parts[0] not in TOP_LEVEL_TAXONOMY_FAMILIES:
        return False
    if any(part in {".", ".."} for part in parts):
        return False
    if Path(parts[-1]).suffix.lower() in AUDIO_LABEL_SUFFIXES:
        return False
    if parts[0] == "_TO_REVIEW":
        return True
    if parts[-1] not in STRUCTURE_TERMINALS:
        return True
    return is_valid_taxonomy_contract_label(normalized)


def write_preview_manifest(path: Path, session: SortPreviewSession) -> None:
    """Write a lightweight preview manifest for GUI sessions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=preview_manifest_fields())
        writer.writeheader()
        for row in session.rows:
            writer.writerow(preview_row_to_csv(row))


def write_approved_plan(
    path: Path,
    session: SortPreviewSession,
    exported_paths: dict[str, Path],
    *,
    mode: ExportMode,
) -> None:
    """Write every approved placement row after export."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [*preview_manifest_fields(), "export_mode", "exported_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in session.rows:
            record = preview_row_to_csv(row)
            record.update(
                {
                    "export_mode": mode,
                    "exported_path": str(exported_paths.get(row.row_id, "")),
                }
            )
            writer.writerow(record)


def write_corrections_csv(path: Path, session: SortPreviewSession) -> None:
    """Write only rows changed by the user."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=preview_manifest_fields())
        writer.writeheader()
        for row in session.rows:
            if row.is_corrected:
                writer.writerow(preview_row_to_csv(row))


def preview_manifest_fields() -> list[str]:
    """Return CSV fields shared by preview and correction reports."""
    return [
        "row_id",
        "source_path",
        "display_name",
        "proposed_folder",
        "approved_folder",
        "is_corrected",
        "final_top",
        "consensus_status",
        "confidence",
        "duration_sec",
        "read_status",
        "decision_reason",
        "diagnostic_summary",
        "candidate_folders_json",
        "neural_decision_state",
        "neural_runtime_status",
        "neural_runtime_message",
        "neural_row_error",
        "neural_folder",
        "neural_known_distribution",
        "neural_ownership_ready",
        "neural_ownership_reason",
        "neural_label_example_count",
        "neural_exact_training_match",
        "neural_similarity",
        "neural_margin",
        "neural_radius_ratio",
        "neural_semantic_status",
        "neural_semantic_family",
        "neural_semantic_score",
        "neural_semantic_margin",
        "neural_prompt_status",
        "neural_prompt_suggestions_json",
        "panns_status",
        "panns_model_id",
        "panns_events_json",
        "panns_support_score",
        "panns_contradiction_score",
        "panns_supporting_events_json",
        "panns_contradicting_events_json",
    ]


def preview_row_to_csv(row: PreviewRow) -> dict[str, str]:
    """Return one preview row as a CSV-safe mapping."""
    return {
        "row_id": row.row_id,
        "source_path": str(row.source_path),
        "display_name": row.display_name,
        "proposed_folder": row.proposed_folder,
        "approved_folder": row.approved_folder,
        "is_corrected": "1" if row.is_corrected else "0",
        "final_top": row.final_top,
        "consensus_status": row.consensus_status,
        "confidence": f"{row.confidence:.6f}",
        "duration_sec": f"{row.duration_sec:.6f}",
        "read_status": row.read_status,
        "decision_reason": row.decision_reason,
        "diagnostic_summary": row.diagnostic_summary,
        "candidate_folders_json": json.dumps(row.candidate_folders, sort_keys=True),
        "neural_decision_state": row.neural_decision_state,
        "neural_runtime_status": row.neural_runtime_status,
        "neural_runtime_message": row.neural_runtime_message,
        "neural_row_error": row.neural_row_error,
        "neural_folder": row.neural_folder,
        "neural_known_distribution": (
            "" if row.neural_known_distribution is None else ("1" if row.neural_known_distribution else "0")
        ),
        "neural_ownership_ready": (
            "" if row.neural_ownership_ready is None else ("1" if row.neural_ownership_ready else "0")
        ),
        "neural_ownership_reason": row.neural_ownership_reason,
        "neural_label_example_count": str(row.neural_label_example_count),
        "neural_exact_training_match": "1" if row.neural_exact_training_match else "0",
        "neural_similarity": f"{row.neural_similarity:.8f}",
        "neural_margin": f"{row.neural_margin:.8f}",
        "neural_radius_ratio": f"{row.neural_radius_ratio:.8f}",
        "neural_semantic_status": row.neural_semantic_status,
        "neural_semantic_family": row.neural_semantic_family,
        "neural_semantic_score": f"{row.neural_semantic_score:.8f}",
        "neural_semantic_margin": f"{row.neural_semantic_margin:.8f}",
        "neural_prompt_status": row.neural_prompt_status,
        "neural_prompt_suggestions_json": json.dumps(row.neural_prompt_suggestions, sort_keys=True),
        "panns_status": row.panns_status,
        "panns_model_id": row.panns_model_id,
        "panns_events_json": json.dumps(row.panns_events, sort_keys=True),
        "panns_support_score": f"{row.panns_support_score:.8f}",
        "panns_contradiction_score": f"{row.panns_contradiction_score:.8f}",
        "panns_supporting_events_json": json.dumps(row.panns_supporting_events, sort_keys=True),
        "panns_contradicting_events_json": json.dumps(row.panns_contradicting_events, sort_keys=True),
    }


def correction_evidence(row: PreviewRow) -> dict[str, Any]:
    """Return measured-evidence payload for one manual correction."""
    payload: dict[str, Any] = {
        "row_id": row.row_id,
        "source_path_for_audit_only": str(row.source_path),
        "proposed_folder": row.proposed_folder,
        "approved_folder": row.approved_folder,
        "duration_sec": row.duration_sec,
        "read_status": row.read_status,
        "consensus_status": row.consensus_status,
        "decision_reason": row.decision_reason,
        "neural_foundation_evidence": {
            "trained_memory_label": row.neural_folder,
            "trained_memory_similarity": row.neural_similarity,
            "trained_memory_margin": row.neural_margin,
            "trained_memory_exact_match": row.neural_exact_training_match,
            "clap_broad_family": row.neural_semantic_family,
            "clap_broad_score": row.neural_semantic_score,
            "clap_broad_margin": row.neural_semantic_margin,
            "clap_detailed_suggestions": row.neural_prompt_suggestions,
            "panns_events": row.panns_events,
            "panns_support_score": row.panns_support_score,
            "panns_contradiction_score": row.panns_contradiction_score,
            "source_name_policy": "audio waveform, content hash, and explicit human target only",
        },
    }
    if row.result is None:
        return payload
    evidence = row.result.facts.evidence if isinstance(row.result.facts.evidence, dict) else {}
    payload.update(
        {
            "feature_values_by_name": row.result.facts.feature_values_by_name,
            "shape_vote": evidence.get("shape_vote", {}),
            "physics_subpanels": evidence.get("physics_subpanels", {}),
            "brain_ensemble_vote_result": evidence.get("brain_ensemble_vote_result", {}),
            "physics_vote_result": evidence.get("physics_vote_result", voter_result_digest(row.result.physics_votes)),
            "authority_trace": row.result.decision.authority_trace,
            "shared_candidates": row.result.decision.shared_candidates,
        }
    )
    return payload


def timestamp() -> str:
    """Return a filesystem-friendly current timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def gui_worker_count() -> int:
    """Return the default worker count for desktop preview sorting."""
    raw = str(os.environ.get("AARON_GUI_SORT_WORKERS", "")).strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 1
    available_cpus = max(1, os.cpu_count() or 2)
    return max(1, min(10, available_cpus))


def gui_analysis_cache_enabled() -> bool:
    """Return whether GUI preview should reuse measured analysis across runs."""
    raw = str(os.environ.get("AARON_GUI_DISABLE_ANALYSIS_CACHE", "")).strip().lower()
    return raw not in {"1", "true", "yes", "on"}


def gui_analysis_cache_dir(project_root: Path) -> Path:
    """Return the GUI persistent measured-analysis cache directory."""
    configured = str(os.environ.get("AARON_GUI_ANALYSIS_CACHE_DIR", "")).strip()
    if configured:
        return Path(configured).expanduser()
    return Path(project_root).expanduser() / "_reports" / "analysis_cache" / "gui_v1"
