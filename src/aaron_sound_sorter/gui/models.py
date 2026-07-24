"""Plain data models used by the desktop GUI.

These models deliberately sit outside classifier logic. They describe the
review/export plan that a human approves after the normal sorter has already
made its measured-evidence decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from aaron_sound_sorter.domain.models import SortFileResult

ExportMode = Literal["copy", "move", "symlink"]


@dataclass
class PreviewRow:
    """One GUI row representing a proposed placement.

    Args:
        row_id: Stable row identifier for the current preview session.
        source_path: Audio file used by the sorter. For ZIP inputs this may be
            an extracted preview-cache path.
        display_name: Human-facing basename for table display only.
        proposed_folder: Folder proposed by the sorter.
        approved_folder: Folder approved by the user. Defaults to the proposed
            folder until manually changed.
        final_top: Sorter's final top family for the proposed folder.
        consensus_status: Sorter decision status.
        confidence: Compact confidence display value from voter/decision
            evidence when available.
        duration_sec: Measured duration in seconds.
        read_status: Audio read status from feature extraction.
        decision_reason: Sorter decision reason.
        diagnostic_summary: Compact human-readable voter/shape summary.
        candidate_folders: Diagnostic candidate folders proposed by voters and
            final arbitration. These are displayed only as human override hints.
        neural_folder: Source-name-blind neural prototype prediction.
        neural_known_distribution: Whether the prediction falls inside a
            learned neighborhood. ``None`` means neural inference was absent.
        neural_similarity: Cosine similarity to the winning prototype.
        neural_margin: Similarity lead over the second label.
        neural_radius_ratio: Distance relative to the learned label radius.
        result: Full in-memory sort result used for correction evidence. Tests
            may provide ``None`` when only export behavior is under test.

    Side Effects:
        None.

    Important Constraints:
        ``display_name`` and ``source_path`` are for GUI display/export only and
        must not feed classification decisions.
    """

    row_id: str
    source_path: Path
    display_name: str
    proposed_folder: str
    approved_folder: str
    final_top: str
    consensus_status: str
    confidence: float
    duration_sec: float
    read_status: str
    decision_reason: str
    diagnostic_summary: str
    candidate_folders: list[str] = field(default_factory=list)
    neural_folder: str = ""
    neural_known_distribution: bool | None = None
    neural_similarity: float = 0.0
    neural_margin: float = 0.0
    neural_radius_ratio: float = 0.0
    result: SortFileResult | None = field(default=None, repr=False)

    @property
    def is_corrected(self) -> bool:
        """Return true when the approved placement differs from the sorter."""
        return normalize_folder(self.approved_folder) != normalize_folder(self.proposed_folder)


@dataclass(frozen=True)
class SortPreviewSession:
    """A complete in-memory GUI preview run.

    Args:
        run_dir: Report folder used for preview diagnostics.
        input_path: User-selected source folder, ZIP, or audio file.
        brain_path: Active brain file used for the preview.
        available_labels: Trained label folders loaded from the active brain.
        rows: Preview rows produced from classifier results.

    Side Effects:
        None.
    """

    run_dir: Path
    input_path: Path
    brain_path: Path
    available_labels: list[str]
    rows: list[PreviewRow]

    @property
    def corrected_count(self) -> int:
        """Return the number of rows with manual corrections."""
        return sum(1 for row in self.rows if row.is_corrected)


@dataclass(frozen=True)
class ExportSummary:
    """Summary of an approved GUI export run.

    Args:
        destination_root: User-selected destination folder.
        sorted_root: Root folder containing approved sorted files.
        approved_plan_path: CSV with every approved placement.
        corrections_path: CSV with only manual corrections.
        correction_evidence_path: Optional JSONL evidence pack path.
        exported_count: Number of files exported successfully.
        corrected_count: Number of manual corrections exported.
        errors: Non-fatal export errors.

    Side Effects:
        None.
    """

    destination_root: Path
    sorted_root: Path
    approved_plan_path: Path
    corrections_path: Path
    correction_evidence_path: Path | None
    exported_count: int
    corrected_count: int
    errors: list[str]


@dataclass(frozen=True)
class TrainingImportSummary:
    """Summary of GUI corrections imported into the trusted training tree.

    Args:
        training_root: Curated training tree receiving corrected audio.
        report_dir: Report folder containing import artifacts.
        manifest_path: CSV manifest for staged and skipped corrections.
        correction_evidence_path: JSONL evidence pack for the same corrections.
        staged_count: Number of corrected audio files copied into training.
        reused_existing_count: Number of byte-identical corrected files already
            present in the approved training slot. These rows are trainable.
        skipped_count: Number of corrected rows skipped with a reason.
        staged_paths: Training-file paths created by the import.
        neural_intake_path: Durable current neural-training inbox.
        neural_event_log_path: Append-only neural approval ledger.
        neural_queued_count: New or relabeled neural examples.
        neural_reaffirmed_count: Existing content/label approvals repeated.
        neural_superseded_count: Older labels replaced by newer approvals.
        errors: Non-fatal per-row import errors.

    Side Effects:
        None. The importer that creates this summary performs filesystem work.
    """

    training_root: Path
    report_dir: Path
    manifest_path: Path
    correction_evidence_path: Path
    staged_count: int
    reused_existing_count: int
    skipped_count: int
    staged_paths: list[Path]
    neural_intake_path: Path
    neural_event_log_path: Path
    neural_queued_count: int
    neural_reaffirmed_count: int
    neural_superseded_count: int
    errors: list[str]


def normalize_folder(folder_path: str) -> str:
    """Normalize a folder label for correction comparison."""
    return "/".join(part.strip() for part in str(folder_path or "").replace("\\", "/").split("/") if part.strip())
