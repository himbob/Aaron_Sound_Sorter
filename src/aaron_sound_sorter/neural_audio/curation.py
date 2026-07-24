"""Provenance-aware neural corpus curation contracts.

These helpers operate on explicit supervised labels and content hashes. They
never infer acoustic identity from an input file or folder name.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace

TRAINING_USE = "prototype_training"
VALIDATION_USE = "prototype_validation"
CALIBRATION_USE = "confidence_calibration"
FINAL_HELDOUT_USE = "final_heldout_evaluation"
EXCLUDED_DUPLICATE_USE = "excluded_duplicate"
EXCLUDED_CONFLICT_USE = "excluded_label_conflict"
EXCLUDED_PENDING_REVIEW_USE = "excluded_pending_human_review"


@dataclass(frozen=True)
class ProvenanceCandidate:
    """One labeled audio occurrence considered for the neural corpus.

    Args:
        candidate_id: Stable occurrence identifier.
        audio_path: Location used only to read or display the audio.
        intended_label: Explicit supervised taxonomy target.
        label_source: Human-readable provenance for the target.
        source_kind: Stable provenance class used by trust policy.
        source_group: Source panel or pack identity used for coverage reports.
        human_approved: Whether the intended label has explicit human approval.
        file_sha256: Byte digest.
        decoded_audio_sha256: Canonical decoded-audio digest.
        normalized_audio_sha256: Gain/silence-normalized derived-copy digest.
        contamination_status: Existing trainer-audit status, when available.
        contamination_warnings: Audit evidence attached to the occurrence.
        embedding_providers: Frozen encoders already cached for this audio.
        allowed_use: Corpus partition or exclusion decision.

    Important Constraints:
        ``audio_path`` and ``source_group`` are provenance/display metadata.
        They must never become classifier features.
    """

    candidate_id: str
    audio_path: str
    intended_label: str
    label_source: str
    source_kind: str
    source_group: str
    human_approved: bool
    file_sha256: str
    decoded_audio_sha256: str
    normalized_audio_sha256: str
    contamination_status: str = ""
    contamination_warnings: str = ""
    embedding_providers: tuple[str, ...] = ()
    allowed_use: str = EXCLUDED_PENDING_REVIEW_USE

    @property
    def duplicate_group_id(self) -> str:
        """Return the strongest available content-derived duplicate identity."""
        identity = self.normalized_audio_sha256 or self.decoded_audio_sha256 or self.file_sha256
        stable_group = hashlib.sha256(identity.encode()).hexdigest()[:20]
        return f"audio_group_{stable_group}"


def assign_leakage_safe_uses(candidates: Iterable[ProvenanceCandidate]) -> tuple[ProvenanceCandidate, ...]:
    """Assign trusted content groups to four non-overlapping corpus partitions.

    Recent GUI corrections remain training-only because they were collected
    through active review. Explicit locked-panel examples can enter validation,
    calibration, and final-heldout partitions when a label has enough distinct
    duplicate groups. Untrusted legacy trainers remain excluded.

    Args:
        candidates: Candidate occurrences with hashes and provenance.

    Returns:
        Stable candidates carrying partition or exclusion decisions.

    Side Effects:
        None.
    """
    rows = list(candidates)
    by_group: dict[str, list[ProvenanceCandidate]] = defaultdict(list)
    for row in rows:
        by_group[row.duplicate_group_id].append(row)

    decisions: dict[str, str] = {}
    trusted_representatives: list[ProvenanceCandidate] = []
    for _group_id, group_rows in sorted(by_group.items()):
        trusted = [row for row in group_rows if row.human_approved]
        if not trusted:
            for row in group_rows:
                decisions[row.candidate_id] = EXCLUDED_PENDING_REVIEW_USE
            continue
        labels = {row.intended_label for row in trusted}
        if len(labels) > 1:
            for row in group_rows:
                decisions[row.candidate_id] = EXCLUDED_CONFLICT_USE
            continue
        trusted.sort(key=_representative_sort_key)
        representative = trusted[0]
        trusted_representatives.append(representative)
        for row in group_rows:
            if row.candidate_id != representative.candidate_id:
                decisions[row.candidate_id] = EXCLUDED_DUPLICATE_USE

    by_label: dict[str, list[ProvenanceCandidate]] = defaultdict(list)
    for row in trusted_representatives:
        by_label[row.intended_label].append(row)

    for _label, label_rows in sorted(by_label.items()):
        forced_training = [row for row in label_rows if row.source_kind == "recent_gui_correction"]
        splittable = [row for row in label_rows if row.source_kind != "recent_gui_correction"]
        for row in forced_training:
            decisions[row.candidate_id] = TRAINING_USE
        ranked = sorted(splittable, key=_partition_sort_key)
        partition_by_id = _partitions_for_label(ranked)
        decisions.update(partition_by_id)

    return tuple(replace(row, allowed_use=decisions.get(row.candidate_id, row.allowed_use)) for row in rows)


def _representative_sort_key(row: ProvenanceCandidate) -> tuple[int, str]:
    source_priority = {
        "recent_gui_correction": 0,
        "locked_human_seed": 1,
    }.get(row.source_kind, 9)
    return source_priority, row.candidate_id


def _partition_sort_key(row: ProvenanceCandidate) -> tuple[str, str]:
    rank = hashlib.sha256(f"aaron-neural-four-way-v1\0{row.duplicate_group_id}".encode()).hexdigest()
    return rank, row.candidate_id


def _partitions_for_label(rows: list[ProvenanceCandidate]) -> dict[str, str]:
    decisions = {row.candidate_id: TRAINING_USE for row in rows}
    count = len(rows)
    if count >= 4:
        decisions[rows[-1].candidate_id] = VALIDATION_USE
    if count >= 5:
        decisions[rows[-2].candidate_id] = CALIBRATION_USE
    if count >= 6:
        decisions[rows[-3].candidate_id] = FINAL_HELDOUT_USE
    return decisions


def readiness_tier(
    *,
    trusted_duplicate_groups: int,
    trusted_source_groups: int,
    final_heldout_examples: int,
) -> tuple[str, str]:
    """Return an evidence-conservative category readiness tier and reason."""
    if trusted_duplicate_groups >= 8 and trusted_source_groups >= 2 and final_heldout_examples >= 1:
        return "A", "enough distinct trusted evidence for a preliminary held-out evaluation"
    if trusted_duplicate_groups >= 4:
        return "B", "enough trusted evidence to train, but held-out or source diversity is not reliable"
    if trusted_duplicate_groups >= 2:
        return "C", "prototype-only evidence; do not claim reliable evaluation"
    return "D", "no trustworthy trainable set or only one trusted audio group"
