"""Core domain models for Aaron Sound Sorter.

These objects are intentionally small and plain. They are safe to inspect in a
breakpoint and are shared by the voters, consensus runner, and output writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AudioPhysics:
    """Measured physics for one audio file.

    Args:
        source_path: Original or staged audio file path.
        fingerprint: Full-file 1D NumPy vector created by the audio analyzer.
        duration_sec: Full-file audio duration in seconds after reading/gating.
        read_status: ``ok`` or a readable failure/status string.
        direct_body_fingerprint: Optional tail-reduced direct/body view. This
            view is computed from onset/peak-centered audio windows so voters can
            inspect the sound before reverb, delay, or long decay dominates the
            full-file fingerprint. It is analysis evidence only; it never
            replaces the full-file fingerprint destructively.
        direct_body_duration_sec: Duration of the direct/body analysis view.
        direct_body_status: Status string for the direct/body analysis view.
    """

    source_path: Path
    fingerprint: np.ndarray
    duration_sec: float
    read_status: str
    direct_body_fingerprint: np.ndarray | None = None
    direct_body_duration_sec: float = 0.0
    direct_body_status: str = "not_computed"
    third_party_feature_profile: dict[str, Any] | None = None


@dataclass(frozen=True)
class SharedAudioFacts:
    """Reusable measured facts all voters may use.

    The five boolean fields are broad structure helpers.  The full 100-feature
    fingerprint is also carried by name so voters, consensus, and reports can
    inspect the same complete evidence universe from the start of sorting.
    """

    is_broken_or_tiny: bool
    is_loop_like: bool
    is_single_event_like: bool
    is_short_hit_like: bool
    is_long: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    feature_values_by_name: dict[str, float] = field(default_factory=dict)
    feature_count: int = 0
    feature_vector: tuple[float, ...] = field(default_factory=tuple)
    feature_groups: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class CategoryGuess:
    """One ranked category/folder possibility from one voter."""

    label: str
    folder_path: str
    top_family: str
    score: float
    confidence: float
    rank: int
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VoterResult:
    """Standard result shape returned by every voter."""

    voter_name: str
    guesses: list[CategoryGuess]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConsensusDecision:
    """Final category decision made from two voter result lists."""

    final_label: str
    final_top: str
    folder_path: str
    consensus_status: str
    reason: str
    shared_winner: str = ""
    brain_rank: int | None = None
    physics_rank: int | None = None
    combined_rank_score: float | None = None
    shared_candidates: list[dict[str, Any]] = field(default_factory=list)
    authority_trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SortRequest:
    """User-facing sort request."""

    input_path: Path
    output_dir: Path
    brain_path: Path
    baby_brain_path: Path | None = None
    core_baby_brain_path: Path | None = None
    spread_baby_brain_path: Path | None = None
    outlier_baby_brain_path: Path | None = None
    user_memory_brain_path: Path | None = None
    physics_memory_brain_path: Path | None = None
    voter_memory_brain_path: Path | None = None
    shape_memory_brain_path: Path | None = None
    shape_starter_memory_brain_path: Path | None = None
    harmonic_core_baby_brain_path: Path | None = None
    harmonic_spread_baby_brain_path: Path | None = None
    harmonic_outlier_baby_brain_path: Path | None = None
    use_baby_brains_in_sort: bool = True
    use_harmonic_brains_in_sort: bool = False
    write_zip: bool = True
    candidate_count: int = 100
    sort_workers: int = 1


@dataclass(frozen=True)
class SortFileResult:
    """Sort result for one audio file."""

    source_path: Path
    placed_path: Path | None
    physics: AudioPhysics
    facts: SharedAudioFacts
    brain_votes: VoterResult
    physics_votes: VoterResult
    decision: ConsensusDecision


@dataclass(frozen=True)
class SortSummary:
    """Summary returned by the product sort use case."""

    output_root: Path
    manifest_path: Path
    summary_path: Path
    zip_path: Path | None
    file_results: list[SortFileResult]

    @property
    def processed_count(self) -> int:
        return len(self.file_results)

    @property
    def review_count(self) -> int:
        return sum(1 for result in self.file_results if result.decision.final_top == "_TO_REVIEW")
