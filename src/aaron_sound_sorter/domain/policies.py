"""Visible policy objects for the clean product sorter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SharedFactsPolicy:
    """Thresholds used only to build broad structural facts."""

    tiny_duration_sec: float = 0.030
    long_duration_sec: float = 4.0
    loop_min_duration_sec: float = 1.20
    loop_min_event_count: float = 4.0
    loop_min_onset_span: float = 0.34
    loop_min_event_rate_hz: float = 0.35
    single_event_max_event_count: float = 2.25
    single_event_max_onset_span: float = 0.42
    short_hit_max_duration_sec: float = 1.60


@dataclass(frozen=True)
class BrainVoterPolicy:
    """Policy for nearest-brain category ranking."""

    top_n: int = 100
    structure_mismatch_penalty: float = 2.0


@dataclass(frozen=True)
class PhysicsVoterPolicy:
    """Policy for full-profile physics matching."""

    top_n: int = 100
    min_reliability: float = 0.15
    min_valid_count: int = 1
    structure_mismatch_penalty: float = 1.5
    severe_profile_penalty: float = 0.45
    minimum_scale: float = 0.035


@dataclass(frozen=True)
class ShapeVoterPolicy:
    """Policy for structural waveform-shape diagnostics."""

    top_n: int = 5
    min_confidence_for_consensus_sanity: float = 0.68
    max_shape_rescue_combined_rank_score: float = 40.0
    shape_conflict_label: str = "_TO_REVIEW/Shape Conflict"


@dataclass(frozen=True)
class ConsensusPolicy:
    """Policy for comparing BrainVoter and PhysicsVoter outputs."""

    top_n: int = 100
    max_combined_rank_score: float = 20.0
    role_sanity_min_strength: float = 0.70
    role_sanity_max_combined_rank_score: float = 32.0
    role_sanity_primary_voter_rank: int = 6
    role_sanity_secondary_voter_rank: int = 35
    no_consensus_label: str = "_TO_REVIEW/No Voter Consensus"
    weak_consensus_label: str = "_TO_REVIEW/No Strong Voter Consensus"
    role_conflict_label: str = "_TO_REVIEW/Measured Role Conflict"
    shape_conflict_label: str = "_TO_REVIEW/Shape Conflict"
    shape_sanity_min_confidence: float = 0.85
    shape_sanity_max_combined_rank_score: float = 40.0
    broken_or_tiny_label: str = "_TO_REVIEW/Broken Or Tiny"


@dataclass(frozen=True)
class SortPolicy:
    """Policy for user-facing product sort."""

    candidate_count: int = 100
    write_zip: bool = True
