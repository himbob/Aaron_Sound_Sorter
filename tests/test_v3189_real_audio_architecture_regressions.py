"""v31.89 architecture-safe regressions from real FX one-file checks.

The real samples were inspected by filename only after sorting.  Production code
must solve these from measured facts, claims, and the arbiter, not source names.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path


def candidate(path: str, score: float) -> dict:
    """Build a synthetic shared candidate row."""
    return {
        "folder_path": path,
        "label": path,
        "top_family": path.split("/", 1)[0],
        "combined_rank_score": score,
        "candidate_role_signature": {},
        "brain_evidence": {"candidate_role_signature": {}},
        "physics_evidence": {"candidate_role_signature": {}},
    }


def raw(path: str, score: float, candidates: list[dict]) -> ConsensusDecision:
    """Build a raw consensus decision."""
    return ConsensusDecision(
        final_label=path,
        final_top=path.split("/", 1)[0],
        folder_path=path,
        consensus_status="strong_consensus",
        reason="synthetic raw decision",
        combined_rank_score=score,
        shared_candidates=candidates,
    )


def loop_facts(
    *,
    shape: str = "vocal_phrase",
    confidence: float = 1.0,
    pitched: float = 1.0,
    percussive: float = 0.0,
    drumlike: float = 0.0,
    pitch_confidence: float = 0.90,
    sustain: float = 0.85,
    onset_count: float = 16.0,
) -> SharedAudioFacts:
    """Build measured facts for loop/phrase architecture tests."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": shape,
                "confidence": confidence,
                "pitched_event_ratio": pitched,
                "percussive_event_ratio": percussive,
                "drumlike_frame_ratio": drumlike,
                "pitch_confidence": pitch_confidence,
                "sustain_ratio": sustain,
                "onset_count": onset_count,
            },
            "measured_roles": {
                "pitched_music_loop": pitched,
                "pitched_music_phrase": max(0.0, pitched - 0.10),
                "vocal_music_phrase": 0.0,
                "voiced_one_shot": 0.0,
            },
        },
    )


def test_decisive_pitched_loop_can_beat_concrete_fx_riser_false_positive() -> None:
    """A transition-shaped pitched loop stays in transition FX when evidence is mixed."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
            3.0,
            [
                candidate("FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX", 3.0),
                candidate("Instruments/Instrument Loops/Loops", 13.0),
            ],
        ),
        EligibilityDecision(
            role_name="pitched_reed_or_instrument_loop",
            confidence=0.80,
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=("FX", "Human", "Voice"),
            broad_folder_path="Instruments/Instrument Loops/Loops",
            reason="synthetic measured sustained pitched loop",
        ),
        loop_facts(shape="transition_drop", confidence=0.80, pitched=0.93, pitch_confidence=0.80),
    )

    assert (
        final.folder_path
        == "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX"
    )


def test_mixed_music_loop_can_escape_ambiguous_coin_leaf_without_name_evidence() -> None:
    """Choppy music-like loops should not stay in Coins just because raw shared there."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Everyday Foley/Keys Coins and Small Objects/Coins/Long FX",
            15.0,
            [
                candidate("FX/Everyday Foley/Keys Coins and Small Objects/Coins/Long FX", 15.0),
                candidate("Instruments/Instrument Loops/Loops", 142.0),
            ],
        ),
        EligibilityDecision(
            role_name="mixed_music_loop",
            confidence=0.62,
            allowed_top_families=("Instruments", "Drums", "_TO_REVIEW"),
            blocked_path_fragments=("Coins", "Keys Coins", "Animals"),
            broad_folder_path="Instruments/Mixed Musical Loops",
            reason="synthetic long choppy music-like loop",
        ),
        loop_facts(shape="transition_riser", confidence=0.72, pitched=0.56, percussive=0.44, pitch_confidence=0.33),
    )

    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_tonal_conflict_loop_goes_to_review_without_candidate_support() -> None:
    """High-tonal conflict loops review when no safe instrument candidate exists."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
            5.0,
            [candidate("FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX", 5.0)],
        ),
        EligibilityDecision(
            role_name="pitched_percussion_conflict_loop",
            confidence=0.74,
            allowed_top_families=("Drums", "Instruments", "_TO_REVIEW"),
            blocked_path_fragments=("FX", "Human", "Voice"),
            broad_folder_path="_TO_REVIEW/Measured Role Conflict",
            reason="synthetic pitched percussion conflict",
        ),
        loop_facts(pitched=1.0, percussive=0.0, drumlike=0.0, pitch_confidence=0.82, sustain=0.87),
    )

    assert final.folder_path == "_TO_REVIEW/Measured Role Conflict"


def test_loop_structure_broad_bucket_beats_specific_instrument_one_shot_leaf() -> None:
    """A loop/phrase claim should beat a Rhodes/Guitar/etc one-shot leaf."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Keys/Rhodes/One Shots",
            14.0,
            [candidate("Instruments/Keys/Rhodes/One Shots", 14.0)],
        ),
        EligibilityDecision(
            role_name="pitched_reed_or_instrument_loop",
            confidence=1.0,
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=("Drums", "FX"),
            broad_folder_path="Instruments/Instrument Loops/Loops",
            reason="synthetic measured loop structure",
        ),
        loop_facts(shape="pitched_phrase", pitched=1.0, pitch_confidence=0.98, sustain=0.95),
    )

    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_specific_fx_raw_beats_unbacked_parent_instrument_loop_claim() -> None:
    """Parent loop claims need architecture-safe evidence before beating concrete FX."""
    raw_claim = claim_from_folder_path(
        folder_path="FX/Impacts and Hits/Short Impact/Long FX",
        source="strong_consensus",
        reason="synthetic raw",
        shared=[],
        raw_candidate_score=8.0,
        brain_rank=4,
        physics_rank=4,
        shared_winner="FX/Impacts and Hits/Short Impact/Long FX",
        can_override=False,
        strength=0.50,
        is_real_candidate=True,
    )
    bass_claim = claim_from_folder_path(
        folder_path="Instruments/Bass/Bass Loops",
        source="role_sanity_broad_bucket",
        reason="synthetic inferred bass",
        shared=[],
        raw_candidate_score=None,
        brain_rank=None,
        physics_rank=None,
        shared_winner="Instruments/Bass/Bass Loops",
        can_override=True,
        strength=0.997,
        is_real_candidate=False,
    )
    loop_claim = claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="parent_eligibility_broad_bucket",
        reason="synthetic parent music loop",
        shared=[],
        raw_candidate_score=9.0,
        brain_rank=None,
        physics_rank=None,
        shared_winner="Instruments/Instrument Loops/Loops",
        can_override=True,
        strength=0.997,
        is_real_candidate=True,
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw_claim, claims=[bass_claim, loop_claim])

    assert winner.folder_path == "FX/Impacts and Hits/Short Impact/Long FX"


def test_repeated_tonal_hit_leaf_broadens_to_brass_woodwind_parent() -> None:
    """Short repeated brass-like phrases broaden to a safer family parent."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Brass/Trumpet/One Shots",
            3.0,
            [candidate("Instruments/Brass/Trumpet/One Shots", 3.0)],
        ),
        EligibilityDecision(
            role_name="pitched_reed_or_instrument_phrase",
            confidence=0.70,
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=("Drums", "FX"),
            broad_folder_path="Instruments/Instrument Loops/Loops",
            reason="synthetic repeated tonal hit phrase",
        ),
        loop_facts(
            shape="hit_with_tail",
            confidence=0.714,
            pitched=1.0,
            percussive=0.0,
            pitch_confidence=0.59,
            sustain=0.95,
            onset_count=3.0,
        ),
    )

    assert final.folder_path == "Instruments/Brass and Woodwinds/Loops"


def test_review_beats_ambiguous_fx_when_broad_loop_is_not_decisive() -> None:
    """Review wins when broad Instrument Loop evidence is still ambiguous."""
    raw_claim = claim_from_folder_path(
        folder_path="FX/Designed Noise FX/Alarm/Long FX",
        source="strong_consensus",
        reason="synthetic ambiguous FX raw",
        shared=[],
        raw_candidate_score=9.0,
        brain_rank=3,
        physics_rank=3,
        shared_winner="FX/Designed Noise FX/Alarm/Long FX",
        can_override=False,
        strength=0.44,
        is_real_candidate=True,
    )
    broad_loop = claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="role_sanity_generic_pitched_broad_instrument_loop",
        reason="generic pitched/mixed loop chose broad parent",
        shared=[],
        raw_candidate_score=14.0,
        brain_rank=8,
        physics_rank=1,
        shared_winner="Instruments/Instrument Loops/Loops",
        can_override=True,
        strength=0.88,
        is_real_candidate=True,
    )
    review = claim_from_folder_path(
        folder_path="_TO_REVIEW/Measured Role Conflict",
        source="parent_eligibility_review",
        reason="synthetic conservative review",
        shared=[],
        raw_candidate_score=None,
        brain_rank=None,
        physics_rank=None,
        shared_winner="_TO_REVIEW/Measured Role Conflict",
        can_override=True,
        strength=0.74,
        is_real_candidate=False,
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw_claim, claims=[broad_loop, review])

    assert winner.folder_path == "_TO_REVIEW/Measured Role Conflict"


def test_baby_reed_claim_does_not_beat_stronger_broad_instrument_loop_shape_claim() -> None:
    """Baby-brain brass/woodwind recall must not steal generic pitched synth/key loops."""
    raw_claim = claim_from_folder_path(
        folder_path="FX/Designed Noise FX/Siren/Long FX",
        source="strong_consensus",
        reason="synthetic ambiguous FX raw",
        shared=[],
        raw_candidate_score=8.0,
        brain_rank=4,
        physics_rank=4,
        shared_winner="FX/Designed Noise FX/Siren/Long FX",
        can_override=False,
        strength=0.56,
        is_real_candidate=True,
    )
    broad_loop = claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="shape_sanity_consensus",
        reason="measured pitched phrase chose broad Instrument Loops",
        shared=[],
        raw_candidate_score=9.0,
        brain_rank=7,
        physics_rank=1,
        shared_winner="Instruments/Instrument Loops/Loops",
        can_override=True,
        strength=0.99,
        is_real_candidate=True,
    )
    baby_reed = claim_from_folder_path(
        folder_path="Instruments/Brass and Woodwinds/Loops",
        source="baby_recall_brass_woodwind_claim",
        reason="synthetic baby recall reed family",
        shared=[],
        raw_candidate_score=5.0,
        brain_rank=2,
        physics_rank=8,
        shared_winner="Instruments/Brass and Woodwinds/Loops",
        can_override=True,
        strength=0.94,
        is_real_candidate=True,
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw_claim, claims=[broad_loop, baby_reed])

    assert winner.folder_path == "FX/Designed Noise FX/Siren/Long FX"
