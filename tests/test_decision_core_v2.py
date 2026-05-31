"""Unit tests for DecisionCoreV2 parent eligibility behavior."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision


def make_raw(path: str, top: str = "FX") -> ConsensusDecision:
    """Build a minimal raw decision."""
    return ConsensusDecision(
        final_label=path,
        final_top=top,
        folder_path=path,
        consensus_status="strong_consensus",
        reason="test raw decision",
    )


def test_decision_core_keeps_eligible_raw_decision() -> None:
    eligibility = EligibilityDecision(
        role_name="drum_loop",
        confidence=0.8,
        allowed_top_families=("Drums", "_TO_REVIEW"),
        blocked_path_fragments=("FX",),
        broad_folder_path="Drums/Drum Loops/Loops",
    )
    raw = make_raw("Drums/Drum Loops/Loops", "Drums")
    final = DecisionCoreV2().apply_eligibility(raw, eligibility)
    assert final.folder_path == raw.folder_path


def test_decision_core_broadens_incompatible_raw_decision() -> None:
    eligibility = EligibilityDecision(
        role_name="vocal_one_shot",
        confidence=0.8,
        allowed_top_families=("FX", "Instruments", "_TO_REVIEW"),
        blocked_path_fragments=("Dog", "Bird", "Animals"),
        broad_folder_path="FX/Human and Voice FX",
    )
    raw = make_raw("FX/Animals and Creatures/Dog/One Shots", "FX")
    final = DecisionCoreV2().apply_eligibility(raw, eligibility)
    assert final.folder_path == raw.folder_path


def test_decision_core_ignores_non_decisive_eligibility() -> None:
    eligibility = EligibilityDecision(
        role_name="unknown",
        confidence=0.0,
        broad_folder_path="_TO_REVIEW/Measured Role Conflict",
    )
    raw = make_raw("FX/Animals and Creatures/Dog/One Shots", "FX")
    final = DecisionCoreV2().apply_eligibility(raw, eligibility)
    assert final.folder_path == raw.folder_path


def test_decision_core_broadens_ambiguous_fx_percussive_hit_to_safe_drum_bucket() -> None:
    raw = ConsensusDecision(
        final_label="FX/Everyday Foley/Keys Coins and Small Objects/Keys/One Shots",
        final_top="FX",
        folder_path="FX/Everyday Foley/Keys Coins and Small Objects/Keys/One Shots",
        consensus_status="strong_consensus",
        reason="raw shared FX leaf",
        combined_rank_score=10,
        shared_candidates=[
            {
                "folder_path": "FX/Everyday Foley/Keys Coins and Small Objects/Keys/One Shots",
                "combined_rank_score": 10,
                "candidate_role_signature": {"percussive_one_shot": 0.89},
            },
            {
                "folder_path": "Drums/Percussion/Generic Percussion/One Shots",
                "combined_rank_score": 14,
                "candidate_role_signature": {"percussive_one_shot": 0.97},
            },
        ],
    )
    eligibility = EligibilityDecision(
        role_name="percussive_one_shot",
        confidence=0.77,
        allowed_top_families=("Drums", "FX", "_TO_REVIEW"),
        blocked_path_fragments=("Drum Loops", "Long FX"),
        broad_folder_path="Drums/Percussion/One Shots",
    )
    final = DecisionCoreV2().apply_eligibility(raw, eligibility)
    assert final.folder_path == "Drums/Percussion/Generic Percussion/One Shots"
    assert final.consensus_status == "parent_eligibility_broad_bucket"
    assert "ambiguous FX one-shot" in final.reason


def test_decision_core_keeps_fx_percussive_hit_without_comparable_drum_support() -> None:
    raw = ConsensusDecision(
        final_label="FX/Everyday Foley/Keys Coins and Small Objects/Keys/One Shots",
        final_top="FX",
        folder_path="FX/Everyday Foley/Keys Coins and Small Objects/Keys/One Shots",
        consensus_status="strong_consensus",
        reason="raw shared FX leaf",
        combined_rank_score=10,
        shared_candidates=[
            {
                "folder_path": "Drums/Percussion/Generic Percussion/One Shots",
                "combined_rank_score": 30,
                "candidate_role_signature": {"percussive_one_shot": 0.97},
            },
        ],
    )
    eligibility = EligibilityDecision(
        role_name="percussive_one_shot",
        confidence=0.77,
        allowed_top_families=("Drums", "FX", "_TO_REVIEW"),
        blocked_path_fragments=("Drum Loops", "Long FX"),
        broad_folder_path="Drums/Percussion/One Shots",
    )
    final = DecisionCoreV2().apply_eligibility(raw, eligibility)
    assert final.folder_path == raw.folder_path


def test_decision_core_broadens_weak_review_when_parent_role_is_decisive() -> None:
    """A confident broad role may rescue weak no-consensus review to broad parent."""
    raw = ConsensusDecision(
        final_label="_TO_REVIEW/No Strong Voter Consensus",
        final_top="_TO_REVIEW",
        folder_path="_TO_REVIEW/No Strong Voter Consensus",
        consensus_status="review",
        reason="best shared category existed but was too weak: combined_rank_score=21.0, limit=20.0",
    )
    eligibility = EligibilityDecision(
        role_name="pitched_reed_or_instrument_phrase",
        confidence=0.76,
        allowed_top_families=("Instruments", "_TO_REVIEW"),
        blocked_path_fragments=("Drums", "FX", "Human", "Voice"),
        broad_folder_path="Instruments/Instrument Loops/Loops",
        reason="measured clean tonal reed/instrument phrase",
    )
    final = DecisionCoreV2().apply_eligibility(raw, eligibility)
    assert final.folder_path == raw.folder_path
    assert final.final_top == "_TO_REVIEW"


def test_decision_core_does_not_broaden_integrity_review() -> None:
    """Integrity/broken/tiny reviews stay review even with eligibility present."""
    raw = ConsensusDecision(
        final_label="_TO_REVIEW/Too Short or Broken",
        final_top="_TO_REVIEW",
        folder_path="_TO_REVIEW/Too Short or Broken",
        consensus_status="review",
        reason="integrity failure: too short or broken",
    )
    eligibility = EligibilityDecision(
        role_name="pitched_reed_or_instrument_phrase",
        confidence=0.90,
        allowed_top_families=("Instruments", "_TO_REVIEW"),
        broad_folder_path="Instruments/Instrument Loops/Loops",
    )
    final = DecisionCoreV2().apply_eligibility(raw, eligibility)
    assert final.folder_path == raw.folder_path


def test_decision_core_broadens_vocal_phrase_out_of_generic_instrument_bucket() -> None:
    """Measured vocal roles should not stay in generic synth/instrument buckets."""
    raw = make_raw("Instruments/Synths/Synth Chord/One Shots", "Instruments")
    eligibility = EligibilityDecision(
        role_name="vocal_phrase",
        confidence=0.82,
        allowed_top_families=("FX", "Instruments", "_TO_REVIEW"),
        blocked_path_fragments=("Drums", "Animals", "Machines"),
        broad_folder_path="FX/Human and Voice FX",
    )
    final = DecisionCoreV2().apply_eligibility(raw, eligibility)
    assert final.folder_path == raw.folder_path
    assert "vocal" not in final.folder_path.lower() or "voice" in final.folder_path.lower()


def test_decision_core_keeps_existing_voice_bucket_for_vocal_phrase() -> None:
    """Already-safe voice buckets are left alone."""
    raw = make_raw("Instruments/Voice/Choir/One Shots", "Instruments")
    eligibility = EligibilityDecision(
        role_name="vocal_phrase",
        confidence=0.82,
        allowed_top_families=("FX", "Instruments", "_TO_REVIEW"),
        broad_folder_path="FX/Human and Voice FX",
    )
    final = DecisionCoreV2().apply_eligibility(raw, eligibility)
    assert final.folder_path == raw.folder_path


def test_decision_core_recovers_voice_when_voice_candidate_beats_generic_loop() -> None:
    """A stronger voice candidate must recover to Instruments/Voice, not FX."""
    raw = ConsensusDecision(
        final_label="Instruments/Instrument Loops/Loops",
        final_top="Instruments",
        folder_path="Instruments/Instrument Loops/Loops",
        consensus_status="parent_eligibility_broad_bucket",
        reason="generic loop fallback",
        combined_rank_score=26,
        shared_candidates=[
            {"folder_path": "FX/Human and Voice FX/Spoken Voice/Long FX", "combined_rank_score": 6},
            {"folder_path": "Instruments/Instrument Loops/Loops", "combined_rank_score": 26},
        ],
    )
    eligibility = EligibilityDecision(
        role_name="vocal_phrase",
        confidence=0.80,
        allowed_top_families=("FX", "Instruments", "_TO_REVIEW"),
        broad_folder_path="FX/Human and Voice FX",
    )
    final = DecisionCoreV2().apply_eligibility(raw, eligibility)
    assert final.folder_path == "Instruments/Voice/Phrase/One Shots"
    assert "vocal role" in final.reason or "Human/Voice" in final.reason


def test_decision_core_does_not_voice_rescue_when_voice_candidate_is_not_stronger() -> None:
    """Sax/reed loops with weak voice candidates must stay in instruments."""
    raw = ConsensusDecision(
        final_label="Instruments/Instrument Loops/Loops",
        final_top="Instruments",
        folder_path="Instruments/Instrument Loops/Loops",
        consensus_status="strong_consensus",
        reason="generic loop winner",
        combined_rank_score=10,
        shared_candidates=[
            {"folder_path": "Instruments/Instrument Loops/Loops", "combined_rank_score": 10},
            {"folder_path": "FX/Human and Voice FX/Spoken Voice/Long FX", "combined_rank_score": 26},
        ],
    )
    eligibility = EligibilityDecision(
        role_name="pitched_reed_or_instrument_loop",
        confidence=0.82,
        allowed_top_families=("Instruments", "_TO_REVIEW"),
        broad_folder_path="Instruments/Brass and Woodwinds/Loops",
    )
    final = DecisionCoreV2().apply_eligibility(raw, eligibility)
    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_decision_core_keeps_concrete_fx_when_generic_instrument_loop_has_no_close_candidate() -> None:
    """Generic Instrument Loops must not replace or review strong concrete FX without support."""
    raw = ConsensusDecision(
        final_label="FX/Designed Noise FX/Alarm/Long FX",
        final_top="FX",
        folder_path="FX/Designed Noise FX/Alarm/Long FX",
        consensus_status="strong_consensus",
        reason="raw FX alarm winner",
        combined_rank_score=3.0,
        shared_candidates=[
            {"folder_path": "FX/Designed Noise FX/Alarm/Long FX", "combined_rank_score": 3.0},
            {"folder_path": "Instruments/Instrument Loops/Loops", "combined_rank_score": 29.0},
        ],
    )
    eligibility = EligibilityDecision(
        role_name="pitched_music_loop",
        confidence=0.94,
        allowed_top_families=("Instruments", "_TO_REVIEW"),
        blocked_path_fragments=("FX", "Animals", "Human", "Voice"),
        broad_folder_path="Instruments/Instrument Loops/Loops",
        reason="measured sustained pitched musical phrase/loop",
    )
    final = DecisionCoreV2().apply_eligibility(raw, eligibility)
    assert final.final_top == "FX"
    assert final.folder_path == "FX/Designed Noise FX/Alarm/Long FX"
