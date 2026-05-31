"""Regression guard for the remaining curated FX smoke review rows.

The curated FX_Aaron2 smoke pack is used as known-good coverage. These tests
encode the remaining review failures observed on 2026-05-15 without using
producer filenames or source folders. The resolver may choose broad parent
buckets, but it must not dump these clear measured structures into review.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision


def candidate(path: str, score: float, *, top: str | None = None) -> dict:
    return {
        "folder_path": path,
        "label": path,
        "top_family": top or path.split("/", 1)[0],
        "combined_rank_score": score,
        "candidate_role_signature": {},
    }


def raw(path: str, top: str, *, score: float = 9.0, candidates: list[dict] | None = None) -> ConsensusDecision:
    return ConsensusDecision(
        final_label=path,
        final_top=top,
        folder_path=path,
        consensus_status="strong_consensus",
        reason="synthetic raw decision",
        combined_rank_score=score,
        shared_candidates=candidates or [],
    )


def eligibility(role: str, allowed: tuple[str, ...], broad: str, *, confidence: float = 0.86) -> EligibilityDecision:
    return EligibilityDecision(
        role_name=role,
        confidence=confidence,
        allowed_top_families=allowed,
        blocked_path_fragments=(),
        broad_folder_path=broad,
        reason="synthetic eligibility",
    )


def facts(shape: str, confidence: float) -> SharedAudioFacts:
    evidence: dict[str, object] = {"shape_vote": shape, "shape_confidence": confidence}
    if shape == "vocal_phrase":
        evidence.update(
            {
                "physics_subpanels": {
                    "flat": {
                        "voice_score": 0.84,
                        "human_spoken_voice_score": 0.80,
                    }
                },
                "f0_voiced_ratio": 0.94,
                "percussive_event_ratio": 0.04,
                "drumlike_frame_ratio": 0.02,
            }
        )
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=shape != "vocal_phrase",
        is_single_event_like=shape == "vocal_phrase",
        is_short_hit_like=shape == "vocal_phrase",
        is_long=shape != "vocal_phrase",
        evidence=evidence,
    )


def apply(
    raw_decision: ConsensusDecision, elig: EligibilityDecision, shape: str, confidence: float
) -> ConsensusDecision:
    return DecisionCoreV2().apply_eligibility(raw_decision, elig, facts(shape, confidence))


def test_sax_like_pitched_loop_raw_fx_becomes_broad_instrument_loop_not_review() -> None:
    """Pitched musical loop structure should not become measured-conflict review."""
    final = apply(
        raw(
            "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
            "FX",
            candidates=[
                candidate("FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX", 8),
                candidate("Instruments/Instrument Loops/Loops", 10),
            ],
        ),
        eligibility("pitched_music_loop", ("Instruments", "_TO_REVIEW"), "_TO_REVIEW/Measured Role Conflict"),
        "pitched_phrase",
        0.91,
    )
    assert final.final_top == "Instruments"
    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_bass_phrase_pitched_loop_raw_wrong_instrument_becomes_instrument_loop_not_review() -> None:
    """Low sax/reed-like musical loop may be broad Instruments, not review."""
    final = apply(
        raw(
            "Instruments/Guitar/Nylon Guitar/One Shots",
            "Instruments",
            candidates=[
                candidate("Instruments/Guitar/Nylon Guitar/One Shots", 8),
                candidate("FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX", 9),
            ],
        ),
        eligibility("pitched_music_loop", ("Instruments", "_TO_REVIEW"), "_TO_REVIEW/Measured Role Conflict"),
        "bass_phrase",
        0.98,
    )
    assert final.final_top == "Instruments"
    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_drum_top_loop_raw_drop_becomes_drum_loop_not_review() -> None:
    """Measured drum/top loop structure should not sit in FX/drop review."""
    final = apply(
        raw(
            "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX",
            "FX",
            candidates=[
                candidate(
                    "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX", 8
                ),
                candidate("FX/Textures/Natural Ambience/Waves/Long FX", 10),
            ],
        ),
        eligibility("unknown", ("_TO_REVIEW",), "_TO_REVIEW/Measured Role Conflict", confidence=0.0),
        "top_loop",
        0.81,
    )
    assert final.final_top == "Drums"
    assert final.folder_path == "Drums/Drum Loops/Loops"


def test_beat_loop_raw_fx_becomes_drum_loop_not_review() -> None:
    """Measured beat loop structure should be a broad drum loop instead of review."""
    final = apply(
        raw(
            "FX/Textures/Natural Ambience/Waves/Long FX",
            "FX",
            candidates=[
                candidate("FX/Textures/Natural Ambience/Waves/Long FX", 8),
                candidate(
                    "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX", 9
                ),
            ],
        ),
        eligibility(
            "percussive_drum_loop", ("Drums", "_TO_REVIEW"), "_TO_REVIEW/Measured Role Conflict", confidence=0.82
        ),
        "beat_loop",
        0.90,
    )
    assert final.final_top == "Drums"
    assert final.folder_path == "Drums/Drum Loops/Loops"


def test_vocal_phrase_raw_riser_or_rim_conflict_becomes_instrument_voice_not_review() -> None:
    """Measured vocal phrase shape should route to Instruments/Voice, not FX."""
    final = apply(
        raw(
            "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
            "FX",
            candidates=[
                candidate("FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX", 8),
                candidate("Drums/Rims and Sticks/Rimshot/One Shots", 9),
            ],
        ),
        eligibility(
            "percussive_one_shot", ("Drums", "FX", "_TO_REVIEW"), "_TO_REVIEW/Measured Role Conflict", confidence=0.78
        ),
        "vocal_phrase",
        0.99,
    )
    assert final.final_top == "Instruments"
    assert final.folder_path == "Instruments/Voice/Phrase/One Shots"


def test_sustained_pad_conflict_becomes_instrument_loop_not_review() -> None:
    """Sustained pitched pad/loop structure should not become review."""
    final = apply(
        raw(
            "FX/Human and Voice FX/Breath/Long FX",
            "FX",
            candidates=[
                candidate("FX/Human and Voice FX/Breath/Long FX", 8),
                candidate("Instruments/Instrument Loops/Loops", 11),
            ],
        ),
        eligibility(
            "pitched_music_loop", ("Instruments", "_TO_REVIEW"), "_TO_REVIEW/Measured Role Conflict", confidence=0.82
        ),
        "sustained_pad",
        0.75,
    )
    assert final.final_top == "Instruments"
    assert final.folder_path == "Instruments/Instrument Loops/Loops"
