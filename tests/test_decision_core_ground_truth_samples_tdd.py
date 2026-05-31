"""Ground-truth true-bucket regressions (updated from earlier review-only policy) from Aaron's listened samples.

These tests intentionally do not use filenames. They encode candidate/role
conflicts seen in the copied audio pack and require useful broad buckets when
there is enough evidence, not blanket review.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision


def raw(path: str, top: str, *, score: float = 10.0, candidates: list[dict] | None = None) -> ConsensusDecision:
    return ConsensusDecision(
        final_label=path,
        final_top=top,
        folder_path=path,
        consensus_status="strong_consensus",
        reason="synthetic raw decision",
        combined_rank_score=score,
        shared_candidates=candidates or [],
    )


def eligibility(role: str, top: tuple[str, ...], broad: str, *, confidence: float = 0.84) -> EligibilityDecision:
    return EligibilityDecision(
        role_name=role,
        confidence=confidence,
        allowed_top_families=top,
        blocked_path_fragments=(),
        broad_folder_path=broad,
        reason="synthetic eligibility",
    )


def unknown_eligibility() -> EligibilityDecision:
    return EligibilityDecision(role_name="unknown", confidence=0.0)


def candidate(path: str, score: float, *, top: str | None = None) -> dict:
    return {
        "folder_path": path,
        "label": path,
        "top_family": top or path.split("/", 1)[0],
        "combined_rank_score": score,
        "candidate_role_signature": {},
    }


def drum_loop_facts() -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "measured_roles": {
                "drum_loop": 0.88,
                "low_rhythmic_drum_loop": 0.76,
                "detected_parent_role": "drum_loop",
            },
            "shape_vote": {"primary_shape": "drum_loop", "confidence": 0.84},
        },
    )


def test_confirmed_kick_loop_reroutes_from_bass_loop_to_drum_loops() -> None:
    """Kick/drum loops with close drum candidates must land near Drum Loops, not Review."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Bass/Bass Loops",
            "Instruments",
            score=9,
            candidates=[
                candidate("Instruments/Bass/Bass Loops", 9),
                candidate("Drums/Drum Loops/Kick Focused Loops", 10),
                candidate("Drums/Kick Drums/Looped Kick Pattern", 11),
            ],
        ),
        eligibility("bass_loop", ("Instruments", "_TO_REVIEW"), "Instruments/Bass/Bass Loops"),
        facts=drum_loop_facts(),
    )
    assert final.final_top == "Drums"
    assert final.folder_path == "Drums/Drum Loops/Loops"
    assert "kick/drum-loop true-bucket rescue" in final.reason


def test_distorted_kick_loop_reroutes_from_generic_instruments_to_drum_loops() -> None:
    """Distorted kick loops can look pitched/noisy but should land near Drum Loops."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            "Instruments",
            score=8,
            candidates=[
                candidate("Instruments/Instrument Loops/Loops", 8),
                candidate("Drums/Kick Drums/Distorted Kick Loop", 11),
                candidate("Drums/Drum Loops/Loops", 12),
            ],
        ),
        eligibility("pitched_music_loop", ("Instruments", "_TO_REVIEW"), "Instruments/Instrument Loops/Loops"),
        facts=drum_loop_facts(),
    )
    assert final.final_top == "Drums"
    assert final.folder_path == "Drums/Drum Loops/Loops"


def test_confirmed_drum_break_reroutes_from_human_voice_fx_to_drum_loops() -> None:
    """Confirmed break loop must be rescued to Drum Loops when drum candidates are close."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Human and Voice FX/Breath/Long FX",
            "FX",
            score=9,
            candidates=[
                candidate("FX/Human and Voice FX/Breath/Long FX", 9),
                candidate("Drums/Drum Loops/Breaks", 10),
                candidate("Drums/Drum Loops/Loops", 12),
            ],
        ),
        unknown_eligibility(),
    )
    assert final.final_top == "Drums"
    assert final.folder_path == "Drums/Drum Loops/Loops"


def test_confirmed_keys_one_shot_reroutes_from_hybrid_fx_to_best_keys_candidate() -> None:
    """Brain+physics Keys/Rhodes agreement must not be overruled into FX or Review."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Hybrid Designed FX",
            "FX",
            score=16,
            candidates=[
                candidate("Instruments/Keys/Rhodes/One Shots", 5),
                candidate("Instruments/Keys/Electric Piano/One Shots", 6),
                candidate("FX/Hybrid Designed FX", 16),
            ],
        ),
        eligibility(
            "percussive_one_shot", ("Drums", "FX", "_TO_REVIEW"), "Drums/Percussion/Generic Percussion/One Shots"
        ),
    )
    assert final.final_top == "Instruments"
    assert final.folder_path == "Instruments/Keys/Rhodes/One Shots"
    assert "instrument true-bucket rescue" in final.reason


def test_confirmed_crazy_seagull_fx_reroutes_from_human_voice_to_best_fx_candidate() -> None:
    """Crazy/seagull FX should stay FX, but not Human/Voice Crowd."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Human and Voice FX/Crowd/One Shots",
            "FX",
            score=12,
            candidates=[
                candidate("FX/Human and Voice FX/Crowd/One Shots", 12),
                candidate("FX/Sweeps and Whooshes/Whoosh/One Shots", 9),
                candidate("FX/Biological Foley and Organic FX/Animals/Bird Call", 10),
            ],
        ),
        eligibility("voiced_one_shot", ("FX", "Instruments", "_TO_REVIEW"), "FX/Human and Voice FX"),
    )
    assert final.final_top == "FX"
    assert final.folder_path == "FX/Sweeps and Whooshes/Whoosh/One Shots"
    assert "non-voice FX true-bucket rescue" in final.reason


def test_growing_metallic_bell_fx_reroutes_from_generic_instruments_to_fx() -> None:
    """Growing bell FX is tonal, but should land near FX rather than generic Instruments."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            "Instruments",
            score=8,
            candidates=[
                candidate("Instruments/Instrument Loops/Loops", 8),
                candidate("FX/Structural and Transitional FX/Risers and Builds/Build", 10),
                candidate("FX/Hybrid Designed FX/Bell-Like FX/Long FX", 11),
            ],
        ),
        eligibility("pitched_music_loop", ("Instruments", "_TO_REVIEW"), "Instruments/Instrument Loops/Loops"),
    )
    assert final.final_top == "FX"
    assert final.folder_path == "FX/Structural and Transitional FX/Risers and Builds/Build"
    assert "tonal FX true-bucket rescue" in final.reason


def test_real_percussion_loop_reroutes_from_generic_instrument_to_drum_loops() -> None:
    """Confirmed percussion/metal percussion loops must not stay generic Instrument Loops."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            "Instruments",
            score=9,
            candidates=[
                candidate("Instruments/Instrument Loops/Loops", 9),
                candidate("Drums/Percussion/Bells and Metallic Percussion/Loops", 12),
                candidate("Drums/Drum Loops/Percussion Loops", 13),
            ],
        ),
        eligibility("pitched_music_loop", ("Instruments", "_TO_REVIEW"), "Instruments/Instrument Loops/Loops"),
        facts=drum_loop_facts(),
    )
    assert final.final_top == "Drums"
    assert final.folder_path == "Drums/Drum Loops/Loops"
    assert "percussion-loop true-bucket rescue" in final.reason


def test_human_like_flute_synth_loop_stays_generic_instrument_control() -> None:
    """Near-human flute/synth loop should not be swept into Voice, FX, or Review."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            "Instruments",
            score=8,
            candidates=[
                candidate("Instruments/Instrument Loops/Loops", 8),
                candidate("Instruments/Synths/Loops", 10),
                candidate("FX/Human and Voice FX/Crowd/Long FX", 20),
            ],
        ),
        eligibility("pitched_music_phrase", ("Instruments", "_TO_REVIEW"), "Instruments/Instrument Loops/Loops"),
    )
    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_reverb_keys_pluck_loop_stays_generic_instrument_control() -> None:
    """Warm low-mid keys/pluck loop must not be misrescued to Drums/Bass/FX."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            "Instruments",
            score=8,
            candidates=[
                candidate("Instruments/Instrument Loops/Loops", 8),
                candidate("Instruments/Keys/Piano and Keys/Loops", 9),
                candidate("Instruments/Bass/Bass Loops", 25),
            ],
        ),
        eligibility("pitched_music_loop", ("Instruments", "_TO_REVIEW"), "Instruments/Instrument Loops/Loops"),
    )
    assert final.folder_path == "Instruments/Instrument Loops/Loops"
