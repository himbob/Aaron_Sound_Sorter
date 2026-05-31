"""Synthetic regressions from FX_Aaron2 after-test label inspection.

These tests do not use source names as sorter evidence. They encode the broad
failure patterns found by inspecting strongly labeled FX_Aaron2 files after a
sort run: sax/vocal loops being rescued into drum loops, Rhodes/key loops being
sent to review as short hits, and voice material being rejected even when the
measured shape is vocal.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision


def candidate(path: str, score: float, *, role_strengths: dict[str, float] | None = None) -> dict:
    """Build a ranked synthetic candidate row."""
    return {
        "folder_path": path,
        "label": path,
        "top_family": path.split("/", 1)[0],
        "combined_rank_score": score,
        "candidate_role_signature": role_strengths or {},
    }


def raw(path: str, *, score: float = 9.0, candidates: list[dict] | None = None) -> ConsensusDecision:
    """Build a raw consensus decision for the policy seam."""
    return ConsensusDecision(
        final_label=path,
        final_top=path.split("/", 1)[0],
        folder_path=path,
        consensus_status="strong_consensus",
        reason="synthetic raw decision",
        combined_rank_score=score,
        shared_candidates=candidates or [],
    )


def eligibility(role: str, broad: str, *, confidence: float = 0.82) -> EligibilityDecision:
    """Build a measured parent-eligibility result."""
    return EligibilityDecision(
        role_name=role,
        confidence=confidence,
        allowed_top_families=(broad.split("/", 1)[0], "_TO_REVIEW"),
        blocked_path_fragments=(),
        broad_folder_path=broad,
        reason="synthetic measured role",
    )


def facts(shape: str, *, confidence: float = 0.95, measured_role: str = "") -> SharedAudioFacts:
    """Build measured audio facts resembling a synthetic loop/phrase."""
    roles = {"detected_parent_role": measured_role, measured_role: 1.0} if measured_role else {}
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {"primary_shape": shape, "confidence": confidence},
            "measured_roles": roles,
        },
    )


def test_pitched_phrase_percussion_role_needs_drum_shape_before_drum_loop_rescue() -> None:
    """Sax/reed-like phrase evidence must not be rescued into Drum Loops."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            candidates=[
                candidate("Instruments/Instrument Loops/Loops", 9),
                candidate("Drums/Percussion/Bells and Metallic Percussion/Loops", 11),
                candidate("FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX", 8),
            ],
        ),
        eligibility("pitched_percussion_loop", "Drums/Drum Loops/Loops", confidence=0.78),
        facts("pitched_phrase", confidence=0.93),
    )
    assert final.final_top == "Instruments"
    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_vocal_phrase_percussion_role_needs_drum_shape_before_drum_loop_rescue() -> None:
    """Vocal phrase evidence must not be rescued into Drum Loops."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Human and Voice FX",
            candidates=[
                candidate("FX/Human and Voice FX", 8, role_strengths={"vocal_phrase": 0.85}),
                candidate("Drums/Drum Loops/Loops", 12),
                candidate("FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX", 9),
            ],
        ),
        eligibility("pitched_percussion_loop", "Drums/Drum Loops/Loops", confidence=0.78),
        facts("vocal_phrase", confidence=0.96),
    )
    assert final.final_top == "Instruments"
    assert final.folder_path == "Instruments/Voice/Phrase/One Shots"


def test_loop_like_bass_phrase_does_not_trip_short_hit_guard() -> None:
    """Rhodes/key-like loops with bass_phrase shape should stay broad Instruments."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            candidates=[
                candidate("Instruments/Instrument Loops/Loops", 9),
                candidate("Instruments/Bass/Electric Bass/One Shots", 10),
                candidate(
                    "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX", 11
                ),
            ],
        ),
        eligibility("pitched_music_loop", "Instruments/Instrument Loops/Loops", confidence=0.90),
        facts("bass_phrase", confidence=0.95, measured_role="pitched_music_loop"),
    )
    assert final.final_top == "Instruments"
    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_measured_vocal_phrase_with_voice_candidate_prefers_human_voice_over_review() -> None:
    """A vocal measured role plus a Human/Voice candidate should not become review."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            candidates=[
                candidate("Instruments/Synths/Synth Chord/One Shots", 6),
                candidate("FX/Human and Voice FX/Applause/One Shots", 10, role_strengths={"vocal_phrase": 0.8}),
            ],
        ),
        eligibility("vocal_phrase", "FX/Human and Voice FX", confidence=0.82),
        facts("vocal_phrase", confidence=0.99, measured_role="vocal_phrase"),
    )
    assert final.final_top == "Instruments"
    assert final.folder_path == "Instruments/Voice/Phrase/One Shots"


def test_measured_vocal_phrase_with_nearby_voice_candidate_prefers_human_voice_over_review() -> None:
    """Processed vocal shouts should not be rejected just because a synth candidate is closer."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            candidates=[
                candidate("Instruments/Synths/Synth Chord/One Shots", 6),
                candidate("FX/Human and Voice FX/Applause/One Shots", 10),
            ],
        ),
        eligibility("vocal_phrase", "FX/Human and Voice FX", confidence=0.82),
        facts("vocal_phrase", confidence=0.99, measured_role="vocal_phrase"),
    )
    assert final.final_top == "Instruments"
    assert final.folder_path == "Instruments/Voice/Phrase/One Shots"


def test_concrete_fx_candidates_can_override_generic_instrument_loop_rescue() -> None:
    """Siren/glitch FX candidate agreement should not be washed into Instrument Loops."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            score=999.0,
            candidates=[
                candidate("FX/Designed Noise FX/Siren/Long FX", 7),
                candidate("FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX", 9),
            ],
        ),
        eligibility("pitched_music_loop", "Instruments/Instrument Loops/Loops", confidence=0.86),
        facts("vocal_phrase", confidence=0.80, measured_role="pitched_music_loop"),
    )
    assert final.final_top == "FX"
    assert "Siren" in final.folder_path or "Glitch" in final.folder_path


def test_concrete_fx_raw_winner_survives_pitched_music_smoke_fallback() -> None:
    """A concrete FX winner should not be flattened to generic Instrument Loops."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Designed Noise FX/Siren/Long FX",
            score=3.0,
            candidates=[
                candidate("FX/Designed Noise FX/Siren/Long FX", 3),
                candidate("FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX", 9),
            ],
        ),
        eligibility("pitched_music_loop", "Instruments/Instrument Loops/Loops", confidence=0.99),
        facts("vocal_phrase", confidence=1.0, measured_role="pitched_music_loop"),
    )
    assert final.final_top == "FX"
    assert final.folder_path == "FX/Designed Noise FX/Siren/Long FX"


def test_concrete_fx_path_survives_even_when_raw_top_family_is_stale() -> None:
    """Use the raw folder path as the stable family clue when top metadata drifts."""
    inconsistent_raw = ConsensusDecision(
        final_label="FX/Designed Noise FX/Siren/Long FX",
        final_top="Instruments",
        folder_path="FX/Designed Noise FX/Siren/Long FX",
        consensus_status="synthetic_inconsistent_raw",
        reason="synthetic raw decision with stale final_top metadata",
        combined_rank_score=3.0,
        shared_candidates=[
            candidate("FX/Designed Noise FX/Siren/Long FX", 3),
            candidate("FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX", 9),
        ],
    )
    final = DecisionCoreV2().apply_eligibility(
        inconsistent_raw,
        eligibility("pitched_music_loop", "Instruments/Instrument Loops/Loops", confidence=0.99),
        facts("vocal_phrase", confidence=1.0, measured_role="pitched_music_loop"),
    )
    assert final.folder_path == "FX/Designed Noise FX/Siren/Long FX"


def test_concrete_fx_raw_winner_beats_weaker_generic_instrument_candidate() -> None:
    """Concrete FX should survive when a generic Instrument fallback is weaker."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Designed Noise FX/Siren/Long FX",
            score=3.0,
            candidates=[
                candidate("FX/Designed Noise FX/Siren/Long FX", 3),
                candidate("Instruments/Instrument Loops/Loops", 8),
            ],
        ),
        eligibility("pitched_music_loop", "Instruments/Instrument Loops/Loops", confidence=0.99),
        facts("vocal_phrase", confidence=1.0, measured_role="pitched_music_loop"),
    )
    assert final.final_top == "FX"
    assert final.folder_path == "FX/Designed Noise FX/Siren/Long FX"


def test_spoken_voice_cluster_with_music_loop_shape_rehomes_to_instrument_voice() -> None:
    """Spoken/voice brain cluster plus music-loop shape should not stay FX or review."""
    shared = [
        candidate("FX/Human and Voice FX/Spoken Voice/Long FX", 8),
        candidate("FX/Human and Voice FX/Applause/Long FX", 13),
        candidate("FX/Human and Voice FX/Crowd/Long FX", 19),
        candidate("Instruments/Guitar/Electric Guitar/One Shots", 41),
        candidate("Instruments/Synths/Synth Lead/One Shots", 44),
    ]
    measured = facts("mixed_instrument_loop", confidence=0.781075, measured_role="pitched_music_loop")
    measured.evidence["shape_vote"].update(
        {
            "f0_voiced_ratio": 0.659933,
            "pitched_event_ratio": 0.78125,
            "pitch_confidence": 0.65155,
            "percussive_event_ratio": 0.1875,
            "drumlike_frame_ratio": 0.166667,
            "tail_ratio": 0.69312,
            "onset_count": 33.0,
            "spectral_flatness_mean": 0.425387,
        }
    )
    measured.evidence["measured_roles"].update({"pitched_music_loop": 0.704719})

    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Human and Voice FX/Spoken Voice/Long FX",
            score=8.0,
            candidates=shared,
        ),
        eligibility("pitched_music_loop", "Instruments/Instrument Loops/Loops", confidence=0.82),
        measured,
    )

    assert final.final_top == "Instruments"
    assert "Voice" in final.folder_path
