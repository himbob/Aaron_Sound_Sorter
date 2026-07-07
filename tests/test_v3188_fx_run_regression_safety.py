"""v31.89 regressions from the post-v31.87 full FX run review.

Filenames from the real FX run are allowed as verification evidence only.  These
unit tests encode the general measured-evidence failures without using source
names in production code.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.consensus import ConsensusRunner
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2


def candidate(path: str, score: float, *, brain_rank: int = 1, physics_rank: int = 1) -> dict:
    return {
        "folder_path": path,
        "label": path,
        "top_family": path.split("/", 1)[0],
        "combined_rank_score": score,
        "brain_rank": brain_rank,
        "physics_rank": physics_rank,
        "candidate_role_signature": {},
        "brain_evidence": {"candidate_role_signature": {}},
        "physics_evidence": {"candidate_role_signature": {}},
    }


def facts(shape: str, confidence: float, roles: dict[str, float]) -> SharedAudioFacts:
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


def designed_tonal_piano_agreement_facts(*, fx_motion: float) -> SharedAudioFacts:
    """Return a tonal, loop-like Instrument case whose shape resembles designed FX."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": "designed_tonal_fx",
                "confidence": 0.88,
                "shape_scores": [
                    ["designed_tonal_fx", 0.88],
                    ["pitched_repetition_phrase", 0.84],
                    ["hybrid_fx_motion", fx_motion],
                ],
            },
            "measured_roles": {"pitched_music_loop": 0.94, "pitched_music_phrase": 0.90},
            "physics_subpanels": {
                "flat": {
                    "struck_keys_score": 0.56,
                    "struck_keys_authority_score": 0.46,
                    "keys_tonal_decay_score": 0.81,
                    "synth_tonal_source_score": 0.68,
                    "fx_motion_score": fx_motion,
                    "fx_transition_authority_score": min(0.62, fx_motion + 0.08),
                    "fx_riser_build_score": 0.20,
                    "fx_drop_downlifter_score": 0.20,
                    "fx_whoosh_sweep_score": 0.20,
                    "fx_reverse_score": 0.20,
                }
            },
        },
    )


def test_baby_reed_recall_must_not_narrow_safe_instrument_loop() -> None:
    """Generic Instrument Loops must not be turned into sax/brass by baby recall."""
    raw = candidate("Instruments/Instrument Loops/Loops", 4.0)

    assert not DecisionCoreV2()._baby_row_supports_brass_woodwind(
        "instruments/woodwinds/saxophone/one shots",
        1,
        raw=type("Raw", (), {"final_top": "Instruments", "folder_path": raw["folder_path"]})(),
        role_name="pitched_reed_or_instrument_loop",
        measured_role="pitched_reed_or_instrument_loop",
        shape_name="pitched_phrase",
        shape_confidence=1.0,
    )


def test_generic_pitched_role_prefers_broad_instrument_loop_over_sax_leaf() -> None:
    """Keys/bells/synth/strings-like loops should not become sax one-shots."""
    shared = [
        candidate("FX/Designed Noise FX/Siren/Long FX", 10.0, brain_rank=6, physics_rank=4),
        candidate("Instruments/Woodwinds/Saxophone/One Shots", 22.0, brain_rank=3, physics_rank=19),
        candidate("Instruments/Synths/Synth Lead/One Shots", 30.0, brain_rank=1, physics_rank=29),
        candidate("Instruments/Instrument Loops/Loops", 31.0, brain_rank=9, physics_rank=22),
    ]
    claim = ConsensusRunner().role_sanity_decision(
        shared=shared,
        winner=shared[0],
        facts=facts("vocal_phrase", 1.0, {"pitched_music_loop": 1.0, "pitched_music_phrase": 0.95}),
        brain_result=VoterResult(voter_name="brain", guesses=[]),
        physics_result=VoterResult(voter_name="physics", guesses=[]),
    )

    assert claim is not None
    assert claim.sub_family == "Instrument Loops"
    assert "Saxophone" not in claim.folder_path


def test_bass_phrase_shape_does_not_steal_measured_drum_loop_to_synth_bass() -> None:
    """A true drum loop with a strong low body should not become Synth Bass."""
    shared = [
        candidate("FX/Impacts and Hits/Boom/Long FX", 13.0, brain_rank=3, physics_rank=10),
        candidate("Instruments/Bass/Synth Bass/One Shots", 22.0, brain_rank=8, physics_rank=14),
        candidate("Drums/Drum Loops/Loops", 27.0, brain_rank=18, physics_rank=9),
    ]
    claim = ConsensusRunner().shape_sanity_decision(
        shared=shared,
        winner=shared[0],
        facts=facts("bass_phrase", 1.0, {"low_rhythmic_drum_loop": 0.42, "percussive_drum_loop": 0.18}),
    )

    assert claim is None


def test_low_motion_designed_tonal_shape_does_not_review_rank_one_piano_consensus() -> None:
    """Synthetic/processed tone alone should not veto concrete Piano agreement."""
    shared = [candidate("Instruments/Keys/Piano/Loops", 2.0, brain_rank=1, physics_rank=1)]

    claim = ConsensusRunner().shape_sanity_decision(
        shared=shared,
        winner=shared[0],
        facts=designed_tonal_piano_agreement_facts(fx_motion=0.23),
    )

    assert claim is None


def test_high_motion_designed_tonal_shape_can_still_review_piano_consensus() -> None:
    """Real FX motion still protects against filing a transition as Piano."""
    shared = [candidate("Instruments/Keys/Piano/Loops", 2.0, brain_rank=1, physics_rank=1)]

    claim = ConsensusRunner().shape_sanity_decision(
        shared=shared,
        winner=shared[0],
        facts=designed_tonal_piano_agreement_facts(fx_motion=0.56),
    )

    assert claim is not None
    assert claim.folder_path == "_TO_REVIEW/Shape Conflict"
    assert claim.source == "measured_shape_conflict_review"
