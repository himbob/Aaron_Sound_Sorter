"""Regression tests for hierarchical abstaining sax/synth arbitration."""

from __future__ import annotations

from dataclasses import replace

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path


def guess(path: str, rank: int = 1, score: float = 1.0) -> CategoryGuess:
    return CategoryGuess(
        label=path,
        folder_path=path,
        top_family=path.split("/", 1)[0],
        score=score,
        confidence=0.90,
        rank=rank,
        reason="test guess",
    )


def shared_row(path: str, score: float, *, brain_rank: int = 2, physics_rank: int = 2) -> dict:
    return {
        "folder_path": path,
        "label": path,
        "top_family": path.split("/", 1)[0],
        "combined_rank_score": score,
        "brain_rank": brain_rank,
        "physics_rank": physics_rank,
    }


def raw_claim(path: str, *, source: str = "strong_consensus", score: float = 5.0, shared: list[dict] | None = None):
    return claim_from_folder_path(
        folder_path=path,
        source=source,
        reason="test raw",
        shared=shared or [],
        raw_candidate_score=score,
        brain_rank=1,
        physics_rank=1,
        shared_winner=path,
        can_override=False,
        strength=0.80,
        is_real_candidate=not path.startswith("_TO_REVIEW"),
    )


def eligibility(role: str, broad: str, confidence: float = 0.94) -> EligibilityDecision:
    return EligibilityDecision(
        role_name=role,
        confidence=confidence,
        allowed_top_families=("Instruments", "FX", "_TO_REVIEW"),
        broad_folder_path=broad,
        reason="synthetic eligibility",
    )


def facts(
    shape: str,
    confidence: float,
    *,
    role: str = "pitched_music_loop",
    role_strengths: dict[str, float] | None = None,
    shape_metrics: dict[str, float] | None = None,
    brain_audit: list[dict] | None = None,
    wetness: float = 0.0,
) -> SharedAudioFacts:
    metrics = {
        "primary_shape": shape,
        "confidence": confidence,
        "pitched_event_ratio": 1.0,
        "f0_voiced_ratio": 1.0,
        "percussive_event_ratio": 0.0,
        "drumlike_frame_ratio": 0.0,
        "low_event_ratio": 0.02,
        "mid_event_ratio": 0.72,
        "high_event_ratio": 0.08,
        "spectral_entropy_mean": 0.31,
        "spectral_flatness_mean": 0.08,
        "onset_count": 8.0,
        "onset_span_ratio": 0.72,
        "pulse_regularity": 0.22,
        "tail_ratio": 0.45,
    }
    metrics.update(shape_metrics or {})
    evidence = {
        "shape_vote": metrics,
        "measured_roles": {
            "detected_parent_role": role,
            "evidence": {
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.95,
                "loop_tonal_to_percussive_balance": 0.90,
                "pitched_music_loop": 1.0,
                **(role_strengths or {}),
            },
            **(role_strengths or {}),
        },
        "parent_role_audit": {"brain_top_20": brain_audit or [], "physics_top_20": []},
        "wetness_profile": {"wetness_score": wetness},
    }
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence=evidence,
    )


def test_parent_audit_sax_candidate_alone_cannot_steal_mallet_leaf() -> None:
    """Parent audit proximity alone is not enough to create a sax identity."""
    raw = raw_claim(
        "Instruments/Mallets and Bells/Vibraphone/Loops",
        shared=[shared_row("Instruments/Mallets and Bells/Vibraphone/Loops", 4.0)],
    )
    measured = facts(
        "pitched_phrase",
        1.0,
        role="pitched_reed_or_instrument_loop",
        brain_audit=[
            {"label": "Instruments/Brass/Trumpet/Loops", "rank": 1, "score": 0.45},
            {"label": "Instruments/Synths/Synth Lead/One Shots", "rank": 2, "score": 0.83},
            {"label": "Instruments/Woodwinds/Saxophone/Loops", "rank": 3, "score": 0.95},
        ],
        wetness=0.80,
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_reed_or_instrument_loop", "Instruments/Woodwinds/Saxophone/Loops"),
        measured,
        brain_result=VoterResult("brain", guesses=[]),
        physics_result=VoterResult("physics", guesses=[]),
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Mallets and Bells/Vibraphone/Loops"
    assert final.consensus_status == "strong_consensus"


def test_wet_bass_phrase_sax_candidate_can_block_synth_pluck_theft() -> None:
    """A wet tenor sax loop can measure as bass_phrase without becoming Synth Pluck."""
    raw = raw_claim(
        "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
        score=6.0,
        shared=[shared_row("Instruments/Woodwinds/Saxophone/One Shots", 34.0, brain_rank=2, physics_rank=32)],
    )
    measured = facts(
        "bass_phrase",
        0.99,
        role="pitched_music_loop",
        brain_audit=[
            {"label": "Instruments/Voice/Vocal Loops/Loops", "rank": 1, "score": 0.50},
            {"label": "Instruments/Woodwinds/Saxophone/Loops", "rank": 3, "score": 0.97},
        ],
        shape_metrics={
            "low_event_ratio": 0.652,
            "mid_event_ratio": 0.306,
            "high_event_ratio": 0.042,
            "spectral_flatness_mean": 0.290,
            "spectral_entropy_mean": 0.335,
            "f0_voiced_ratio": 0.927,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 0.917,
            "onset_count": 13.0,
        },
        wetness=0.90,
    )
    measured.evidence["brain_ensemble_vote_result"] = {
        "top_guesses": [
            {"label": "Instruments/Woodwinds/Saxophone/One Shots", "rank": 2, "score": 1.05},
            {"label": "Instruments/Synths/Synth Pluck/Loops", "rank": 3, "score": 1.40},
        ]
    }
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_music_loop", "Instruments/Instrument Loops/Loops"),
        measured,
        brain_result=VoterResult("brain", guesses=[]),
        physics_result=VoterResult("physics", guesses=[]),
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Woodwinds/Saxophone/Loops"
    assert final.consensus_status == "final_measured_sax_loop_invariant"


def test_reviewed_sax_with_rank_one_internal_sax_candidate_can_place_sax_loop() -> None:
    """Review is an abstention, not a permanent block, when a strong sax specialist exists."""
    raw = raw_claim("_TO_REVIEW/No Strong Voter Consensus", source="weak_voter_consensus", score=28.0)
    measured = facts(
        "vocal_phrase",
        1.0,
        role="pitched_reed_or_instrument_loop",
        brain_audit=[{"label": "Instruments/Woodwinds/Saxophone/One Shots", "rank": 1, "score": 1.05}],
        shape_metrics={"pitched_event_ratio": 1.0, "f0_voiced_ratio": 1.0, "high_event_ratio": 0.10},
        wetness=0.80,
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_reed_or_instrument_loop", "Instruments/Woodwinds/Saxophone/Loops"),
        measured,
        brain_result=VoterResult("brain", guesses=[]),
        physics_result=VoterResult("physics", guesses=[]),
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Woodwinds/Saxophone/Loops"
    assert final.consensus_status == "profile_candidate_sax_leaf_claim"


def test_final_sax_invariant_does_not_steal_clean_low_mid_keys_loop() -> None:
    """A nearby sax audit candidate cannot override measured clean keys physics."""
    raw = raw_claim("Instruments/Instrument Loops/Loops", score=5.0)
    measured = facts(
        "pitched_phrase",
        0.87,
        role="pitched_reed_or_instrument_loop",
        brain_audit=[{"label": "Instruments/Woodwinds/Saxophone/One Shots", "rank": 1, "score": 1.05}],
        shape_metrics={
            "pitch_confidence": 0.59,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "f0_voiced_ratio": 0.95,
            "low_event_ratio": 0.406,
            "mid_event_ratio": 0.573,
            "high_event_ratio": 0.021,
            "spectral_flatness_mean": 0.047,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
    )
    measured.feature_values_by_name.update(
        {
            "pitch_confidence": 0.59,
            "loop_pitched_event_ratio": 1.0,
            "loop_sustained_tonal_frame_ratio": 1.0,
            "presence_ratio_2000_8000hz": 0.015,
            "air_ratio_gt_8000hz": 0.002,
            "body_noise_ratio": 0.17,
            "tail_noise_ratio": 0.22,
        }
    )
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"
    assert final.consensus_status == "strong_consensus"


def test_final_sax_invariant_does_not_steal_clean_synth_pad_loop() -> None:
    """Sustained low/highless synth pads are not sax just because they are shiny."""
    raw = raw_claim("Instruments/Instrument Loops/Loops", score=5.0)
    measured = facts(
        "vocal_phrase",
        1.0,
        role="pitched_reed_or_instrument_loop",
        brain_audit=[{"label": "Instruments/Woodwinds/Saxophone/One Shots", "rank": 1, "score": 1.05}],
        shape_metrics={
            "pitch_confidence": 0.80,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "f0_voiced_ratio": 1.0,
            "low_event_ratio": 0.64,
            "mid_event_ratio": 0.36,
            "high_event_ratio": 0.0,
            "spectral_flatness_mean": 0.001,
            "spectral_entropy_mean": 0.30,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
    )
    measured.feature_values_by_name.update(
        {
            "presence_ratio_2000_8000hz": 0.0,
            "air_ratio_gt_8000hz": 0.0,
            "body_noise_ratio": 0.18,
            "tail_noise_ratio": 0.12,
        }
    )
    measured.evidence["physics_layer_decision"] = {
        "physics_layer_branch": "Synth",
        "instrument_branch_selected_confidence": 0.70,
        "instrument_branch_Synth": 0.70,
        "instrument_branch_Woodwinds": 0.50,
        "instrument_woodwind_source_signal": False,
    }
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"
    assert final.consensus_status == "strong_consensus"


def test_final_sax_invariant_does_not_steal_clean_mid_vocal_phrase_loop() -> None:
    """A clean mid-heavy vocal/synth-like phrase loop is not a sax leaf."""
    raw = raw_claim("Instruments/Instrument Loops/Loops", score=5.0)
    measured = facts(
        "vocal_phrase",
        1.0,
        role="pitched_reed_or_instrument_loop",
        brain_audit=[{"label": "Instruments/Woodwinds/Saxophone/One Shots", "rank": 1, "score": 1.05}],
        shape_metrics={
            "pitch_confidence": 0.88,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "f0_voiced_ratio": 0.94,
            "low_event_ratio": 0.34,
            "mid_event_ratio": 0.65,
            "high_event_ratio": 0.008,
            "spectral_flatness_mean": 0.087,
            "spectral_entropy_mean": 0.35,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
    )
    measured.feature_values_by_name.update(
        {
            "harmonic_energy_ratio": 0.37,
            "presence_ratio_2000_8000hz": 0.008,
            "air_ratio_gt_8000hz": 0.0,
        }
    )
    measured.evidence["physics_layer_decision"] = {
        "physics_layer_branch": "MixedInstrument",
        "instrument_branch_selected_confidence": 0.58,
        "instrument_branch_Woodwinds": 0.46,
        "instrument_woodwind_source_signal": False,
    }
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"
    assert final.consensus_status == "strong_consensus"


def test_final_sax_invariant_does_not_steal_processed_vocal_one_shot() -> None:
    """A bright vocal/formant shot with Voice branch support must not become sax."""
    raw = raw_claim("Instruments/Voice/Phrase/One Shots", score=5.0)
    measured = facts(
        "vocal_phrase",
        0.80,
        role="pitched_reed_or_instrument_phrase",
        brain_audit=[{"label": "Instruments/Woodwinds/Saxophone/One Shots", "rank": 1, "score": 1.05}],
        shape_metrics={
            "pitch_confidence": 0.86,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 0.88,
            "f0_voiced_ratio": 0.94,
            "low_event_ratio": 0.32,
            "mid_event_ratio": 0.28,
            "high_event_ratio": 0.40,
            "spectral_flatness_mean": 0.11,
            "spectral_entropy_mean": 0.61,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
    )
    measured.evidence["physics_layer_decision"] = {
        "physics_layer_branch": "MalletBell",
        "instrument_branch_selected_confidence": 0.73,
        "instrument_branch_Voice": 0.68,
        "instrument_branch_Woodwinds": 0.64,
        "instrument_woodwind_source_signal": False,
    }
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "Instruments/Voice/Phrase/One Shots"
    assert final.consensus_status == "strong_consensus"


def test_compound_music_physics_broadens_specific_sax_leaf() -> None:
    """Measured compound-loop physics can keep sax color from stealing a mixed loop."""
    raw = raw_claim("Instruments/Woodwinds/Saxophone/Loops", score=4.0)
    measured = facts(
        "pitched_phrase",
        0.96,
        role="pitched_music_loop",
        shape_metrics={
            "onset_count": 48.0,
            "mid_event_ratio": 0.48,
            "high_event_ratio": 0.18,
            "spectral_flatness_mean": 0.24,
            "spectral_entropy_mean": 0.52,
        },
    )
    measured.evidence["physics_layer_decision"] = {
        "physics_layer_top_family": "Instruments",
        "physics_layer_top_confidence": 0.90,
        "physics_layer_branch": "MixedInstrument",
        "physics_layer_branch_confidence": 0.89,
        "instrument_branch_MixedInstrument": 0.89,
        "compound_music_strength": 0.82,
        "compound_music_prefer_broad_loop": True,
    }
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"
    assert final.consensus_status == "final_compound_music_broad_instrument_loop_invariant"


def test_compound_shape_v2_broadens_specific_sax_leaf_without_identity_claim() -> None:
    """ShapeVoter V2 can prove compound structure without claiming an identity."""
    raw = raw_claim("Instruments/Woodwinds/Saxophone/Loops", score=4.0)
    measured = facts(
        "mixed_instrument_loop",
        0.82,
        role="pitched_music_loop",
        shape_metrics={
            "shape_scores": [
                ["mixed_instrument_loop", 0.82],
                ["compound_musical_loop", 0.80],
                ["layered_phrase", 0.78],
            ],
            "solo_isolation_score": 0.24,
            "layered_loop_score": 0.82,
            "instrument_plus_fx_loop_score": 0.73,
            "true_repetition_score": 0.76,
        },
    )
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"
    assert final.consensus_status == "final_compound_music_broad_instrument_loop_invariant"


def test_solo_isolation_shape_v2_does_not_broaden_sax_leaf() -> None:
    raw = raw_claim("Instruments/Woodwinds/Saxophone/Loops", score=4.0)
    measured = facts(
        "solo_phrase",
        0.88,
        role="pitched_music_loop",
        shape_metrics={
            "shape_scores": [["solo_phrase", 0.88], ["pitched_phrase", 0.86]],
            "solo_isolation_score": 0.84,
            "layered_loop_score": 0.28,
            "instrument_plus_fx_loop_score": 0.31,
            "true_repetition_score": 0.50,
        },
    )
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "Instruments/Woodwinds/Saxophone/Loops"
    assert final.consensus_status == "strong_consensus"


def test_bass_phrase_with_internal_synth_candidate_refines_to_synth_loop() -> None:
    """A low synth loop should not flatten when the ensemble has synth evidence."""
    raw = raw_claim("Instruments/Instrument Loops/Loops", score=14.0)
    measured = facts(
        "bass_phrase",
        0.94,
        role="pitched_music_loop",
        role_strengths={"pitched_music_loop": 0.92, "bass_loop": 0.59},
        shape_metrics={
            "pitch_confidence": 0.71,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "low_event_ratio": 0.84,
            "mid_event_ratio": 0.11,
            "high_event_ratio": 0.04,
            "spectral_flatness_mean": 0.30,
            "onset_count": 18.0,
        },
    )
    measured.evidence["brain_ensemble_vote_result"] = {
        "top_guesses": [
            {
                "label": "Instruments/Bass/Synth Bass/One Shots",
                "folder_path": "Instruments/Bass/Synth Bass/One Shots",
                "rank": 1,
                "score": 1.04,
            },
        ]
    }
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "Instruments/Synths/Synth Loops"
    assert final.consensus_status == "final_measured_synth_loop_invariant"


def test_measured_synth_lead_specialist_can_beat_fx_siren_false_positive() -> None:
    """A clean mid-band pitched phrase should not be reviewed as reed/FX conflict."""
    raw = raw_claim("FX/Designed Noise FX/Siren/Long FX", score=5.0)
    measured = facts(
        "pitched_phrase",
        0.996,
        role="pitched_reed_or_instrument_loop",
        shape_metrics={
            "low_event_ratio": 0.0001,
            "mid_event_ratio": 0.86,
            "high_event_ratio": 0.14,
            "spectral_entropy_mean": 0.338,
            "spectral_flatness_mean": 0.106,
        },
        wetness=0.90,
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_reed_or_instrument_loop", "Instruments/Woodwinds/Saxophone/Loops"),
        measured,
        brain_result=VoterResult("brain", guesses=[]),
        physics_result=VoterResult("physics", guesses=[]),
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Synths/Synth Lead/Loops"
    assert final.consensus_status == "final_measured_synth_loop_invariant"


def test_sax_loop_invariant_broadens_when_instrument_branch_is_not_woodwind() -> None:
    """A strong non-woodwind measured branch prevents sax leaf re-promotion."""
    raw = raw_claim(
        "Instruments/Woodwinds/Saxophone/Loops",
        score=5.0,
        shared=[
            shared_row("Instruments/Woodwinds/Saxophone/One Shots", 6.0, brain_rank=4, physics_rank=1),
            shared_row("Instruments/Instrument Loops/Loops", 8.0, brain_rank=7, physics_rank=2),
        ],
    )
    measured = facts(
        "vocal_phrase",
        0.95,
        role="pitched_music_loop",
        brain_audit=[
            {"label": "Instruments/Woodwinds/Saxophone/One Shots", "rank": 4, "score": 0.80},
        ],
        shape_metrics={
            "low_event_ratio": 0.15,
            "mid_event_ratio": 0.73,
            "high_event_ratio": 0.12,
            "spectral_flatness_mean": 0.26,
            "spectral_entropy_mean": 0.53,
            "pitch_confidence": 0.67,
            "f0_voiced_ratio": 1.0,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "onset_count": 43.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "harmonic_energy_ratio": 0.41,
            "body_noise_ratio": 0.39,
            "tail_noise_ratio": 0.35,
        },
    )
    measured.evidence["physics_layer_decision"] = {
        "physics_layer_branch": "MalletBell",
        "physics_layer_branch_confidence": 0.79,
        "instrument_branch_selected": "MalletBell",
        "instrument_branch_selected_confidence": 0.79,
        "instrument_branch_Woodwinds": 0.77,
        "instrument_woodwind_source_signal": False,
        "instrument_MalletBell_subpanel_selected": "SteelPanHandpan",
        "instrument_MalletBell_subpanel_confidence": 0.82,
        "instrument_MalletBell_subpanel_margin": 0.15,
    }

    final = DecisionCoreV2().arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[],
        facts=measured,
    )

    assert final.folder_path != "Instruments/Woodwinds/Saxophone/Loops"


def test_noisy_bassloop_subpanel_can_refine_broad_instrument_loop_to_bass() -> None:
    """BassLoop panel authority may tolerate noisy bass timbre without stealing mixed lows."""
    raw = raw_claim(
        "Instruments/Instrument Loops/Loops",
        score=5.0,
        shared=[
            shared_row("Instruments/Bass/Electric Bass/One Shots", 6.0, brain_rank=3, physics_rank=1),
            shared_row("Instruments/Instrument Loops/Loops", 7.0, brain_rank=4, physics_rank=2),
        ],
    )
    measured = facts(
        "bass_phrase",
        1.0,
        role="bass_loop",
        role_strengths={"bass_loop": 1.0, "pitched_music_loop": 1.0},
        shape_metrics={
            "low_event_ratio": 0.93,
            "mid_event_ratio": 0.06,
            "high_event_ratio": 0.01,
            "spectral_flatness_mean": 0.12,
            "pitch_confidence": 0.96,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
    )
    measured.evidence["physics_layer_decision"] = {
        "physics_layer_branch": "Bass",
        "physics_layer_branch_confidence": 1.0,
        "instrument_branch_selected": "Bass",
        "instrument_branch_selected_confidence": 1.0,
        "instrument_branch_Bass": 1.0,
        "instrument_branch_MixedInstrument": 0.45,
        "instrument_branch_Synth": 0.58,
        "instrument_branch_plausible_count": 8,
        "instrument_clean_bass_phrase": True,
        "instrument_low_pitch_bass_identity": True,
        "instrument_high_register_low_band_conflict": False,
        "instrument_Bass_subpanel_selected": "BassLoop",
        "instrument_Bass_subpanel_confidence": 0.88,
        "instrument_Bass_subpanel_margin": 0.015,
    }

    final = DecisionCoreV2().arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[],
        facts=measured,
    )

    assert final.folder_path == "Instruments/Bass/Bass Loops"


def test_measured_synth_lead_specialist_abstains_on_airy_reed_like_body() -> None:
    """The synth specialist must not become another sax/flute stealer."""
    raw = raw_claim("FX/Designed Noise FX/Siren/Long FX", score=5.0)
    measured = facts(
        "pitched_phrase",
        0.99,
        role="pitched_reed_or_instrument_loop",
        shape_metrics={
            "low_event_ratio": 0.01,
            "mid_event_ratio": 0.61,
            "high_event_ratio": 0.24,
            "spectral_entropy_mean": 0.42,
            "spectral_flatness_mean": 0.22,
        },
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_reed_or_instrument_loop", "Instruments/Woodwinds/Saxophone/Loops"),
        measured,
        brain_result=VoterResult("brain", guesses=[]),
        physics_result=VoterResult("physics", guesses=[]),
    )

    assert not any(claim.source == "profile_candidate_measured_synth_lead_claim" for claim in claims)


def test_candidate_backed_bass_phrase_sax_loop_survives_generic_loop_broadening() -> None:
    """A sax loop can measure as bass_phrase without being flattened to generic loops."""
    raw = raw_claim(
        "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
        score=10.0,
        shared=[
            shared_row(
                "Instruments/Woodwinds/Saxophone/One Shots",
                34.0,
                brain_rank=2,
                physics_rank=32,
            ),
            shared_row("Instruments/Instrument Loops/Loops", 24.0, brain_rank=21, physics_rank=3),
        ],
    )
    measured = facts(
        "bass_phrase",
        0.956,
        role="pitched_music_loop",
        shape_metrics={
            "mid_event_ratio": 0.306,
            "high_event_ratio": 0.042,
            "spectral_flatness_mean": 0.290,
            "f0_voiced_ratio": 0.927,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 0.917,
            "onset_count": 13.0,
        },
    )

    final = DecisionCoreV2().arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[],
        facts=measured,
    )

    assert final.folder_path == "Instruments/Woodwinds/Saxophone/Loops"
    assert final.consensus_status == "final_measured_sax_loop_invariant"


def test_clean_tonal_reed_body_can_route_to_sax_without_source_name() -> None:
    """Clean tonal sax bodies do not need noisy flatness to qualify as sax/reed."""
    raw = raw_claim(
        "FX/Designed Noise FX/Siren/Long FX",
        score=5.0,
        shared=[shared_row("FX/Designed Noise FX/Siren/Long FX", 5.0, brain_rank=4, physics_rank=1)],
    )
    measured = facts(
        "pitched_phrase",
        1.0,
        role="pitched_music_phrase",
        shape_metrics={
            "mid_event_ratio": 0.415,
            "high_event_ratio": 0.038,
            "spectral_flatness_mean": 0.091,
            "spectral_entropy_mean": 0.313,
            "f0_voiced_ratio": 1.0,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "onset_count": 57.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
    )

    final = DecisionCoreV2().arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[],
        facts=measured,
    )

    assert final.folder_path == "Instruments/Woodwinds/Saxophone/Loops"
    assert final.consensus_status == "final_measured_sax_loop_invariant"


def test_trumpet_or_synth_support_alone_cannot_trigger_final_sax_invariant() -> None:
    """The final sax invariant must not steal synth leads from non-sax candidates."""
    raw = raw_claim(
        "FX/Designed Noise FX/Siren/Long FX",
        score=5.0,
        shared=[
            shared_row("Instruments/Brass/Trumpet/One Shots", 31.0, brain_rank=2, physics_rank=29),
            shared_row("Instruments/Synths/Synth Lead/Loops", 7.0, brain_rank=1, physics_rank=32),
            shared_row("Instruments/Instrument Loops/Loops", 33.0, brain_rank=25, physics_rank=8),
        ],
    )
    measured = facts(
        "pitched_phrase",
        0.997,
        role="pitched_music_loop",
        shape_metrics={
            "mid_event_ratio": 0.857,
            "high_event_ratio": 0.143,
            "spectral_flatness_mean": 0.106,
            "spectral_entropy_mean": 0.339,
            "f0_voiced_ratio": 1.0,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "onset_count": 15.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
    )

    final = DecisionCoreV2().arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[],
        facts=measured,
    )

    assert "Saxophone" not in final.folder_path


def test_low_heavy_voice_like_phrase_without_sax_candidate_does_not_become_sax() -> None:
    """Low-heavy voiced loops need direct sax evidence before the sax invariant runs."""
    raw = raw_claim(
        "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
        score=6.0,
        shared=[shared_row("Instruments/Instrument Loops/Loops", 8.0, brain_rank=20, physics_rank=4)],
    )
    measured = facts(
        "pitched_phrase",
        0.912,
        role="pitched_music_loop",
        shape_metrics={
            "low_event_ratio": 0.572,
            "mid_event_ratio": 0.340,
            "high_event_ratio": 0.088,
            "spectral_flatness_mean": 0.253,
            "spectral_entropy_mean": 0.481,
            "f0_voiced_ratio": 0.998,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "onset_count": 29.0,
        },
    )

    final = DecisionCoreV2().arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[],
        facts=measured,
    )

    assert "Saxophone" not in final.folder_path


def test_repeated_phrase_loop_with_physics_voice_branch_stays_voice() -> None:
    """ShapeVoter V2 can call a rap vocal repeated_phrase_loop without flattening it."""
    raw = raw_claim(
        "Instruments/Instrument Loops/Loops",
        score=12.0,
        shared=[shared_row("Instruments/Instrument Loops/Loops", 12.0, brain_rank=12, physics_rank=4)],
    )
    measured = facts(
        "repeated_phrase_loop",
        0.81,
        role="pitched_music_loop",
        shape_metrics={
            "f0_voiced_ratio": 0.93,
            "percussive_event_ratio": 0.08,
            "drumlike_frame_ratio": 0.06,
            "spectral_flatness_mean": 0.28,
            "onset_count": 17.0,
        },
    )
    measured.evidence["physics_layer_decision"] = {
        "physics_layer_branch": "Voice",
        "physics_layer_branch_confidence": 0.59,
        "instrument_rap_voice_texture": 0.62,
    }

    final = DecisionCoreV2().arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[],
        facts=measured,
    )

    assert final.folder_path == "Instruments/Voice/Vocal Loops/Loops"
    assert final.consensus_status == "final_measured_voice_invariant"


def test_short_voice_hit_beats_tonal_stab_synth_fallback() -> None:
    """Voice branch evidence must win before a drum leaf becomes Synth Chord."""
    raw = raw_claim(
        "Drums/Cymbals/Crash Cymbal/One Shots",
        score=7.0,
        shared=[shared_row("Drums/Cymbals/Crash Cymbal/One Shots", 7.0, brain_rank=1, physics_rank=9)],
    )
    measured = facts(
        "hit_with_tail",
        0.73,
        role="voiced_one_shot",
        role_strengths={"voiced_one_shot": 0.74},
        shape_metrics={
            "duration_sec": 0.82,
            "onset_count": 1.0,
            "sustained_tonal_frame_ratio": 0.58,
            "f0_voiced_ratio": 0.83,
            "percussive_event_ratio": 0.04,
            "drumlike_frame_ratio": 0.03,
        },
    )
    measured = replace(
        measured,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
    )
    measured.evidence.update(
        {
            "duration_sec": 0.82,
            "voice_score": 0.63,
            "human_spoken_voice_score": 0.71,
            "human_breath_mouth_score": 0.86,
            "synth_tonal_source_score": 0.41,
            "synth_chord_score": 0.14,
        }
    )
    measured.evidence["physics_subpanels"] = {
        "flat": {
            "voice_score": 0.63,
            "human_spoken_voice_score": 0.71,
            "human_breath_mouth_score": 0.86,
            "synth_tonal_source_score": 0.41,
            "synth_chord_score": 0.14,
            "drum_hit_score": 0.34,
            "pitched_metal_percussion_score": 0.50,
        }
    }

    arbiter = DecisionCoreV2().arbiter
    tonal_stab_claim = arbiter._protect_measured_tonal_chord_stab(raw, measured)
    assert tonal_stab_claim.folder_path == "Instruments/Voice/Phrase/One Shots"
    assert tonal_stab_claim.source == "final_measured_voice_before_tonal_stab_invariant"


def test_physics_fx_role_layer_can_rescue_transition_from_instrument_loop() -> None:
    """Measured FX role evidence can beat a broad instrument-loop false fallback."""
    raw = raw_claim(
        "Instruments/Instrument Loops/Loops",
        score=14.0,
        shared=[
            shared_row(
                "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX",
                10.0,
                brain_rank=5,
                physics_rank=3,
            ),
            shared_row("Instruments/Instrument Loops/Loops", 14.0, brain_rank=4, physics_rank=12),
        ],
    )
    measured = facts(
        "bass_phrase",
        0.92,
        role="pitched_music_loop",
        shape_metrics={
            "drumlike_frame_ratio": 0.04,
            "pitched_event_ratio": 0.72,
            "sustained_tonal_frame_ratio": 0.50,
        },
    )
    measured.evidence["physics_layer_decision"] = {
        "physics_layer_top_family": "FX",
        "physics_layer_top_confidence": 0.62,
        "physics_layer_branch": "RiserBuild",
        "physics_layer_branch_confidence": 0.57,
        "fx_branch_selected": "RiserBuild",
        "fx_role_strength": 0.62,
        "fx_role_conflict_strength": 0.22,
        "fx_role_allows_fx": True,
    }

    final = DecisionCoreV2().arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[],
        facts=measured,
    )

    assert final.folder_path == "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX"
    assert final.consensus_status == "final_measured_transition_fx_invariant"


def test_physics_fx_role_layer_respects_instrument_conflict() -> None:
    """A pitched musical loop with high FX conflict should not be forced into FX."""
    raw = raw_claim(
        "Instruments/Instrument Loops/Loops",
        score=8.0,
        shared=[
            shared_row(
                "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX",
                12.0,
                brain_rank=6,
                physics_rank=6,
            )
        ],
    )
    measured = facts(
        "pitched_phrase",
        0.95,
        role="pitched_music_loop",
        shape_metrics={
            "pitched_event_ratio": 0.94,
            "sustained_tonal_frame_ratio": 0.88,
            "drumlike_frame_ratio": 0.02,
        },
    )
    measured.evidence["physics_layer_decision"] = {
        "physics_layer_top_family": "Instruments",
        "physics_layer_top_confidence": 0.88,
        "physics_layer_branch": "Synth",
        "fx_branch_selected": "RiserBuild",
        "fx_role_strength": 0.44,
        "fx_role_conflict_strength": 0.72,
        "fx_role_allows_fx": False,
    }

    final = DecisionCoreV2().arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[],
        facts=measured,
    )

    assert final.folder_path.startswith("Instruments/")
    assert final.consensus_status != "final_measured_transition_fx_invariant"


def test_percussive_firewall_preserves_strong_snare_leaf() -> None:
    """The FX firewall should not flatten strong snare evidence to generic percussion."""
    raw = raw_claim(
        "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/One Shots",
        source="shared_brain_physics",
        score=5.0,
        shared=[shared_row("Drums/Rims and Sticks/Rimshot/One Shots", 9.0, brain_rank=3, physics_rank=14)],
    )
    measured = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.248,
            "shape_vote": {
                "primary_shape": "single_hit",
                "confidence": 0.83,
                "percussive_event_ratio": 0.95,
                "drumlike_frame_ratio": 0.92,
                "pitched_event_ratio": 0.0,
                "f0_voiced_ratio": 0.0,
                "onset_count": 1.0,
                "temporal_centroid_ratio": 0.10,
                "spectral_entropy_mean": 0.45,
                "spectral_flatness_mean": 0.32,
                "duration_sec": 0.248,
            },
            "measured_roles": {
                "detected_parent_role": "percussive_one_shot",
                "evidence": {
                    "percussive_one_shot": 0.95,
                    "protected_percussive_one_shot": 0.95,
                },
                "percussive_one_shot": 0.95,
                "protected_percussive_one_shot": 0.95,
            },
            "brain_ensemble_vote_result": {
                "top_guesses": [
                    {"label": "Drums/Snares/Acoustic Snare/One Shots", "rank": 1, "score": 0.25, "support": 3.9},
                    {"label": "Drums/Rims and Sticks/Rimshot/One Shots", "rank": 2, "score": 0.80, "support": 1.2},
                ]
            },
        },
    )

    final = DecisionCoreV2().arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[],
        facts=measured,
    )

    assert final.folder_path == "Drums/Snares/Acoustic Snare/One Shots"
    assert final.consensus_status == "percussive_one_shot_parent_firewall"


def test_low_heavy_wet_reed_witness_does_not_overrule_stronger_synth_claim() -> None:
    """A weak sax witness stays diagnostic when the raw instrument claim is stronger."""
    raw = raw_claim(
        "Instruments/Synths/Synth Loops",
        score=7.0,
        shared=[shared_row("Instruments/Instrument Loops/Loops", 35.0, brain_rank=31, physics_rank=4)],
    )
    measured = facts(
        "bass_phrase",
        1.0,
        role="pitched_music_loop",
        brain_audit=[
            {"label": "Instruments/Guitar/Nylon Guitar/One Shots", "rank": 1, "score": 0.30},
            {"label": "Instruments/Synths/Synth Lead/One Shots", "rank": 4, "score": 1.43},
            {"label": "Instruments/Woodwinds/Saxophone/One Shots", "rank": 6, "score": 1.52},
        ],
        shape_metrics={
            "low_event_ratio": 0.889,
            "mid_event_ratio": 0.082,
            "high_event_ratio": 0.029,
            "spectral_flatness_mean": 0.203,
            "spectral_entropy_mean": 0.256,
            "pitch_confidence": 0.883,
            "f0_voiced_ratio": 0.996,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "onset_count": 14.0,
            "harmonic_energy_ratio": 0.464,
            "inharmonicity": 0.178,
            "tail_energy_ratio": 0.438,
            "body_flatness": 0.141,
            "body_entropy": 0.186,
            "f0_median_hz": 401.0,
            "f0_stability_cents": 560.0,
        },
    )

    final = DecisionCoreV2().arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[],
        facts=measured,
    )

    assert final.folder_path == "Instruments/Synths/Synth Loops"
    assert final.consensus_status == "strong_consensus"


def test_clean_synth_loop_with_incidental_sax_candidate_routes_synth_not_piano_or_sax() -> None:
    """A clean synth loop can have a near sax candidate but lacks reed physics."""
    raw = raw_claim(
        "Instruments/Instrument Loops/Loops",
        score=10.0,
        shared=[
            shared_row("Instruments/Synths/Synth Lead/One Shots", 31.0, brain_rank=1, physics_rank=30),
            shared_row("Instruments/Instrument Loops/Loops", 10.0, brain_rank=6, physics_rank=4),
        ],
    )
    measured = facts(
        "pitched_phrase",
        0.912,
        role="pitched_music_loop",
        brain_audit=[
            {"label": "Instruments/Synths/Synth Lead/One Shots", "rank": 1, "score": 0.28},
            {"label": "Instruments/Woodwinds/Saxophone/One Shots", "rank": 3, "score": 0.97},
        ],
        shape_metrics={
            "low_event_ratio": 0.238,
            "mid_event_ratio": 0.762,
            "high_event_ratio": 0.0,
            "spectral_flatness_mean": 0.002,
            "spectral_entropy_mean": 0.355,
            "pitch_confidence": 0.674,
            "f0_voiced_ratio": 1.0,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "onset_count": 92.0,
            "harmonic_energy_ratio": 0.239,
            "inharmonicity": 0.225,
            "fundamental_dominance_ratio": 0.993,
            "tail_energy_ratio": 0.730,
            "body_flatness": 0.002,
            "f0_median_hz": 513.0,
            "f0_stability_cents": 1061.0,
        },
    )

    final = DecisionCoreV2().arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[],
        facts=measured,
    )

    assert final.folder_path == "Instruments/Synths/Synth Lead/Loops"
    assert final.consensus_status == "final_measured_synth_loop_invariant"


def test_synth_panel_can_correct_weak_sax_leaf_without_filename_evidence() -> None:
    """Strong synth physics can replace a sax leaf when reed authority is weak."""
    raw = raw_claim(
        "Instruments/Woodwinds/Saxophone/Loops",
        score=8.0,
        shared=[
            shared_row("Instruments/Synths/Synth Lead/One Shots", 12.0, brain_rank=2, physics_rank=9),
            shared_row("Instruments/Woodwinds/Saxophone/Loops", 8.0, brain_rank=5, physics_rank=3),
        ],
    )
    measured = facts(
        "pitched_phrase",
        0.98,
        role="pitched_music_loop",
        shape_metrics={
            "low_event_ratio": 0.02,
            "mid_event_ratio": 0.84,
            "high_event_ratio": 0.12,
            "spectral_flatness_mean": 0.14,
            "spectral_entropy_mean": 0.46,
            "pitch_confidence": 0.86,
            "f0_voiced_ratio": 1.0,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
    )
    measured.evidence["physics_subpanels"] = {
        "flat": {
            "synth_lead_score": 0.84,
            "synth_chord_score": 0.64,
            "synth_tonal_source_score": 0.62,
            "woodwind_sax_score": 0.69,
            "reed_wind_score": 0.56,
            "reed_wind_authority_score": 0.41,
        }
    }

    final = DecisionCoreV2().arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[],
        facts=measured,
    )

    assert final.folder_path == "Instruments/Synths/Synth Lead/Loops"
    assert final.consensus_status in {
        "final_false_sax_leaf_synth_loop_invariant",
        "final_measured_synth_loop_invariant",
    }


def test_tonal_repeated_phrase_does_not_gain_drum_loop_protection() -> None:
    """Repeated pitched phrases need drum body before the drum-loop invariant wins."""
    raw = raw_claim("Instruments/Instrument Loops/Loops", score=8.0)
    measured = facts(
        "repeated_phrase_loop",
        0.86,
        role="pitched_music_loop",
        role_strengths={"pitched_music_loop": 0.88, "drum_loop": 0.24},
        shape_metrics={
            "low_event_ratio": 0.20,
            "mid_event_ratio": 0.72,
            "high_event_ratio": 0.08,
            "spectral_flatness_mean": 0.40,
            "spectral_entropy_mean": 0.46,
            "pitch_confidence": 0.80,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 0.71,
            "non_event_tonal_ratio": 0.62,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.13,
            "onset_count": 16.0,
            "true_repetition_score": 0.72,
            "pulse_regularity": 0.30,
        },
    )
    measured.evidence["physics_subpanels"] = {
        "flat": {
            "onset_pitched_onset_score": 0.57,
            "onset_percussive_onset_score": 0.39,
            "rhythmic_break_loop_score": 0.60,
        }
    }

    protected = DecisionCoreV2().arbiter._protect_measured_rhythmic_break_loop(raw, measured)

    assert protected.folder_path == "Instruments/Instrument Loops/Loops"
    assert protected.source == "strong_consensus"


def test_short_synth_hit_keeps_synth_identity_over_weak_reed_one_shot() -> None:
    """Short clean synth events should not inherit a weak sax physics top path."""
    raw = raw_claim("Instruments/Woodwinds/Saxophone/Loops", score=8.0)
    measured = facts(
        "pitched_phrase",
        0.98,
        role="pitched_music_phrase",
        shape_metrics={
            "low_event_ratio": 0.02,
            "mid_event_ratio": 0.86,
            "high_event_ratio": 0.12,
            "spectral_flatness_mean": 0.14,
            "spectral_entropy_mean": 0.45,
            "pitch_confidence": 0.86,
            "f0_voiced_ratio": 1.0,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
    )
    measured.feature_values_by_name["duration_sec"] = 0.73
    measured.evidence["physics_subpanels"] = {
        "flat": {
            "synth_lead_score": 0.84,
            "synth_chord_score": 0.63,
            "synth_tonal_source_score": 0.60,
            "woodwind_sax_score": 0.69,
            "reed_wind_authority_score": 0.40,
        }
    }
    measured.evidence["physics_vote_result"] = {
        "top_guesses": [
            {
                "folder_path": "Instruments/Woodwinds/Saxophone/One Shots",
                "label": "Instruments/Woodwinds/Saxophone/One Shots",
                "rank": 1,
                "score": 0.50,
            }
        ]
    }

    protected = DecisionCoreV2().arbiter._protect_short_instrument_one_shot_from_loop_bucket(raw, measured)

    assert protected.folder_path == "Instruments/Synths/Synth One Shots"
    assert protected.source == "final_short_synth_one_shot_invariant"


def test_dominant_wet_reed_loop_keeps_sax_depth_after_one_shot_broadening() -> None:
    """A decisive Woodwinds branch should not flatten a wet sax loop to its parent."""
    raw = raw_claim(
        "Instruments/Guitar/Nylon Guitar/One Shots",
        shared=[
            shared_row("Instruments/Woodwinds/Saxophone/One Shots", 12.0, brain_rank=5, physics_rank=1),
            shared_row("Instruments/Guitar/Nylon Guitar/One Shots", 4.0, brain_rank=1, physics_rank=98),
        ],
    )
    measured = facts(
        "bass_phrase",
        1.0,
        role="pitched_music_loop",
        shape_metrics={
            "duration_sec": 11.9,
            "onset_count": 14.0,
            "onset_span_ratio": 0.81,
            "true_repetition_score": 0.72,
            "low_event_ratio": 0.89,
            "mid_event_ratio": 0.08,
            "high_event_ratio": 0.03,
            "spectral_flatness_mean": 0.20,
            "spectral_entropy_mean": 0.26,
            "pitch_confidence": 0.88,
            "f0_voiced_ratio": 0.99,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
    )
    measured.evidence["physics_layer_decision"] = {
        "physics_layer_branch": "Woodwinds",
        "physics_layer_branch_confidence": 0.94,
        "instrument_branch_Woodwinds": 0.94,
        "instrument_branch_Synth": 0.68,
        "instrument_branch_PluckedString": 0.61,
        "instrument_branch_KeysPiano": 0.50,
        "instrument_branch_Voice": 0.40,
        "instrument_woodwind_source_signal": True,
    }
    measured.evidence["physics_subpanels"] = {
        "flat": {
            "woodwind_sax_score": 0.55,
            "reed_wind_score": 0.54,
            "reed_wind_authority_score": 0.41,
            "synth_tonal_source_score": 0.62,
            "synth_pad_score": 0.76,
            "plucked_string_score": 0.52,
            "plucked_string_authority_score": 0.50,
            "struck_keys_score": 0.45,
            "voice_score": 0.22,
            "human_spoken_voice_score": 0.47,
        }
    }
    measured.evidence["physics_vote_result"] = {
        "top_guesses": [
            {
                "folder_path": "Instruments/Woodwinds/Saxophone/One Shots",
                "label": "Instruments/Woodwinds/Saxophone/One Shots",
                "rank": 1,
                "score": 0.50,
            }
        ]
    }

    protected = DecisionCoreV2().arbiter._protect_instrument_one_shot_leaf_from_measured_loop(raw, measured)

    assert protected.folder_path == "Instruments/Woodwinds/Saxophone/Loops"
    assert protected.source == "final_measured_branch_loop_broad_bucket"
