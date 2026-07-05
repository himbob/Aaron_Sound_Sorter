from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_producers.measured_transition_fx import MeasuredTransitionFxClaimProducer
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.eligibility import EligibilityDecision, infer_parent_eligibility
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path, review_claim
from aaron_sound_sorter.voters.shape_voter import shape_compatible_tops


def _shared_row(path: str, score: float = 8.0, *, brain_rank: int = 3, physics_rank: int = 3) -> dict:
    return {
        "folder_path": path,
        "label": path,
        "top_family": path.split("/", 1)[0],
        "combined_rank_score": score,
        "brain_rank": brain_rank,
        "physics_rank": physics_rank,
    }


def _claim(path: str, *, shared: list[dict] | None = None, strength: float = 0.40):
    return claim_from_folder_path(
        folder_path=path,
        source="strong_consensus",
        reason="test raw",
        shared=shared or [],
        raw_candidate_score=8.0,
        brain_rank=3,
        physics_rank=3,
        shared_winner=path,
        can_override=False,
        strength=strength,
        is_real_candidate=True,
    )


def _eligibility() -> EligibilityDecision:
    return EligibilityDecision(
        role_name="pitched_music_loop",
        confidence=0.90,
        allowed_top_families=("Instruments", "FX", "_TO_REVIEW"),
        broad_folder_path="Instruments/Instrument Loops/Loops",
        reason="synthetic eligibility",
    )


def _instrument_only_eligibility() -> EligibilityDecision:
    return EligibilityDecision(
        role_name="pitched_music_loop",
        confidence=0.90,
        allowed_top_families=("Instruments", "_TO_REVIEW"),
        broad_folder_path="Instruments/Instrument Loops/Loops",
        reason="synthetic instrument-only eligibility",
    )


def _facts(
    shape: str,
    *,
    flat: dict[str, float],
    metrics: dict[str, float] | None = None,
    confidence: float = 0.80,
) -> SharedAudioFacts:
    shape_vote = {
        "primary_shape": shape,
        "secondary_shape": "pitched_repetition_phrase",
        "confidence": confidence,
        "shape_scores": [[shape, confidence], ["pitched_repetition_phrase", 0.72], ["beat_loop", 0.42]],
        "onset_count": 12.0,
        "true_repetition_score": 0.70,
        "low_event_ratio": 0.02,
        "mid_event_ratio": 0.82,
        "high_event_ratio": 0.16,
        "pitched_event_ratio": 0.88,
        "percussive_event_ratio": 0.04,
        "drumlike_frame_ratio": 0.02,
        "sustained_tonal_frame_ratio": 0.62,
        "non_event_tonal_ratio": 0.58,
        "pitch_confidence": 0.72,
        "tail_ratio": 0.60,
        "spectral_flatness_mean": 0.24,
        "centroid_slope_norm": 0.02,
    }
    shape_vote.update(metrics or {})
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": shape_vote,
            "physics_subpanels": {"flat": flat},
            "measured_roles": {"detected_parent_role": "pitched_music_loop", "pitched_music_loop": 0.78},
        },
    )


def test_designed_tonal_shape_can_rehome_instrument_decoy_to_broad_fx() -> None:
    """Designed-FX shape may beat a weak instrument decoy when FX subpanels agree."""
    raw = _claim(
        "Instruments/Guitar/Electric Guitar/One Shots",
        shared=[
            _shared_row("FX/Hybrid Designed FX", 7.0, brain_rank=4, physics_rank=4),
            _shared_row("Instruments/Guitar/Electric Guitar/One Shots", 8.0, brain_rank=3, physics_rank=3),
        ],
    )
    facts = _facts(
        "designed_tonal_fx",
        flat={
            "fx_formant_score": 0.74,
            "fx_glitch_stutter_score": 0.62,
            "fx_motion_score": 0.22,
            "drum_hit_score": 0.16,
            "synth_tonal_source_score": 0.34,
        },
    )

    claims = MeasuredTransitionFxClaimProducer().produce(DecisionContext(raw, _eligibility(), facts))
    final = FamilyClaimArbiter().adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=facts)

    assert shape_compatible_tops("designed_tonal_fx") == ["FX"]
    assert claims[0].folder_path == "FX/Hybrid Designed FX"
    assert final.folder_path == "FX/Hybrid Designed FX"
    assert final.consensus_status == "final_measured_transition_fx_invariant"


def test_designed_fx_shape_stands_down_for_real_drum_material() -> None:
    """A designed-looking shape must not steal obvious drum material."""
    raw = _claim(
        "Drums/Drum Loops/Loops",
        shared=[
            _shared_row("Drums/Drum Loops/Loops", 5.0, brain_rank=2, physics_rank=2),
            _shared_row("FX/Hybrid Designed FX", 10.0, brain_rank=8, physics_rank=8),
        ],
    )
    facts = _facts(
        "designed_motion_fx_loop",
        flat={
            "fx_motion_score": 0.30,
            "fx_whoosh_sweep_score": 0.34,
            "fx_glitch_stutter_score": 0.52,
            "drum_snare_source_score": 0.76,
            "drum_cymbal_source_score": 0.74,
            "drum_shaker_tambourine_source_score": 0.72,
            "drum_hit_score": 0.78,
        },
        metrics={
            "percussive_event_ratio": 0.72,
            "drumlike_frame_ratio": 0.58,
            "shape_scores": [["designed_motion_fx_loop", 0.82], ["beat_loop", 0.78]],
        },
    )

    assert MeasuredTransitionFxClaimProducer().produce(DecisionContext(raw, _eligibility(), facts)) == []


def test_clean_instrument_loop_without_fx_candidate_support_stays_instrument() -> None:
    """Clean stable instrument evidence blocks designed-FX rescue when no FX candidate supports it."""
    raw = _claim(
        "Instruments/Synths/Synth Loops",
        shared=[_shared_row("Instruments/Synths/Synth Loops", 4.0, brain_rank=1, physics_rank=1)],
    )
    facts = _facts(
        "designed_tonal_fx",
        flat={
            "fx_formant_score": 0.58,
            "fx_glitch_stutter_score": 0.40,
            "fx_motion_score": 0.18,
            "fx_transition_authority_score": 0.18,
            "synth_tonal_source_score": 0.78,
        },
        metrics={
            "pitch_confidence": 0.91,
            "pitched_event_ratio": 0.92,
            "sustained_tonal_frame_ratio": 0.86,
            "spectral_flatness_mean": 0.08,
            "shape_scores": [["designed_tonal_fx", 0.80], ["pitched_repetition_phrase", 0.78]],
        },
    )

    assert MeasuredTransitionFxClaimProducer().produce(DecisionContext(raw, _eligibility(), facts)) == []


def test_high_confidence_tonal_riser_shape_beats_crash_cymbal_decoy_even_with_review() -> None:
    """A noisy tonal transition shape owns the parent when cymbal/drum evidence is only a decoy."""
    shared = [
        _shared_row("Drums/Cymbals/Crash Cymbal/One Shots", 4.0, brain_rank=3, physics_rank=1),
        _shared_row(
            "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX",
            9.0,
            brain_rank=7,
            physics_rank=9,
        ),
        _shared_row("FX/Hybrid Designed FX", 10.0, brain_rank=8, physics_rank=10),
    ]
    raw = _claim("Drums/Cymbals/Crash Cymbal/One Shots", shared=shared, strength=0.75)
    facts = _facts(
        "transition_riser",
        confidence=0.986,
        flat={
            "fx_transition_authority_score": 0.32,
            "fx_riser_build_score": 0.45,
            "fx_whoosh_sweep_score": 0.62,
            "fx_motion_score": 0.32,
            "drum_cymbal_source_score": 0.93,
            "drum_loop_source_score": 0.66,
            "drum_hit_score": 0.54,
            "onset_percussive_onset_score": 0.65,
        },
        metrics={
            "onset_count": 15.0,
            "onset_span_ratio": 0.93,
            "pulse_regularity": 0.0,
            "centroid_slope_norm": 0.213,
            "tail_ratio": 0.68,
            "high_event_ratio": 0.91,
            "mid_event_ratio": 0.08,
            "low_event_ratio": 0.01,
            "pitched_event_ratio": 0.73,
            "percussive_event_ratio": 0.20,
            "drumlike_frame_ratio": 0.02,
            "sustained_tonal_frame_ratio": 0.85,
            "non_event_tonal_ratio": 0.86,
            "spectral_flatness_mean": 0.39,
            "spectral_entropy_mean": 0.65,
            "shape_scores": [
                ["transition_riser", 0.986],
                ["designed_tonal_fx", 0.80],
                ["whoosh_sweep", 0.69],
                ["hybrid_fx_motion", 0.67],
                ["top_loop", 0.65],
                ["beat_loop", 0.54],
            ],
        },
    )

    transition_claims = MeasuredTransitionFxClaimProducer().produce(DecisionContext(raw, _eligibility(), facts))
    review = review_claim(
        label="_TO_REVIEW/Measured Role Conflict",
        reason="synthetic role conflict",
        source="role_candidate_conflict_review",
        shared=shared,
        winner=raw,
        strength=1.0,
    )
    final = FamilyClaimArbiter().adjudicate(
        raw_claim=raw,
        consensus_claims=[review],
        eligibility_claims=transition_claims,
        facts=facts,
    )

    assert transition_claims
    assert transition_claims[0].folder_path.startswith("FX/Structural and Transitional FX/Risers")
    assert final.final_top == "FX"
    assert final.folder_path.startswith("FX/Structural and Transitional FX/Risers")
    assert final.consensus_status == "final_measured_transition_fx_invariant"


def test_clean_tonal_instrument_rising_body_does_not_become_transition_fx() -> None:
    """Shape ownership must stand down for clean low-noise instrument-like tonal motion."""
    raw = _claim(
        "Instruments/Instrument Loops/Loops",
        shared=[
            _shared_row("Instruments/Instrument Loops/Loops", 4.0, brain_rank=1, physics_rank=1),
            _shared_row("FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX", 12.0),
        ],
        strength=0.86,
    )
    facts = _facts(
        "transition_riser",
        confidence=0.94,
        flat={
            "fx_transition_authority_score": 0.26,
            "fx_riser_build_score": 0.30,
            "fx_whoosh_sweep_score": 0.28,
            "fx_motion_score": 0.22,
            "synth_tonal_source_score": 0.74,
        },
        metrics={
            "centroid_slope_norm": 0.14,
            "onset_span_ratio": 0.72,
            "sustained_tonal_frame_ratio": 0.88,
            "non_event_tonal_ratio": 0.84,
            "spectral_flatness_mean": 0.08,
            "spectral_entropy_mean": 0.32,
            "percussive_event_ratio": 0.04,
            "drumlike_frame_ratio": 0.02,
            "shape_scores": [["transition_riser", 0.94], ["whoosh_sweep", 0.31], ["hybrid_fx_motion", 0.24]],
        },
    )

    assert MeasuredTransitionFxClaimProducer().produce(DecisionContext(raw, _eligibility(), facts)) == []


def _eligibility_facts_for_shape_owner(
    *,
    shape: str,
    confidence: float,
    pulse: float,
    drum_source: float,
    drum_hit: float,
    designed: float,
) -> SharedAudioFacts:
    shape_vote = {
        "primary_shape": shape,
        "confidence": confidence,
        "shape_scores": [
            [shape, confidence],
            ["designed_tonal_fx", designed],
            ["glitch_stutter", max(0.0, designed - 0.03)],
            ["beat_loop", 0.82],
        ],
        "onset_count": 18.0,
        "pulse_regularity": pulse,
        "low_event_ratio": 0.08,
        "mid_event_ratio": 0.62,
        "high_event_ratio": 0.30,
        "percussive_event_ratio": 0.20,
        "drumlike_frame_ratio": 0.18,
    }
    features = {
        "loop_percussive_event_ratio": 0.18,
        "loop_drumlike_frame_ratio": 0.18,
        "loop_pitched_event_ratio": 0.42,
        "loop_sustained_tonal_frame_ratio": 0.34,
        "loop_non_event_tonal_ratio": 0.28,
        "spectral_flatness_mean": 0.27,
        "spectral_entropy_mean": 0.62,
        "presence_ratio_2000_8000hz": 0.22,
        "air_ratio_gt_8000hz": 0.08,
        "mid_ratio_500_2000hz": 0.55,
        "event_rate_hz": 5.0,
        "log_crest": 1.22,
        "pitch_confidence": 0.22,
        "f0_voiced_ratio": 0.18,
        "duration_sec": 4.0,
    }
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        feature_values_by_name=features,
        evidence={
            "duration_sec": 4.0,
            "event_count_estimate": 18.0,
            "event_rate_hz": 5.0,
            "onset_span_ratio": 0.92,
            "shape_vote": shape_vote,
            "measured_roles": {"bright_drum_loop": 0.72, "percussive_drum_loop": 0.68},
            "fx_glitch_stutter_score": designed,
            "fx_alarm_score": max(0.0, designed - 0.12),
            "pulse_regularity": pulse,
            "drum_loop_source_score": drum_source,
            "drum_hit_score": drum_hit,
            "drum_cymbal_source_score": max(0.0, drum_hit - 0.04),
        },
    )


def test_parent_eligibility_lets_designed_fx_loop_shape_own_parent() -> None:
    """A laser/stutter top-loop decoy may be FX when pulse and drum identity are weak."""
    decision = infer_parent_eligibility(
        _eligibility_facts_for_shape_owner(
            shape="top_loop",
            confidence=0.96,
            pulse=0.0,
            drum_source=0.66,
            drum_hit=0.59,
            designed=0.74,
        )
    )

    assert decision.role_name == "designed_fx_loop_shape"
    assert decision.allowed_top_families == ("FX", "_TO_REVIEW")
    assert decision.broad_folder_path == "FX/Hybrid Designed FX"


def test_parent_eligibility_keeps_real_pulsed_drum_loop_in_drums() -> None:
    """Shape ownership stands down when real pulse/drum-source anchors are present."""
    decision = infer_parent_eligibility(
        _eligibility_facts_for_shape_owner(
            shape="top_loop",
            confidence=0.94,
            pulse=0.44,
            drum_source=0.82,
            drum_hit=0.80,
            designed=0.72,
        )
    )

    assert decision.role_name == "drum_loop"
    assert "Drums" in decision.allowed_top_families
    assert "FX" not in decision.allowed_top_families


def test_designed_low_fx_motion_claim_beats_broad_drum_parent() -> None:
    """A measured designed-FX body must not be swallowed by a broad drum-loop parent."""
    shared = [
        _shared_row("FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX", 7.0),
        _shared_row("Drums/Drum Loops/Loops", 8.0),
    ]
    raw = _claim("FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX", shared=shared)
    facts = _facts(
        "designed_low_fx",
        confidence=0.91,
        flat={
            "fx_transition_authority_score": 0.58,
            "fx_whoosh_sweep_score": 0.79,
            "fx_drop_downlifter_score": 0.67,
            "fx_reverse_score": 0.62,
            "fx_glitch_stutter_score": 0.64,
            "fx_radio_electrical_score": 0.80,
            "fx_formant_score": 0.54,
            "drum_loop_source_score": 0.66,
            "drum_hit_score": 0.42,
            "drum_cymbal_source_score": 0.84,
            "drum_shaker_tambourine_source_score": 0.72,
        },
        metrics={
            "onset_count": 35.0,
            "onset_span_ratio": 0.88,
            "true_repetition_score": 0.80,
            "pulse_regularity": 0.10,
            "pitched_event_ratio": 0.72,
            "percussive_event_ratio": 0.22,
            "drumlike_frame_ratio": 0.42,
            "sustained_tonal_frame_ratio": 0.52,
            "non_event_tonal_ratio": 0.39,
            "pitch_confidence": 0.15,
            "tail_ratio": 0.88,
            "spectral_flatness_mean": 0.25,
            "spectral_entropy_mean": 0.72,
            "centroid_slope_norm": -0.24,
            "shape_scores": [
                ["designed_low_fx", 0.91],
                ["designed_motion_fx_loop", 0.88],
                ["designed_tonal_fx", 0.86],
                ["hybrid_fx_motion", 0.86],
                ["top_loop", 0.77],
                ["beat_loop", 0.70],
            ],
        },
    )
    broad_drum_parent = claim_from_folder_path(
        folder_path="Drums/Drum Loops/Loops",
        source="parent_eligibility_broad_bucket",
        reason="synthetic broad drum parent",
        shared=shared,
        raw_candidate_score=7.0,
        brain_rank=3,
        physics_rank=3,
        shared_winner="Drums/Drum Loops/Loops",
        can_override=True,
        strength=0.90,
        is_real_candidate=False,
    )

    transition_claims = MeasuredTransitionFxClaimProducer().produce(DecisionContext(raw, _eligibility(), facts))
    final = FamilyClaimArbiter().adjudicate(
        raw_claim=raw,
        consensus_claims=[broad_drum_parent],
        eligibility_claims=transition_claims,
        facts=facts,
    )

    assert transition_claims
    assert final.final_top == "FX"
    assert final.consensus_status == "final_measured_transition_fx_invariant"


def test_real_drum_owner_still_blocks_designed_fx_motion_claim() -> None:
    """A real pulsed drum body keeps the designed-FX shape from becoming a broad FX rescue."""
    raw = _claim(
        "Drums/Drum Loops/Loops",
        shared=[
            _shared_row("Drums/Drum Loops/Loops", 5.0),
            _shared_row("FX/Hybrid Designed FX", 9.0),
        ],
    )
    facts = _facts(
        "designed_motion_fx_loop",
        confidence=0.84,
        flat={
            "fx_motion_score": 0.56,
            "fx_glitch_stutter_score": 0.61,
            "drum_loop_source_score": 0.84,
            "drum_hit_score": 0.80,
            "drum_cymbal_source_score": 0.82,
            "drum_shaker_tambourine_source_score": 0.80,
        },
        metrics={
            "onset_count": 32.0,
            "onset_span_ratio": 0.91,
            "true_repetition_score": 0.85,
            "pulse_regularity": 0.48,
            "percussive_event_ratio": 0.64,
            "drumlike_frame_ratio": 0.54,
            "shape_scores": [["designed_motion_fx_loop", 0.84], ["beat_loop", 0.82]],
        },
    )

    assert MeasuredTransitionFxClaimProducer().produce(DecisionContext(raw, _eligibility(), facts)) == []


def test_parent_instrument_eligibility_blocks_clean_designed_tonal_fx_decoy() -> None:
    """Formant-ish designed-tonal shape alone must not steal clean musical loops."""
    raw = _claim(
        "Instruments/Woodwinds/Saxophone/One Shots",
        shared=[
            _shared_row("Instruments/Woodwinds/Saxophone/One Shots", 4.0),
            _shared_row("FX/Hybrid Designed FX", 8.0),
        ],
    )
    facts = _facts(
        "designed_tonal_fx",
        confidence=0.78,
        flat={
            "fx_formant_score": 0.72,
            "fx_glitch_stutter_score": 0.40,
            "fx_motion_score": 0.14,
            "fx_transition_authority_score": 0.21,
            "voice_score": 0.72,
            "human_spoken_voice_score": 0.77,
            "woodwind_sax_score": 0.61,
            "keys_tonal_decay_score": 0.68,
        },
        metrics={
            "onset_count": 72.0,
            "onset_span_ratio": 0.99,
            "true_repetition_score": 0.82,
            "pitched_event_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "sustained_tonal_frame_ratio": 1.0,
            "non_event_tonal_ratio": 1.0,
            "pitch_confidence": 0.63,
            "f0_voiced_ratio": 0.81,
            "spectral_flatness_mean": 0.22,
            "spectral_entropy_mean": 0.56,
            "shape_scores": [
                ["designed_tonal_fx", 0.78],
                ["pitched_repetition_phrase", 0.75],
                ["repeated_phrase_loop", 0.73],
            ],
        },
    )

    claims = MeasuredTransitionFxClaimProducer().produce(DecisionContext(raw, _instrument_only_eligibility(), facts))

    assert claims == []


def test_parent_instrument_eligibility_allows_strong_noisy_designed_motion_fx() -> None:
    """Instrument parent eligibility can stand down for strong noisy designed motion."""
    raw = _claim(
        "FX/Everyday Foley/Keys Coins and Small Objects/Keys/Long FX",
        shared=[
            _shared_row("FX/Hybrid Designed FX", 8.0),
            _shared_row("Drums/Drum Loops/Loops", 9.0),
        ],
    )
    facts = _facts(
        "designed_low_fx",
        confidence=0.88,
        flat={
            "fx_transition_authority_score": 0.58,
            "fx_whoosh_sweep_score": 0.80,
            "fx_drop_downlifter_score": 0.67,
            "fx_reverse_score": 0.62,
            "fx_glitch_stutter_score": 0.64,
            "fx_radio_electrical_score": 0.80,
            "drum_loop_source_score": 0.66,
            "drum_hit_score": 0.42,
        },
        metrics={
            "onset_count": 35.0,
            "onset_span_ratio": 0.88,
            "true_repetition_score": 0.80,
            "pitched_event_ratio": 0.72,
            "percussive_event_ratio": 0.22,
            "drumlike_frame_ratio": 0.42,
            "sustained_tonal_frame_ratio": 0.52,
            "non_event_tonal_ratio": 0.39,
            "tail_ratio": 0.88,
            "spectral_flatness_mean": 0.25,
            "spectral_entropy_mean": 0.72,
            "shape_scores": [
                ["designed_low_fx", 0.88],
                ["designed_motion_fx_loop", 0.88],
                ["hybrid_fx_motion", 0.86],
                ["transition_drop", 0.73],
            ],
        },
    )

    claims = MeasuredTransitionFxClaimProducer().produce(DecisionContext(raw, _instrument_only_eligibility(), facts))

    assert claims
    assert claims[0].folder_path.startswith("FX/")


def test_high_confidence_transition_drop_beats_broad_instrument_parent() -> None:
    """A measured drop/downlifter owner may beat a broad synth/instrument-loop parent."""
    shared = [
        _shared_row("Instruments/Synths/Synth Lead/One Shots", 5.0),
        _shared_row("Instruments/Instrument Loops/Loops", 6.0),
    ]
    raw = _claim("Instruments/Synths/Synth Lead/One Shots", shared=shared)
    facts = _facts(
        "transition_drop",
        confidence=0.97,
        flat={
            "fx_drop_downlifter_score": 0.69,
            "fx_transition_authority_score": 0.44,
            "fx_motion_score": 0.42,
            "fx_whoosh_sweep_score": 0.38,
            "fx_boom_score": 0.60,
            "bass_synth_score": 0.60,
            "synth_tonal_source_score": 0.60,
            "drum_loop_source_score": 0.17,
            "drum_hit_score": 0.34,
        },
        metrics={
            "onset_count": 12.0,
            "onset_span_ratio": 0.87,
            "true_repetition_score": 0.80,
            "pulse_regularity": 0.42,
            "low_event_ratio": 0.73,
            "pitched_event_ratio": 0.92,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "sustained_tonal_frame_ratio": 0.95,
            "non_event_tonal_ratio": 1.0,
            "pitch_confidence": 0.48,
            "tail_ratio": 0.62,
            "spectral_flatness_mean": 0.15,
            "spectral_entropy_mean": 0.39,
            "centroid_slope_norm": -0.31,
            "shape_scores": [
                ["transition_drop", 0.97],
                ["hybrid_fx_motion", 0.77],
                ["designed_low_fx", 0.76],
                ["pitched_repetition_phrase", 0.73],
            ],
        },
    )
    broad_instrument_parent = claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="instrument_sibling_conflict_parent_claim",
        reason="synthetic broad instrument parent",
        shared=shared,
        raw_candidate_score=5.0,
        brain_rank=3,
        physics_rank=3,
        shared_winner="Instruments/Instrument Loops/Loops",
        can_override=True,
        strength=0.88,
        is_real_candidate=False,
    )

    transition_claims = MeasuredTransitionFxClaimProducer().produce(DecisionContext(raw, _eligibility(), facts))
    final = FamilyClaimArbiter().adjudicate(
        raw_claim=raw,
        consensus_claims=[broad_instrument_parent],
        eligibility_claims=transition_claims,
        facts=facts,
    )

    assert transition_claims
    assert transition_claims[0].folder_path.startswith("FX/Structural and Transitional FX/Drops")
    assert final.final_top == "FX"


def test_pulsed_down_sweep_shape_owns_fx_parent_over_drum_review() -> None:
    """A rhythmic down-sweep may look like a beat loop, but strong directional motion owns FX."""
    shared = [
        _shared_row("Drums/Snares/Generic Snare/One Shots", 5.0, brain_rank=2, physics_rank=2),
        _shared_row(
            "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX",
            10.0,
            brain_rank=8,
            physics_rank=8,
        ),
        _shared_row("FX/Hybrid Designed FX", 11.0, brain_rank=9, physics_rank=9),
    ]
    raw = _claim("Drums/Snares/Generic Snare/One Shots", shared=shared, strength=0.76)
    facts = _facts(
        "designed_motion_fx_loop",
        confidence=0.886,
        flat={
            "fx_drop_downlifter_score": 0.60,
            "fx_whoosh_sweep_score": 0.75,
            "fx_glitch_stutter_score": 0.80,
            "fx_radio_electrical_score": 0.82,
            "fx_motion_score": 0.43,
            "fx_transition_authority_score": 0.42,
            "drum_snare_source_score": 0.82,
            "drum_closed_hat_source_score": 0.81,
            "drum_shaker_tambourine_source_score": 0.89,
            "drum_hit_score": 0.85,
            "drum_loop_source_score": 0.81,
        },
        metrics={
            "onset_count": 60.0,
            "true_repetition_score": 0.90,
            "pulse_regularity": 0.40,
            "centroid_slope_norm": -0.63,
            "percussive_event_ratio": 0.81,
            "drumlike_frame_ratio": 0.62,
            "pitch_confidence": 0.06,
            "pitched_event_ratio": 0.0,
            "sustained_tonal_frame_ratio": 0.0,
            "spectral_flatness_mean": 0.28,
            "shape_scores": [
                ["designed_motion_fx_loop", 0.886],
                ["hybrid_fx_motion", 0.866],
                ["whoosh_sweep", 0.81],
                ["beat_loop", 0.69],
            ],
        },
    )

    transition_claims = MeasuredTransitionFxClaimProducer().produce(DecisionContext(raw, _eligibility(), facts))
    review = review_claim(
        label="_TO_REVIEW/Measured Role Conflict",
        reason="synthetic blocked role/shape review",
        source="blocked_role_shape_routing_review",
        shared=shared,
        winner=raw,
        strength=1.0,
    )
    final = FamilyClaimArbiter().adjudicate(
        raw_claim=raw,
        consensus_claims=[review],
        eligibility_claims=transition_claims,
        facts=facts,
    )

    assert transition_claims
    assert transition_claims[0].folder_path.startswith("FX/Structural and Transitional FX/Drops")
    assert final.final_top == "FX"
    assert final.folder_path.startswith("FX/Structural and Transitional FX/Drops")


def test_high_confidence_hybrid_fx_motion_releases_weak_review_to_fx() -> None:
    """Weak consensus review should not beat measured hybrid FX motion."""
    shared = [
        _shared_row("FX/Animals and Creatures/Cat/One Shots", 24.0, brain_rank=10, physics_rank=59),
        _shared_row(
            "FX/Structural and Transitional FX/Reverses and Tails/Generic Reverse/Long FX",
            12.0,
            brain_rank=8,
            physics_rank=8,
        ),
    ]
    raw = _claim("FX/Animals and Creatures/Cat/One Shots", shared=shared, strength=0.0)
    facts = _facts(
        "hybrid_fx_motion",
        confidence=0.897,
        flat={
            "fx_reverse_score": 0.82,
            "fx_motion_score": 0.54,
            "fx_transition_authority_score": 0.56,
            "fx_glitch_stutter_score": 0.66,
            "drum_hit_score": 0.40,
            "drum_loop_source_score": 0.14,
            "synth_tonal_source_score": 0.19,
        },
        metrics={
            "onset_count": 4.0,
            "onset_span_ratio": 0.44,
            "pulse_regularity": 0.82,
            "tail_ratio": 0.99,
            "temporal_centroid_ratio": 0.85,
            "centroid_slope_norm": -0.03,
            "percussive_event_ratio": 0.25,
            "drumlike_frame_ratio": 0.50,
            "spectral_flatness_mean": 0.51,
            "spectral_entropy_mean": 0.75,
            "shape_scores": [
                ["hybrid_fx_motion", 0.897],
                ["texture_bed", 0.86],
                ["reverse_swell", 0.75],
            ],
        },
    )

    transition_claims = MeasuredTransitionFxClaimProducer().produce(DecisionContext(raw, _eligibility(), facts))
    review = review_claim(
        label="_TO_REVIEW/No Strong Voter Consensus",
        reason="synthetic weak review",
        source="weak_voter_consensus",
        shared=shared,
        winner=raw,
        strength=1.0,
    )
    final = FamilyClaimArbiter().adjudicate(
        raw_claim=raw,
        consensus_claims=[review],
        eligibility_claims=transition_claims,
        facts=facts,
    )

    assert transition_claims
    assert transition_claims[0].folder_path.startswith("FX/Structural and Transitional FX/Reverses")
    assert final.final_top == "FX"
    assert final.consensus_status == "final_measured_transition_fx_invariant"


def test_single_event_noisy_texture_hit_does_not_become_fx_without_motion_owner() -> None:
    """Noisy one-hit percussion fixtures can look like texture; they need motion ownership to become FX."""
    raw = _claim(
        "Drums/Percussion/Bells and Metallic Percussion/One Shots",
        shared=[
            _shared_row("Drums/Percussion/Bells and Metallic Percussion/One Shots", 5.0),
            _shared_row("FX/Hybrid Designed FX", 9.0),
        ],
        strength=0.70,
    )
    facts = _facts(
        "texture_bed",
        confidence=0.84,
        flat={
            "fx_whoosh_sweep_score": 0.72,
            "fx_motion_score": 0.54,
            "fx_transition_authority_score": 0.47,
            "drum_hit_score": 0.73,
            "drum_cymbal_source_score": 0.69,
        },
        metrics={
            "onset_count": 1.0,
            "onset_span_ratio": 0.0,
            "tail_ratio": 0.01,
            "spectral_flatness_mean": 0.80,
            "spectral_entropy_mean": 0.61,
            "shape_scores": [
                ["texture_bed", 0.84],
                ["single_hit", 0.83],
                ["noise_texture", 0.82],
                ["hybrid_fx_motion", 0.79],
                ["designed_motion_fx_loop", 0.52],
                ["whoosh_sweep", 0.55],
            ],
        },
    )

    assert MeasuredTransitionFxClaimProducer().produce(DecisionContext(raw, _eligibility(), facts)) == []


def test_multi_event_noisy_designed_texture_can_own_fx_parent() -> None:
    """A noisy multi-event scrape/crash body with measured motion is FX, not a generic drum fallback."""
    raw = _claim(
        "Drums/Drum Loops/Loops",
        shared=[
            _shared_row("Drums/Drum Loops/Loops", 5.0),
            _shared_row("FX/Hybrid Designed FX", 10.0),
        ],
        strength=0.72,
    )
    facts = _facts(
        "noise_texture",
        confidence=0.86,
        flat={
            "fx_glitch_stutter_score": 0.81,
            "fx_whoosh_sweep_score": 0.70,
            "fx_motion_score": 0.62,
            "fx_transition_authority_score": 0.58,
            "drum_hit_score": 0.47,
            "drum_loop_source_score": 0.32,
        },
        metrics={
            "onset_count": 4.0,
            "onset_span_ratio": 0.33,
            "tail_ratio": 0.16,
            "spectral_flatness_mean": 0.72,
            "spectral_entropy_mean": 0.68,
            "percussive_event_ratio": 0.10,
            "drumlike_frame_ratio": 0.02,
            "shape_scores": [
                ["noise_texture", 0.86],
                ["designed_motion_fx_loop", 0.81],
                ["hybrid_fx_motion", 0.80],
                ["whoosh_sweep", 0.72],
                ["beat_loop", 0.40],
            ],
        },
    )

    claims = MeasuredTransitionFxClaimProducer().produce(DecisionContext(raw, _eligibility(), facts))

    assert claims
    assert claims[0].folder_path == "FX/Hybrid Designed FX"
