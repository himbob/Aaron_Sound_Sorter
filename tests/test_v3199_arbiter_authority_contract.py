"""Authority-contract regressions for final claim arbitration.

These tests protect the architecture problem found in the 2026-05-26
uploaded examples: final invariants must not override stronger identity
evidence, and role broadening must preserve the strongest measured branch.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path


def make_claim(path: str, *, source: str = "strong_consensus", shared: list[dict] | None = None) -> ConsensusClaim:
    """Build a real candidate-like claim for arbiter unit tests."""
    return claim_from_folder_path(
        folder_path=path,
        source=source,
        reason="synthetic authority-contract test claim",
        shared=shared or [],
        raw_candidate_score=1.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner=path,
        can_override=True,
        strength=0.90,
        is_real_candidate=True,
    )


def candidate(path: str, *, score: float = 1.0, brain_rank: int = 1, physics_rank: int = 1) -> dict:
    """Build a shared candidate row."""
    return {
        "folder_path": path,
        "label": path,
        "top_family": path.split("/", 1)[0],
        "combined_rank_score": score,
        "brain_rank": brain_rank,
        "physics_rank": physics_rank,
    }


def tonal_loop_facts(*, concrete_fx_agreement: bool = False, layer: dict | None = None) -> SharedAudioFacts:
    """Build measured pitched-loop facts without using source names."""
    shape = {
        "primary_shape": "pitched_phrase",
        "confidence": 1.0,
        "pitched_event_ratio": 1.0,
        "f0_voiced_ratio": 1.0,
        "percussive_event_ratio": 0.0,
        "drumlike_frame_ratio": 0.0,
        "high_event_ratio": 0.012,
        "mid_event_ratio": 0.97,
        "low_event_ratio": 0.016,
        "spectral_flatness_mean": 0.040,
        "spectral_entropy_mean": 0.28,
        "pitch_confidence": 0.86,
    }
    evidence = {
        "shape_vote": shape,
        "measured_roles": {
            "pitched_music_loop": 0.98,
            "pitched_music_phrase": 0.90,
            "vocal_music_phrase": 0.0,
            "voiced_one_shot": 0.0,
            "drum_loop": 0.0,
        },
        "physics_layer_decision": layer or {},
    }
    if concrete_fx_agreement:
        evidence["brain_ensemble_vote_result"] = {
            "top_guesses": [
                {
                    "folder_path": "FX/Designed Noise FX/Siren/Long FX",
                    "label": "FX/Designed Noise FX/Siren/Long FX",
                    "top_family": "FX",
                    "lanes": ["full", "core_baby", "spread_baby", "outlier_baby"],
                    "confidence": 0.62,
                    "score": 0.88,
                    "rank": 1,
                }
            ]
        }
    feature_values = {
        "harmonic_energy_ratio": 0.04,
        "harmonic_to_noise_ratio": 0.04,
        "spectral_peak_stability": 0.06,
        "spectral_flatness_mean": 0.040,
        "spectral_entropy_mean": 0.28,
        "high_event_ratio": 0.012,
        "mid_event_ratio": 0.97,
        "low_event_ratio": 0.016,
        "pitched_event_ratio": 1.0,
        "f0_voiced_ratio": 1.0,
        "percussive_event_ratio": 0.0,
        "drumlike_frame_ratio": 0.0,
        "pitch_confidence": 0.86,
    }
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence=evidence,
        feature_values_by_name=feature_values,
    )


def test_concrete_fx_lane_agreement_blocks_clean_keys_loop_invariant() -> None:
    """Siren-like FX agreement must not be overwritten by a tonal keys invariant."""
    arbiter = FamilyClaimArbiter()
    shared = [
        candidate("FX/Designed Noise FX/Siren/Long FX", score=0.88, brain_rank=1, physics_rank=5),
        candidate("Instruments/Keys/Rhodes/One Shots", score=4.0, brain_rank=20, physics_rank=1),
    ]
    winning = make_claim("FX/Designed Noise FX/Siren/Long FX", shared=shared)

    repaired = arbiter._protect_clean_keys_loop_from_sax_or_fx(
        winning,
        tonal_loop_facts(concrete_fx_agreement=True),
    )

    assert repaired.folder_path == "FX/Designed Noise FX/Siren/Long FX"
    assert repaired.source == "strong_consensus"


def test_loop_broadening_preserves_strong_measured_keys_branch_over_sax_leaf() -> None:
    """A sax one-shot false positive should broaden to Keys when physics says Keys."""
    arbiter = FamilyClaimArbiter()
    layer = {
        "physics_layer_branch": "KeysPiano",
        "physics_layer_branch_confidence": 0.82,
        "instrument_branch_KeysPiano": 0.82,
        "instrument_branch_Woodwinds": 0.66,
        "instrument_branch_Brass": 0.60,
    }
    winning = make_claim("Instruments/Woodwinds/Saxophone/One Shots")

    repaired = arbiter._protect_instrument_one_shot_leaf_from_measured_loop(
        winning,
        tonal_loop_facts(layer=layer),
    )

    assert repaired.folder_path == "Instruments/Keys/Electric Piano/Loops"
    assert repaired.source == "final_measured_branch_loop_broad_bucket"


def test_loop_broadening_keeps_brass_woodwind_parent_when_measured_branch_is_not_stronger() -> None:
    """Branch preservation needs a real measured margin, not just any keys score."""
    arbiter = FamilyClaimArbiter()
    layer = {
        "physics_layer_branch": "KeysPiano",
        "physics_layer_branch_confidence": 0.72,
        "instrument_branch_KeysPiano": 0.72,
        "instrument_branch_Woodwinds": 0.68,
    }
    winning = make_claim("Instruments/Woodwinds/Saxophone/One Shots")

    repaired = arbiter._protect_instrument_one_shot_leaf_from_measured_loop(
        winning,
        tonal_loop_facts(layer=layer),
    )

    assert repaired.folder_path == "Instruments/Brass and Woodwinds/Loops"
    assert repaired.source == "parent_eligibility_broad_bucket"


def test_ambiguous_plucked_branch_does_not_steal_pitched_loop_to_guitar() -> None:
    """A crowded PluckedString branch is not enough authority to route Guitar Loops."""
    arbiter = FamilyClaimArbiter()
    layer = {
        "physics_layer_branch": "PluckedString",
        "physics_layer_branch_confidence": 0.815,
        "instrument_branch_PluckedString": 0.815,
        "instrument_branch_Synth": 0.813,
        "instrument_branch_KeysPiano": 0.730,
        "instrument_branch_Woodwinds": 0.659,
        "instrument_branch_MixedInstrument": 0.681,
    }
    winning = make_claim("Instruments/Synths/Synth Lead/One Shots")

    repaired = arbiter._protect_instrument_one_shot_leaf_from_measured_loop(
        winning,
        tonal_loop_facts(layer=layer),
    )

    assert repaired.folder_path == "Instruments/Instrument Loops/Loops"
    assert repaired.source == "final_instrument_one_shot_leaf_broad_instrument_loop_invariant"


def test_dark_low_mid_reed_branch_can_route_to_sax_loop_without_source_names() -> None:
    """A measured low-mid reed branch may choose Saxophone, not generic guitar."""
    arbiter = FamilyClaimArbiter()
    layer = {
        "physics_layer_branch": "Woodwinds",
        "physics_layer_branch_confidence": 0.97,
        "instrument_branch_Woodwinds": 0.97,
        "instrument_branch_PluckedString": 0.66,
        "instrument_branch_Synth": 0.75,
        "instrument_branch_KeysPiano": 0.72,
        "instrument_dark_low_mid_reed_loop_signal": True,
    }
    winning = make_claim("Instruments/Synths/Synth Lead/One Shots")

    repaired = arbiter._protect_instrument_one_shot_leaf_from_measured_loop(
        winning,
        tonal_loop_facts(layer=layer),
    )

    assert repaired.folder_path == "Instruments/Woodwinds/Saxophone/Loops"
    assert repaired.source == "final_measured_branch_loop_broad_bucket"


def test_authoritative_sax_subpanel_can_override_mixed_instrument_safety_bucket() -> None:
    """A broad MixedInstrument safety branch must not hide a measured sax body."""
    arbiter = FamilyClaimArbiter()
    layer = {
        "physics_layer_branch": "MixedInstrument",
        "physics_layer_branch_confidence": 0.738,
        "instrument_branch_selected": "MixedInstrument",
        "instrument_branch_selected_confidence": 0.738,
        "instrument_branch_MixedInstrument": 0.738,
        "instrument_branch_Woodwinds": 0.648,
        "instrument_Woodwinds_subpanel_selected": "Sax",
        "instrument_Woodwinds_subpanel_confidence": 0.747,
        "instrument_Woodwinds_subpanel_margin": 0.101,
        "compound_music_strength": 0.598,
        "instrument_woodwind_source_signal": False,
        "instrument_clean_tonal_reed_solo_signal": False,
        "instrument_dark_low_mid_reed_loop_signal": False,
    }
    shape = {
        "primary_shape": "pitched_phrase",
        "confidence": 0.921,
        "onset_count": 38.0,
        "pitched_event_ratio": 1.0,
        "f0_voiced_ratio": 0.927,
        "percussive_event_ratio": 0.0,
        "drumlike_frame_ratio": 0.083,
        "sustained_tonal_frame_ratio": 0.917,
        "mid_event_ratio": 0.451,
        "high_event_ratio": 0.041,
        "low_event_ratio": 0.508,
        "spectral_flatness_mean": 0.233,
        "spectral_entropy_mean": 0.369,
        "pitch_confidence": 0.971,
    }
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": shape,
            "measured_roles": {"pitched_music_loop": 0.997, "vocal_music_phrase": 0.0},
            "physics_layer_decision": layer,
        },
        feature_values_by_name={
            "pitch_confidence": 0.971,
            "f0_voiced_ratio": 0.927,
            "spectral_flatness_mean": 0.233,
            "spectral_entropy_mean": 0.369,
            "body_noise_ratio": 0.14,
            "tail_noise_ratio": 0.22,
            "presence_ratio_2000_8000hz": 0.035,
            "air_ratio_gt_8000hz": 0.0001,
        },
    )
    winning = make_claim(
        "Instruments/Instrument Loops/Loops",
        shared=[
            candidate("Instruments/Instrument Loops/Loops", score=1.0, brain_rank=20, physics_rank=1),
            candidate("Instruments/Woodwinds/Saxophone/One Shots", score=4.0, brain_rank=8, physics_rank=5),
        ],
    )

    repaired = arbiter._protect_measured_sax_loop_from_fx_or_review(winning, facts)

    assert repaired.folder_path == "Instruments/Woodwinds/Saxophone/Loops"
    assert repaired.source == "final_measured_sax_loop_invariant"


def test_plucked_subpanel_conflict_blocks_false_sax_loop_rescue() -> None:
    """Strong measured plucked-string evidence must block false Sax rescue."""
    arbiter = FamilyClaimArbiter()
    layer = {
        "physics_layer_branch": "Woodwinds",
        "physics_layer_branch_confidence": 0.95,
        "instrument_branch_selected": "Woodwinds",
        "instrument_branch_selected_confidence": 0.95,
        "instrument_branch_Woodwinds": 0.95,
        "instrument_branch_MalletBell": 0.94,
        "instrument_branch_PluckedString": 0.58,
        "instrument_Woodwinds_subpanel_selected": "AiryWoodwind",
        "instrument_Woodwinds_subpanel_confidence": 0.866,
        "instrument_Woodwinds_subpanel_margin": 0.169,
        "instrument_PluckedString_subpanel_selected": "WorldPluck",
        "instrument_PluckedString_subpanel_confidence": 0.745,
        "instrument_PluckedString_subpanel_margin": 0.084,
        "instrument_woodwind_source_signal": True,
        "instrument_clean_tonal_reed_solo_signal": False,
        "instrument_dark_low_mid_reed_loop_signal": False,
        "compound_music_strength": 0.505,
    }
    shape = {
        "primary_shape": "pitched_phrase",
        "confidence": 0.79,
        "onset_count": 11.0,
        "pitched_event_ratio": 1.0,
        "f0_voiced_ratio": 0.99,
        "percussive_event_ratio": 0.0,
        "drumlike_frame_ratio": 0.0,
        "sustained_tonal_frame_ratio": 1.0,
        "mid_event_ratio": 0.51,
        "high_event_ratio": 0.45,
        "low_event_ratio": 0.049,
        "spectral_flatness_mean": 0.046,
        "spectral_entropy_mean": 0.493,
        "pitch_confidence": 0.752,
    }
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": shape,
            "measured_roles": {"pitched_music_loop": 0.94, "vocal_music_phrase": 0.0},
            "physics_layer_decision": layer,
        },
        feature_values_by_name={
            "pitch_confidence": 0.752,
            "f0_voiced_ratio": 0.99,
            "spectral_flatness_mean": 0.046,
            "spectral_entropy_mean": 0.493,
            "body_noise_ratio": 0.291,
            "tail_noise_ratio": 0.310,
            "presence_ratio_2000_8000hz": 0.426,
            "air_ratio_gt_8000hz": 0.0,
            "harmonic_energy_ratio": 0.458,
        },
    )
    winning = make_claim(
        "Instruments/Instrument Loops/Loops",
        shared=[
            candidate("Instruments/Guitar/Electric Guitar/One Shots", score=13.0, brain_rank=3, physics_rank=10),
            candidate("Instruments/Woodwinds/Saxophone/One Shots", score=14.0, brain_rank=10, physics_rank=4),
        ],
    )

    repaired = arbiter._protect_measured_sax_loop_from_fx_or_review(winning, facts)

    assert repaired.folder_path == "Instruments/Instrument Loops/Loops"
    assert repaired.source != "final_measured_sax_loop_invariant"


def test_repeated_pitched_loop_body_broadens_false_guitar_one_shot_leaf() -> None:
    """A repeated pitched loop must not remain a Guitar/Nylon one-shot leaf."""
    arbiter = FamilyClaimArbiter()
    layer = {
        "physics_layer_branch": "MalletBell",
        "physics_layer_branch_confidence": 0.783,
        "instrument_branch_selected": "MalletBell",
        "instrument_branch_selected_confidence": 0.783,
        "instrument_branch_MalletBell": 0.783,
        "instrument_branch_Woodwinds": 0.568,
        "instrument_branch_PluckedString": 0.556,
        "instrument_branch_MixedInstrument": 0.588,
        "instrument_MalletBell_subpanel_selected": "MetallicBell",
        "instrument_MalletBell_subpanel_confidence": 0.71,
        "instrument_MalletBell_subpanel_margin": 0.04,
        "instrument_Woodwinds_subpanel_selected": "Sax",
        "instrument_Woodwinds_subpanel_confidence": 0.674,
        "instrument_Woodwinds_subpanel_margin": 0.013,
        "instrument_PluckedString_subpanel_selected": "ElectricGuitar",
        "instrument_PluckedString_subpanel_confidence": 0.680,
        "instrument_PluckedString_subpanel_margin": 0.041,
        "compound_music_strength": 0.635,
    }
    shape = {
        "primary_shape": "repeated_phrase_loop",
        "confidence": 0.784,
        "onset_count": 15.0,
        "pitched_event_ratio": 1.0,
        "f0_voiced_ratio": 0.996,
        "percussive_event_ratio": 0.0,
        "drumlike_frame_ratio": 0.0,
        "sustained_tonal_frame_ratio": 0.92,
        "mid_event_ratio": 0.48,
        "high_event_ratio": 0.342,
        "low_event_ratio": 0.18,
        "spectral_flatness_mean": 0.290,
        "spectral_entropy_mean": 0.598,
        "pitch_confidence": 0.351,
    }
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": shape,
            "measured_roles": {
                "pitched_music_loop": 0.75,
                "pitched_reed_or_instrument_loop": 0.76,
                "pitched_music_phrase": 0.72,
                "vocal_music_phrase": 0.0,
            },
            "physics_layer_decision": layer,
        },
        feature_values_by_name={
            "pitch_confidence": 0.351,
            "f0_voiced_ratio": 0.996,
            "spectral_flatness_mean": 0.290,
            "spectral_entropy_mean": 0.598,
            "harmonic_energy_ratio": 0.40,
        },
    )
    winning = make_claim("Instruments/Guitar/Nylon Guitar/One Shots")

    repaired = arbiter._protect_instrument_one_shot_leaf_from_measured_loop(winning, facts)

    assert repaired.folder_path == "Instruments/Instrument Loops/Loops"
    assert repaired.source == "final_instrument_one_shot_leaf_broad_instrument_loop_invariant"


def test_solo_phrase_guitar_body_releases_false_generic_percussion_leaf() -> None:
    """Solo pitched guitar-like phrases must not stay in Generic Percussion."""
    arbiter = FamilyClaimArbiter()
    shape = {
        "primary_shape": "solo_phrase",
        "confidence": 0.78,
        "onset_count": 14.0,
        "pitched_event_ratio": 1.0,
        "f0_voiced_ratio": 0.843,
        "percussive_event_ratio": 0.0,
        "drumlike_frame_ratio": 0.0,
        "sustained_tonal_frame_ratio": 1.0,
        "mid_event_ratio": 0.646,
        "high_event_ratio": 0.045,
        "low_event_ratio": 0.309,
        "spectral_flatness_mean": 0.218,
        "spectral_entropy_mean": 0.388,
        "pitch_confidence": 0.610,
    }
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": shape,
            "measured_roles": {
                "percussive_one_shot": 0.763,
                "voiced_one_shot": 0.431,
                "pitched_music_loop": 0.0,
                "pitched_music_phrase": 0.0,
                "drum_loop": 0.0,
            },
            "physics_layer_decision": {},
        },
        feature_values_by_name={
            "duration_sec": 7.90,
            "pitch_confidence": 0.610,
            "f0_voiced_ratio": 0.843,
            "spectral_flatness_mean": 0.218,
            "spectral_entropy_mean": 0.388,
        },
    )
    winning = make_claim(
        "Drums/Percussion/Generic Percussion/One Shots",
        shared=[
            candidate("Drums/Percussion/Generic Percussion/One Shots", score=5.0, brain_rank=1, physics_rank=4),
            candidate("Instruments/Guitar/Electric Guitar/One Shots", score=13.0, brain_rank=3, physics_rank=10),
            candidate("Instruments/Instrument Loops/Loops", score=14.0, brain_rank=12, physics_rank=3),
        ],
    )

    repaired = arbiter._review_pitched_music_hit_stolen_by_drum_leaf(winning, facts)

    assert repaired.folder_path == "Instruments/Instrument Loops/Loops"
    assert repaired.source == "final_pitched_hit_broad_instrument_release_invariant"


def test_strong_low_mid_sax_subpanel_can_beat_false_synth_loop() -> None:
    """Strong Sax subpanel plus low-mid reed body may correct a Synth loop leaf."""
    arbiter = FamilyClaimArbiter()
    layer = {
        "physics_layer_branch": "MalletBell",
        "physics_layer_branch_confidence": 0.766,
        "instrument_branch_selected": "MalletBell",
        "instrument_branch_selected_confidence": 0.766,
        "instrument_branch_MalletBell": 0.766,
        "instrument_branch_Woodwinds": 0.748,
        "instrument_branch_PluckedString": 0.444,
        "instrument_branch_Synth": 0.601,
        "instrument_branch_MixedInstrument": 0.655,
        "instrument_Woodwinds_subpanel_selected": "Sax",
        "instrument_Woodwinds_subpanel_confidence": 0.836,
        "instrument_Woodwinds_subpanel_margin": 0.051,
        "instrument_PluckedString_subpanel_selected": "WorldPluck",
        "instrument_PluckedString_subpanel_confidence": 0.475,
        "instrument_PluckedString_subpanel_margin": 0.053,
        "instrument_woodwind_source_signal": False,
        "instrument_clean_tonal_reed_solo_signal": False,
        "instrument_dark_low_mid_reed_loop_signal": False,
        "compound_music_strength": 0.456,
    }
    shape = {
        "primary_shape": "bass_phrase",
        "confidence": 0.989,
        "onset_count": 13.0,
        "pitched_event_ratio": 1.0,
        "f0_voiced_ratio": 0.795,
        "percussive_event_ratio": 0.0,
        "drumlike_frame_ratio": 0.0,
        "sustained_tonal_frame_ratio": 0.854,
        "mid_event_ratio": 0.23,
        "high_event_ratio": 0.098,
        "low_event_ratio": 0.671,
        "spectral_flatness_mean": 0.231,
        "spectral_entropy_mean": 0.279,
        "pitch_confidence": 0.955,
    }
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": shape,
            "measured_roles": {"pitched_music_loop": 0.90, "vocal_music_phrase": 0.0},
            "physics_layer_decision": layer,
        },
        feature_values_by_name={
            "pitch_confidence": 0.955,
            "f0_voiced_ratio": 0.795,
            "spectral_flatness_mean": 0.231,
            "spectral_entropy_mean": 0.279,
            "body_noise_ratio": 0.20,
            "tail_noise_ratio": 0.22,
            "presence_ratio_2000_8000hz": 0.03,
            "air_ratio_gt_8000hz": 0.0,
            "harmonic_energy_ratio": 0.38,
        },
    )
    winning = make_claim(
        "Instruments/Synths/Synth Loops",
        shared=[
            candidate("Instruments/Synths/Synth Lead/One Shots", score=8.0, brain_rank=1, physics_rank=8),
            candidate("Instruments/Woodwinds/Saxophone/One Shots", score=11.0, brain_rank=4, physics_rank=6),
        ],
    )

    repaired = arbiter._protect_measured_sax_loop_from_fx_or_review(winning, facts)

    assert repaired.folder_path == "Instruments/Woodwinds/Saxophone/Loops"
    assert repaired.source == "final_measured_sax_loop_invariant"


def test_measured_plucked_source_preserves_guitar_branch_after_broadening() -> None:
    """A real measured PluckedString loop may deepen from broad Instrument Loops."""
    arbiter = FamilyClaimArbiter()
    layer = {
        "physics_layer_branch": "PluckedString",
        "physics_layer_branch_confidence": 0.82,
        "instrument_branch_PluckedString": 0.82,
        "instrument_branch_MalletBell": 0.80,
        "instrument_branch_Woodwinds": 0.68,
        "instrument_branch_MixedInstrument": 0.74,
        "instrument_PluckedString_subpanel_selected": "NylonOrSoftPluck",
        "instrument_PluckedString_subpanel_confidence": 0.84,
        "instrument_PluckedString_subpanel_margin": 0.030,
        "instrument_plucked_string_source_signal": True,
        "instrument_reed_woodwind_source_signal": False,
        "instrument_woodwind_source_signal": False,
    }
    winning = make_claim(
        "Instruments/Instrument Loops/Loops",
        source="final_false_voice_loop_broad_instrument_invariant",
    )

    repaired = arbiter._preserve_measured_instrument_branch_after_broadening(
        winning,
        tonal_loop_facts(layer=layer),
    )

    assert repaired.folder_path == "Instruments/Guitar/Nylon Guitar/Loops"
    assert repaired.source == "final_measured_branch_loop_broad_bucket"


def test_close_internal_sax_candidate_blocks_false_guitar_depth_preserve() -> None:
    """A sax-like internal candidate plus close Woodwinds evidence should not become Guitar."""
    arbiter = FamilyClaimArbiter()
    layer = {
        "physics_layer_branch": "PluckedString",
        "physics_layer_branch_confidence": 0.82,
        "instrument_branch_PluckedString": 0.82,
        "instrument_branch_Woodwinds": 0.78,
        "instrument_branch_MalletBell": 0.79,
        "instrument_PluckedString_subpanel_selected": "AcousticGuitar",
        "instrument_PluckedString_subpanel_confidence": 0.83,
        "instrument_PluckedString_subpanel_margin": 0.025,
        "instrument_Woodwinds_subpanel_selected": "AiryWoodwind",
        "instrument_Woodwinds_subpanel_confidence": 0.81,
        "instrument_low_total": 0.08,
        "instrument_mid_ratio": 0.72,
        "instrument_plucked_string_source_signal": True,
        "instrument_reed_woodwind_source_signal": False,
        "instrument_woodwind_source_signal": False,
    }
    winning = make_claim(
        "Instruments/Instrument Loops/Loops",
        source="final_false_voice_loop_broad_instrument_invariant",
        shared=[candidate("Instruments/Woodwinds/Saxophone/One Shots", score=8.0, brain_rank=4, physics_rank=8)],
    )

    repaired = arbiter._preserve_measured_instrument_branch_after_broadening(
        winning,
        tonal_loop_facts(layer=layer),
    )

    assert repaired.folder_path == "Instruments/Instrument Loops/Loops"
    assert repaired.source == "final_false_voice_loop_broad_instrument_invariant"


def test_sax_panel_blocks_false_synth_loop_invariant_without_forcing_sax() -> None:
    """Moderately flat Sax-panel physics should block synth rescue and stay broad."""
    arbiter = FamilyClaimArbiter()
    layer = {
        "instrument_Woodwinds_subpanel_selected": "Sax",
        "instrument_Woodwinds_subpanel_confidence": 0.742,
    }
    shape = {
        "primary_shape": "bass_phrase",
        "confidence": 0.96,
        "pitched_event_ratio": 1.0,
        "f0_voiced_ratio": 0.94,
        "percussive_event_ratio": 0.0,
        "drumlike_frame_ratio": 0.0,
        "low_event_ratio": 0.61,
        "mid_event_ratio": 0.31,
        "high_event_ratio": 0.04,
        "spectral_flatness_mean": 0.29,
        "spectral_entropy_mean": 0.34,
        "pitch_confidence": 0.80,
    }
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": shape,
            "measured_roles": {"pitched_music_loop": 0.90, "drum_loop": 0.0},
            "physics_layer_decision": layer,
        },
        feature_values_by_name={
            "spectral_flatness_mean": 0.29,
            "mid_event_ratio": 0.31,
            "high_event_ratio": 0.04,
            "low_event_ratio": 0.61,
            "pitch_confidence": 0.80,
        },
    )
    winning = make_claim(
        "Instruments/Instrument Loops/Loops",
        shared=[candidate("Instruments/Synths/Synth Loops", score=4.0, brain_rank=2, physics_rank=7)],
    )

    assert not arbiter._facts_support_synth_loop(facts, winning)
