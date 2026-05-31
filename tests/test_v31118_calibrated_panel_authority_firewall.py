"""Calibrated voter-panel authority firewall tests."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path


def _claim(folder_path: str, *, strength: float = 0.90, source: str = "test_claim") -> ConsensusClaim:
    return claim_from_folder_path(
        folder_path=folder_path,
        source=source,
        reason="synthetic calibrated panel authority test claim",
        shared=[],
        raw_candidate_score=4.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner=folder_path,
        can_override=True,
        strength=strength,
        is_real_candidate=True,
    )


def _facts(evidence: dict[str, object]) -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=False,
        evidence=evidence,
        feature_values_by_name={},
    )


def test_synth_pad_panel_blocks_keys_decoy_competition() -> None:
    arbiter = FamilyClaimArbiter()
    raw = _claim("Instruments/Synths/Pads/Loops", source="raw_synth_pad")
    keys_decoy = _claim("Instruments/Keys/Electric Piano/Loops", source="profile_candidate_keys_claim")
    facts = _facts(
        {
            "physics_subpanels": {
                "synth_tonal_source_score": 0.90,
                "instrument_panel_Synth_synth_pad": 0.88,
                "instrument_panel_KeysPiano_electric_piano": 0.42,
            },
            "loop_sustained_tonal_frame_ratio": 0.85,
        }
    )

    assert arbiter._calibrated_panel_blocks_decoy_claim(raw, keys_decoy, facts)
    assert not arbiter._claim_can_compete(raw, keys_decoy, facts=facts)


def test_synth_pad_panel_does_not_block_when_keys_decoy_pressure_is_stronger() -> None:
    arbiter = FamilyClaimArbiter()
    raw = _claim("Instruments/Synths/Pads/Loops", source="raw_synth_pad")
    keys_candidate = _claim("Instruments/Keys/Electric Piano/Loops", source="profile_candidate_keys_claim")
    facts = _facts(
        {
            "physics_subpanels": {
                "synth_tonal_source_score": 0.52,
                "instrument_panel_Synth_synth_pad": 0.50,
                "instrument_panel_KeysPiano_electric_piano": 0.91,
                "struck_keys_score": 0.89,
            }
        }
    )

    assert not arbiter._calibrated_panel_blocks_decoy_claim(raw, keys_candidate, facts)


def test_mixed_loop_panel_blocks_specific_single_instrument_leaf() -> None:
    arbiter = FamilyClaimArbiter()
    raw = _claim("Instruments/Instrument Loops/Loops", source="mixed_instrument_loop_role_claim")
    sax_decoy = _claim("Instruments/Brass and Woodwinds/Sax/Loops", source="profile_candidate_sax_claim")
    facts = _facts(
        {
            "role_loop_score": 0.84,
            "role_phrase_score": 0.80,
            "mixed_instrument_loop_role_score": 0.82,
            "physics_subpanels": {
                "instrument_panel_Woodwinds_sax": 0.68,
                "instrument_panel_Synth_synth_pad": 0.66,
                "instrument_panel_KeysPiano_electric_piano": 0.64,
            },
        }
    )

    assert arbiter._calibrated_panel_blocks_decoy_claim(raw, sax_decoy, facts)
    assert not arbiter._claim_can_compete(raw, sax_decoy, facts=facts)


def test_mixed_loop_panel_does_not_block_broad_instrument_loop_claim() -> None:
    arbiter = FamilyClaimArbiter()
    raw = _claim("Instruments/Synths/Synth Loops", source="raw_synth_loop")
    broad = _claim("Instruments/Instrument Loops/Loops", source="mixed_instrument_loop_role_claim")
    facts = _facts(
        {
            "role_loop_score": 0.84,
            "mixed_instrument_loop_role_score": 0.82,
            "physics_subpanels": {
                "instrument_panel_Woodwinds_sax": 0.68,
                "instrument_panel_Synth_synth_pad": 0.66,
                "instrument_panel_KeysPiano_electric_piano": 0.64,
            },
        }
    )

    assert not arbiter._calibrated_panel_blocks_decoy_claim(raw, broad, facts)


def test_blip_panel_blocks_drum_decoy_only_when_blip_support_is_authoritative() -> None:
    arbiter = FamilyClaimArbiter()
    raw = _claim("FX/UI Beeps and Alerts/Blip/One Shots", source="raw_blip")
    snare_decoy = _claim("Drums/Snares/Generic Snare/One Shots", source="profile_candidate_snare_claim")
    facts = _facts(
        {
            "physics_subpanels": {
                "fx_blip_beep_score": 0.92,
                "fx_blip_score": 0.88,
                "drum_snare_source_score": 0.40,
                "drum_clap_source_score": 0.32,
            }
        }
    )

    assert arbiter._calibrated_panel_blocks_decoy_claim(raw, snare_decoy, facts)
