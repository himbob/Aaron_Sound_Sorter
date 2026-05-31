from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path
from aaron_sound_sorter.engine.voter_calibration_panels import build_voter_calibration_panels


def _claim(folder_path: str, *, source: str = "test", strength: float = 0.90):
    return claim_from_folder_path(
        folder_path=folder_path,
        source=source,
        reason="source-specific authority split test",
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


def test_generic_sustained_tonal_loop_support_cannot_block_sax_identity_claim() -> None:
    raw = _claim("_TO_REVIEW/No Strong Voter Consensus", source="weak_voter_consensus")
    sax_claim = _claim(
        "Instruments/Woodwinds/Saxophone/Loops",
        source="profile_candidate_sax_leaf_claim",
        strength=0.99,
    )
    facts = _facts(
        {
            "loop_sustained_tonal_frame_ratio": 0.95,
            "sustained_tonal_frame_ratio": 0.94,
            "pitched_event_ratio": 1.0,
            "f0_voiced_ratio": 1.0,
        }
    )

    panels = {panel.target: panel for panel in build_voter_calibration_panels(claims=[raw, sax_claim], facts=facts)}

    assert panels["synth_pad"].raw_support >= 0.90
    assert panels["synth_pad"].source_specific_support < 0.82
    assert panels["synth_pad"].authority != "safe_to_block_decoys"
    assert not FamilyClaimArbiter()._calibrated_panel_blocks_decoy_claim(raw, sax_claim, facts)


def test_synth_pad_source_specific_support_can_still_block_keys_decoy() -> None:
    raw = _claim("Instruments/Synths/Pads/Loops", source="raw_synth_pad")
    keys_decoy = _claim("Instruments/Keys/Electric Piano/Loops", source="profile_candidate_keys_claim")
    facts = _facts(
        {
            "physics_subpanels": {
                "synth_tonal_source_score": 0.91,
                "instrument_panel_Synth_synth_pad": 0.88,
                "instrument_panel_KeysPiano_electric_piano": 0.42,
            },
            "loop_sustained_tonal_frame_ratio": 0.95,
        }
    )

    panels = {panel.target: panel for panel in build_voter_calibration_panels(claims=[raw, keys_decoy], facts=facts)}

    assert panels["synth_pad"].source_specific_support >= 0.82
    assert panels["synth_pad"].authority == "safe_to_block_decoys"
    assert FamilyClaimArbiter()._calibrated_panel_blocks_decoy_claim(raw, keys_decoy, facts)
