from __future__ import annotations

from pathlib import Path

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path
from aaron_sound_sorter.engine.voter_calibration_panels import build_voter_calibration_panels


def _claim(folder_path: str, *, strength: float, source: str) -> ConsensusClaim:
    return claim_from_folder_path(
        folder_path=folder_path,
        strength=strength,
        source=source,
        reason="test claim",
        shared=[],
        raw_candidate_score=None,
        brain_rank=None,
        physics_rank=None,
        shared_winner="",
        can_override=True,
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


def test_synth_pad_panel_can_block_keys_decoy_when_measured_support_is_strong() -> None:
    facts = _facts(
        {
            "physics_subpanels": {
                "synth_tonal_source_score": 0.88,
                "instrument_panel_Synth_synth_pad": 0.86,
                "instrument_panel_KeysPiano_electric_piano": 0.42,
            },
            "loop_sustained_tonal_frame_ratio": 0.84,
        }
    )
    claims = [_claim("Instruments/Synths/Pads/Loops", strength=0.86, source="physics_synth_pad_claim")]

    panels = {panel.target: panel for panel in build_voter_calibration_panels(claims=claims, facts=facts)}

    assert panels["synth_pad"].authority == "safe_to_block_decoys"
    assert panels["synth_pad"].confidence_tier == "strong"
    assert "instrument_panel_Synth_synth_pad" in panels["synth_pad"].evidence_keys


def test_synth_pad_panel_reports_decoy_pressure_when_keys_are_stronger() -> None:
    facts = _facts(
        {
            "physics_subpanels": {
                "synth_tonal_source_score": 0.54,
                "instrument_panel_Synth_synth_pad": 0.50,
                "instrument_panel_KeysPiano_electric_piano": 0.91,
                "struck_keys_score": 0.89,
            }
        }
    )
    claims = [_claim("Instruments/Keys/Electric Piano/Loops", strength=0.89, source="physics_keys_claim")]

    panels = {panel.target: panel for panel in build_voter_calibration_panels(claims=claims, facts=facts)}

    assert panels["synth_pad"].authority == "decoy_pressure_warning"
    assert "instrument_panel_KeysPiano_electric_piano" in panels["synth_pad"].decoy_keys


def test_mixed_instrument_loop_panel_is_broadening_only_not_leaf_authority() -> None:
    facts = _facts(
        {
            "role_loop_score": 0.82,
            "role_phrase_score": 0.76,
            "physics_subpanels": {
                "instrument_panel_Woodwinds_sax": 0.69,
                "instrument_panel_Synth_synth_pad": 0.67,
                "instrument_panel_KeysPiano_electric_piano": 0.65,
            },
        }
    )
    claims = [
        _claim("Instruments/Instrument Loops/Loops", strength=0.74, source="mixed_instrument_loop_role_claim"),
        _claim("Instruments/Brass and Woodwinds/Sax/Loops", strength=0.58, source="weak_sax_decoy_claim"),
    ]

    panels = {panel.target: panel for panel in build_voter_calibration_panels(claims=claims, facts=facts)}

    assert panels["mixed_instrument_loop"].authority == "safe_to_broaden_only"
    assert panels["mixed_instrument_loop"].claim_type == "role"


def test_blip_panel_warns_when_drum_hit_decoys_are_stronger() -> None:
    facts = _facts(
        {
            "physics_subpanels": {
                "fx_blip_beep_score": 0.45,
                "fx_blip_score": 0.42,
                "drum_snare_source_score": 0.88,
                "drum_clap_source_score": 0.81,
                "role_one_shot_score": 0.91,
            }
        }
    )
    claims = [_claim("Drums/Snares/Generic Snare/One Shots", strength=0.90, source="physics_drum_claim")]

    panels = {panel.target: panel for panel in build_voter_calibration_panels(claims=claims, facts=facts)}

    assert panels["blip"].authority == "decoy_pressure_warning"
    assert "drum_snare_source_score" in panels["blip"].decoy_keys


def test_debug_claims_file_includes_calibration_panels(tmp_path: Path, monkeypatch) -> None:
    debug_path = tmp_path / "claims.txt"
    monkeypatch.setenv("AARON_DEBUG_CLAIMS_FILE", str(debug_path))
    raw = _claim("Instruments/Synths/Pads/Loops", strength=0.86, source="raw_candidate")
    facts = _facts(
        {
            "physics_subpanels": {
                "synth_tonal_source_score": 0.88,
                "instrument_panel_Synth_synth_pad": 0.86,
            }
        }
    )

    FamilyClaimArbiter().adjudicate(
        raw_claim=raw,
        consensus_claims=[raw],
        eligibility_claims=[],
        facts=facts,
    )

    text = debug_path.read_text(encoding="utf-8")
    assert "CALIBRATION_PANEL target=synth_pad" in text
    assert "CONTRACT" in text
