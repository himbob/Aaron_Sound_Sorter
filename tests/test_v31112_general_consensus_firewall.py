"""General final-invariant consensus firewall tests."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path


def make_claim(path: str, *, source: str = "strong_consensus", score: float = 1.0) -> ConsensusClaim:
    """Build a real candidate-like claim for final firewall tests."""
    return claim_from_folder_path(
        folder_path=path,
        source=source,
        reason="synthetic final consensus firewall test claim",
        shared=[],
        raw_candidate_score=score,
        brain_rank=1,
        physics_rank=1,
        shared_winner=path,
        can_override=True,
        strength=0.94,
        is_real_candidate=True,
    )


def test_final_invariant_cannot_steal_strong_fx_consensus_to_instruments() -> None:
    """Late final invariants must not cross a strong concrete FX consensus."""
    arbiter = FamilyClaimArbiter()
    raw = make_claim("FX/Designed Noise FX/Siren/Long FX")
    final = make_claim(
        "Instruments/Keys/Electric Piano/Loops",
        source="final_clean_keys_loop_invariant",
    )

    guarded = arbiter._enforce_final_invariant_consensus_firewall(raw, final, facts=None)

    assert guarded is raw


def test_final_invariant_cannot_steal_strong_drum_consensus_to_fx() -> None:
    """The drum firewall must be generic, not limited to one old blip case."""
    arbiter = FamilyClaimArbiter()
    raw = make_claim("Drums/Snares/Generic Snare/One Shots")
    final = make_claim(
        "FX/Designed Noise FX/Blip/One Shots",
        source="final_clean_tonal_non_drum_hit_fx_invariant",
    )

    guarded = arbiter._enforce_final_invariant_consensus_firewall(raw, final, facts=None)

    assert guarded is raw


def test_hard_measured_authority_can_cross_final_family_firewall(monkeypatch) -> None:
    """A final invariant may cross only when hard measured authority says so."""
    arbiter = FamilyClaimArbiter()
    raw = make_claim("FX/Designed Noise FX/Blip/One Shots")
    final = make_claim(
        "Instruments/Instrument Loops/Loops",
        source="final_clean_pitched_instrument_loop_invariant",
    )

    monkeypatch.setattr(
        arbiter,
        "_facts_support_clean_tonal_instrument_phrase",
        lambda facts: True,
    )

    guarded = arbiter._enforce_final_invariant_consensus_firewall(raw, final, facts=None)

    assert guarded is final


def test_non_final_claims_keep_existing_arbiter_behavior() -> None:
    """The firewall protects late final invariants, not normal claim competition."""
    arbiter = FamilyClaimArbiter()
    raw = make_claim("FX/Designed Noise FX/Blip/One Shots")
    competing = make_claim(
        "Instruments/Instrument Loops/Loops",
        source="candidate_true_bucket_rescue",
    )

    guarded = arbiter._enforce_final_invariant_consensus_firewall(raw, competing, facts=None)

    assert guarded is competing


def test_measured_short_tonal_fx_raw_survives_blocked_instrument_shortcut() -> None:
    """Blocked old role shortcuts should not review a measured short FX blip."""
    arbiter = FamilyClaimArbiter()
    raw = make_claim("FX/Designed Noise FX/Beep/One Shots", score=14.0)
    blocked = make_claim(
        "Instruments/Keys/Wurlitzer/One Shots",
        source="shape_sanity_consensus",
        score=21.0,
    )
    measured = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=False,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "shape_vote": {
                "primary_shape": "solo_phrase",
                "confidence": 0.88,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
            },
            "physics_subpanels": {
                "flat": {
                    "fx_blip_beep_score": 0.86,
                    "physics_subpanel_clean_tone": 0.94,
                    "onset_pitched_onset_score": 0.93,
                    "onset_percussive_onset_score": 0.57,
                    "drum_hit_score": 0.34,
                    "drum_kick_source_score": 0.42,
                    "drum_snare_source_score": 0.38,
                    "drum_clap_source_score": 0.37,
                    "drum_tom_conga_source_score": 0.39,
                    "drum_rim_stick_source_score": 0.43,
                    "drum_cymbal_source_score": 0.26,
                    "drum_metallic_percussion_source_score": 0.31,
                }
            },
        },
        feature_values_by_name={"duration_sec": 0.48},
    )

    final = arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[blocked],
        facts=measured,
    )

    assert final.folder_path == "FX/Designed Noise FX/Beep/One Shots"
    assert final.consensus_status == "strong_consensus"
