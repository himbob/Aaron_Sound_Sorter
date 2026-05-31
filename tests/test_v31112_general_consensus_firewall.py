"""General final-invariant consensus firewall tests."""

from __future__ import annotations

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
