from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter


def _low_kick_parent_facts(*, with_fx_impact_owner: bool) -> SharedAudioFacts:
    evidence = {
        "parent_eligibility_v2": {
            "role_name": "low_kick_like_hit",
            "allowed_top_families": ["Drums"],
            "broad_folder_path": "Drums/Kick Drums/Generic Kick/One Shots",
        },
        "measured_roles": {
            "percussive_one_shot": 0.82,
        },
        "shape_vote": {
            "primary_shape": "impact_with_tail" if with_fx_impact_owner else "single_hit",
            "confidence": 0.96,
            "onset_count": 1.0,
            "low_event_ratio": 0.74,
            "temporal_centroid_ratio": 0.18,
        },
        "duration_sec": 0.42,
    }
    if with_fx_impact_owner:
        evidence.update(
            {
                "brain_ensemble_vote_1": {
                    "folder_path": "FX/Impacts and Hits/Boom/One Shots",
                    "label": "FX/Impacts and Hits/Boom/One Shots",
                },
                "physics_vote_1": {
                    "folder_path": "FX/Impacts and Hits/Boom/One Shots",
                    "label": "FX/Impacts and Hits/Boom/One Shots",
                },
                "learned_voter_memory": {
                    "matched": True,
                    "top_family": "FX",
                    "label": "FX/Impacts and Hits/Boom/One Shots",
                    "confidence": 0.94,
                    "role": "fx_impact",
                },
                "learned_physics_memory": {
                    "matched": True,
                    "top_family": "FX",
                    "label": "FX/Impacts and Hits/Boom/One Shots",
                    "confidence": 0.96,
                    "branch": "ImpactHit",
                },
            }
        )
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence=evidence,
    )


def test_fx_impact_memory_blocks_low_kick_parent_release() -> None:
    supported = FamilyClaimArbiter()._facts_support_parent_low_kick_like_hit(
        _low_kick_parent_facts(with_fx_impact_owner=True)
    )

    assert supported is False


def test_low_kick_parent_release_still_supports_unowned_single_hit() -> None:
    supported = FamilyClaimArbiter()._facts_support_parent_low_kick_like_hit(
        _low_kick_parent_facts(with_fx_impact_owner=False)
    )

    assert supported is True
