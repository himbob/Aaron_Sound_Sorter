"""Synthetic locks for real FX_Aaron2 one-file panel failures.

The test names describe the verification case, but the assertions use only
internal voter categories and measured audio facts.  These are regression locks
for the Claim-Arbiter design, not filename evidence.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path


def guess(path: str, rank: int) -> CategoryGuess:
    """Build a ranked candidate guess."""
    return CategoryGuess(
        label=path,
        folder_path=path,
        top_family=path.split("/", 1)[0],
        score=float(rank),
        confidence=max(0.0, 1.0 - rank / 30.0),
        rank=rank,
        reason="synthetic candidate",
        evidence={},
    )


def facts(shape: str, confidence: float, roles: dict[str, float]) -> SharedAudioFacts:
    """Build measured facts without source names."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {"primary_shape": shape, "confidence": confidence},
            "measured_roles": roles,
            "direct_body_view": {"available": True, "measured_roles": roles},
        },
    )


def raw_claim(path: str, shared: list[dict], score: float = 3.0):
    """Build the raw claim produced before eligibility claims."""
    return claim_from_folder_path(
        folder_path=path,
        source="strong_consensus",
        reason="synthetic raw shared winner",
        shared=shared,
        raw_candidate_score=score,
        brain_rank=1,
        physics_rank=2,
        shared_winner=path,
        can_override=False,
        strength=0.80,
        is_real_candidate=True,
    )


def shared_row(path: str, score: float, roles: dict[str, float] | None = None) -> dict:
    """Build one shared candidate row."""
    role_signature = roles or {}
    return {
        "label": path,
        "folder_path": path,
        "top_family": path.split("/", 1)[0],
        "brain_rank": int(score),
        "physics_rank": int(score),
        "combined_rank_score": float(score),
        "candidate_role_signature": role_signature,
        "brain_evidence": {"candidate_role_signature": role_signature},
        "physics_evidence": {"candidate_role_signature": role_signature},
    }


def eligibility(role: str, broad: str, confidence: float = 0.90) -> EligibilityDecision:
    """Build an explicit parent-eligibility decision."""
    return EligibilityDecision(
        role_name=role,
        confidence=confidence,
        allowed_top_families=("Instruments", "_TO_REVIEW"),
        blocked_path_fragments=(),
        broad_folder_path=broad,
        reason="synthetic eligibility",
    )


def test_nearby_woodwind_profile_claim_does_not_overrule_concrete_fx_without_specific_support() -> None:
    """A weak nearby woodwind row must not become another overcorrection vacuum."""
    shared = [
        shared_row("FX/Designed Noise FX/Alarm/Long FX", 3.0),
        shared_row("Instruments/Woodwinds/Flute/One Shots", 6.0),
        shared_row("Instruments/Instrument Loops/Loops", 11.0),
    ]
    brain_result = VoterResult(
        voter_name="brain_full",
        guesses=[
            guess("FX/Designed Noise FX/Alarm/Long FX", 1),
            guess("Instruments/Instrument Loops/Loops", 3),
            guess("Instruments/Woodwinds/Flute/One Shots", 6),
        ],
    )
    raw = raw_claim("FX/Designed Noise FX/Alarm/Long FX", shared)
    measured = facts("pitched_phrase", 0.91, {"pitched_music_loop": 0.94, "vocal_music_phrase": 0.20})
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_reed_or_instrument_loop", "Instruments/Woodwinds/Saxophone/Loops", 0.94),
        measured,
        brain_result=brain_result,
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "FX/Designed Noise FX/Alarm/Long FX"
    assert final.consensus_status == "strong_consensus"


def test_stronger_reed_candidate_stays_broad_when_identity_is_not_physics_supported() -> None:
    """A reed-like loop may stay broad instead of claiming a specific sibling branch."""
    shared = [
        shared_row("Instruments/Instrument Loops/Loops", 16.0),
        shared_row("Instruments/Woodwinds/Saxophone/One Shots", 6.0),
    ]
    brain_result = VoterResult(
        voter_name="brain_full",
        guesses=[
            guess("Instruments/Woodwinds/Saxophone/One Shots", 1),
            guess("Instruments/Instrument Loops/Loops", 3),
        ],
    )
    raw = raw_claim("Instruments/Instrument Loops/Loops", shared, score=16.0)
    measured = facts(
        "pitched_phrase",
        0.92,
        {"pitched_reed_or_instrument_loop": 0.92, "pitched_music_loop": 0.88},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_reed_or_instrument_loop", "Instruments/Woodwinds/Saxophone/Loops", 0.92),
        measured,
        brain_result=brain_result,
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"
    assert final.consensus_status in {"strong_consensus", "profile_candidate_ambiguous_reed_like_parent_claim"}


def test_reed_one_shot_leaf_broadens_to_brass_woodwind_loop_bucket() -> None:
    """A sax/woodwind one-shot leaf with loop structure should become broad reed loops."""
    shared = [
        shared_row("Instruments/Woodwinds/Saxophone/One Shots", 4.0),
        shared_row("Instruments/Instrument Loops/Loops", 12.0),
    ]
    raw = raw_claim("Instruments/Woodwinds/Saxophone/One Shots", shared, score=4.0)
    measured = facts(
        "pitched_phrase",
        0.94,
        {"pitched_music_loop": 0.94, "pitched_music_phrase": 0.90},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_music_loop", "Instruments/Instrument Loops/Loops", 0.94),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Brass and Woodwinds/Loops"
    assert final.consensus_status in {"candidate_true_bucket_rescue", "parent_eligibility_broad_bucket"}


def test_non_reed_one_shot_leaf_still_broadens_to_generic_instrument_loops() -> None:
    """Generic one-shot instrument leaves should stay broad unless a safe subfamily exists."""
    shared = [
        shared_row("Instruments/Guitar/Nylon Guitar/One Shots", 4.0),
        shared_row("Instruments/Instrument Loops/Loops", 12.0),
    ]
    raw = raw_claim("Instruments/Guitar/Nylon Guitar/One Shots", shared, score=4.0)
    measured = facts(
        "pitched_phrase",
        0.94,
        {"pitched_music_loop": 0.94, "pitched_music_phrase": 0.90},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_music_loop", "Instruments/Instrument Loops/Loops", 0.94),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_weaker_reed_candidate_does_not_deepen_generic_instrument_loop() -> None:
    """A reed-like role alone must not over-narrow broad Instrument Loops."""
    shared = [
        shared_row("Instruments/Instrument Loops/Loops", 8.0),
        shared_row("Instruments/Woodwinds/Saxophone/One Shots", 15.0),
    ]
    brain_result = VoterResult(
        voter_name="brain_full",
        guesses=[
            guess("Instruments/Woodwinds/Saxophone/One Shots", 1),
            guess("Instruments/Instrument Loops/Loops", 3),
        ],
    )
    raw = raw_claim("Instruments/Instrument Loops/Loops", shared, score=8.0)
    measured = facts(
        "pitched_phrase",
        0.92,
        {"pitched_reed_or_instrument_loop": 0.92, "pitched_music_loop": 0.88},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_reed_or_instrument_loop", "Instruments/Woodwinds/Saxophone/Loops", 0.92),
        measured,
        brain_result=brain_result,
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_bass_phrase_with_drum_loop_role_does_not_steal_drum_loop_to_bass() -> None:
    """A bass-shaped drum loop must stay in Drum Loops when the measured role is drum_loop."""
    shared = [
        shared_row("FX/Impacts and Hits/Generic Impact/Long FX", 3.0),
        shared_row("Drums/Drum Loops/Loops", 6.0),
    ]
    brain_result = VoterResult(
        voter_name="brain_full",
        guesses=[
            guess("FX/Impacts and Hits/Generic Impact/Long FX", 1),
            guess("Instruments/Bass/Synth Bass/One Shots", 2),
        ],
    )
    raw = raw_claim("FX/Impacts and Hits/Generic Impact/Long FX", shared)
    measured = facts("bass_phrase", 0.96, {"drum_loop": 0.78, "low_rhythmic_drum_loop": 0.74})
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("drum_loop", "Drums/Drum Loops/Loops", 0.78),
        measured,
        brain_result=brain_result,
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Drums/Drum Loops/Loops"
    assert "Bass" not in final.folder_path


def test_stable_pitched_phrase_blocks_vocal_shape_without_voice_candidate() -> None:
    """A vocal-shaped pitched phrase with no real voice candidate should not become Human/Voice."""
    shared = [
        shared_row("FX/Designed Noise FX/Alarm/Long FX", 3.0),
        shared_row("Instruments/Instrument Loops/Loops", 18.0),
    ]
    raw = raw_claim("FX/Designed Noise FX/Alarm/Long FX", shared)
    measured = facts(
        "vocal_phrase",
        1.0,
        {"pitched_music_loop": 0.98, "pitched_music_phrase": 0.85, "vocal_music_phrase": 1.0},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("vocal_music_phrase", "FX/Human and Voice FX", 1.0),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert "Human and Voice" not in final.folder_path


def test_stronger_voice_candidate_can_deepen_generic_instrument_loop() -> None:
    """A real stronger voice candidate may beat generic Instrument Loops."""
    shared = [
        shared_row("Instruments/Instrument Loops/Loops", 25.0),
        shared_row("FX/Human and Voice FX/Spoken Voice/Long FX", 12.0, {"vocal_music_phrase": 0.88}),
    ]
    raw = raw_claim("Instruments/Instrument Loops/Loops", shared, score=25.0)
    measured = facts("vocal_phrase", 0.94, {"vocal_music_phrase": 0.91, "pitched_music_loop": 0.70})
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        EligibilityDecision(
            role_name="vocal_music_phrase",
            confidence=0.91,
            allowed_top_families=("FX", "Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(),
            broad_folder_path="FX/Human and Voice FX",
            reason="synthetic vocal phrase eligibility",
        ),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Voice/Phrase/One Shots"
    assert final.consensus_status == "human_voice_true_bucket_rescue"


def test_voice_candidate_does_not_steal_generic_pitched_instrument_loop() -> None:
    """Voice depth needs measured vocal eligibility, not just a vocal-looking row."""
    shared = [
        shared_row("Instruments/Instrument Loops/Loops", 8.0),
        shared_row("FX/Human and Voice FX/Spoken Voice/Long FX", 12.0, {"vocal_music_phrase": 0.88}),
    ]
    raw = raw_claim("Instruments/Instrument Loops/Loops", shared, score=8.0)
    measured = facts("pitched_phrase", 0.94, {"vocal_music_phrase": 0.20, "pitched_music_loop": 0.91})
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        EligibilityDecision(
            role_name="pitched_music_loop",
            confidence=0.91,
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(),
            broad_folder_path="Instruments/Instrument Loops/Loops",
            reason="synthetic pitched phrase eligibility",
        ),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_measured_percussive_hit_can_escape_fx_glitch_leaf() -> None:
    """A measured percussion one-shot may broaden an FX glitch false positive."""
    shared = [
        shared_row("FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/One Shots", 4.0),
        shared_row("Drums/Rims and Sticks/Rimshot/One Shots", 8.0, {"percussive_one_shot": 0.98}),
    ]
    raw = raw_claim(
        "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/One Shots",
        shared,
        score=4.0,
    )
    measured = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.25,
            "event_count_estimate": 1.0,
            "shape_vote": {"primary_shape": "single_hit", "confidence": 0.83, "onset_count": 1.0},
            "measured_roles": {"percussive_one_shot": 0.97, "primary_roles": ["percussive_one_shot"]},
        },
        feature_values_by_name={"duration_sec": 0.25, "event_count_estimate": 1.0},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        EligibilityDecision(
            role_name="protected_percussive_one_shot",
            confidence=0.97,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=("FX", "Glitch", "Stutter"),
            broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
            reason="synthetic measured percussion one-shot",
        ),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Drums/Percussion/Generic Percussion/One Shots"


def test_measured_percussive_hit_does_not_steal_specific_bass_one_shot() -> None:
    """The percussion broad bucket must not swallow a specific instrument hit."""
    shared = [
        shared_row("Instruments/Bass/808 Bass/One Shots", 3.0),
        shared_row("Drums/Percussion/Generic Percussion/One Shots", 9.0, {"percussive_one_shot": 0.98}),
    ]
    raw = raw_claim("Instruments/Bass/808 Bass/One Shots", shared, score=3.0)
    measured = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.75,
            "event_count_estimate": 1.0,
            "shape_vote": {"primary_shape": "single_hit", "confidence": 0.82, "onset_count": 1.0},
            "measured_roles": {"percussive_one_shot": 0.90, "primary_roles": ["percussive_one_shot"]},
        },
        feature_values_by_name={"duration_sec": 0.75, "event_count_estimate": 1.0},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        EligibilityDecision(
            role_name="protected_percussive_one_shot",
            confidence=0.90,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=("Instruments", "Bass"),
            broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
            reason="synthetic measured percussion one-shot",
        ),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Bass/808 Bass/One Shots"
