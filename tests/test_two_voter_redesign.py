from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from aaron_sound_sorter.cli import CommandLineParser, request_from_args
from aaron_sound_sorter.core import FEATURE_NAMES, FP_SIZE
from aaron_sound_sorter.domain.facts import build_shared_audio_facts
from aaron_sound_sorter.domain.models import AudioPhysics, CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.domain.policies import ConsensusPolicy, PhysicsVoterPolicy
from aaron_sound_sorter.engine.consensus import ConsensusRunner
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.voters import BrainVoter, PhysicsVoter, Voter
from aaron_sound_sorter.voters.brain_recall import combine_full_and_balanced_brain_votes


def guess(label: str, rank: int, score: float = 1.0) -> CategoryGuess:
    top = label.split("/", 1)[0] if "/" in label else "_TO_REVIEW"
    return CategoryGuess(
        label=label,
        folder_path=label,
        top_family=top,
        score=score,
        confidence=1.0 / (1.0 + score),
        rank=rank,
        reason="test",
    )


def role_guess(label: str, rank: int, role_signature: dict[str, float]) -> CategoryGuess:
    top = label.split("/", 1)[0] if "/" in label else "_TO_REVIEW"
    return CategoryGuess(
        label=label,
        folder_path=label,
        top_family=top,
        score=1.0,
        confidence=0.5,
        rank=rank,
        reason="test",
        evidence={"candidate_role_signature": role_signature},
    )


def physics_from_values(values: np.ndarray, duration: float = 0.5, status: str = "ok") -> AudioPhysics:
    return AudioPhysics(Path("test.wav"), values.astype(np.float32), duration, status)


def profile(values: dict[str, float]) -> dict:
    return {
        "effective_count": 8,
        "fact_profile_strength": "test",
        "feature_stats": {
            name: {
                "median": value,
                "iqr": 0.10,
                "mad": 0.05,
                "p10": value - 0.12,
                "p90": value + 0.12,
                "reliability": 0.90,
                "valid_count": 4,
            }
            for name, value in values.items()
        },
    }


def finalize_consensus(
    runner: ConsensusRunner,
    brain: VoterResult,
    physics: VoterResult,
    facts: SharedAudioFacts,
):
    """Run the full claim-producing consensus path through DecisionCoreV2."""
    return DecisionCoreV2(raw_consensus=runner).choose(brain, physics, facts)


def test_consensus_can_only_choose_a_shared_category() -> None:
    facts = SharedAudioFacts(False, False, True, True, False)
    brain = VoterResult(
        "brain", [guess("Drums/Short Hits/One Shots", 1), guess("Instruments/Voice/Shout/One Shots", 2)]
    )
    physics = VoterResult(
        "physics", [guess("Instruments/Voice/Shout/One Shots", 1), guess("FX/Textures/Noise/One Shots", 2)]
    )

    decision = finalize_consensus(ConsensusRunner(), brain, physics, facts)

    assert decision.consensus_status == "strong_consensus"
    assert decision.final_label == "Instruments/Voice/Shout/One Shots"
    assert decision.final_label != "Drums/Short Hits/One Shots"


def test_consensus_reviews_when_voters_do_not_overlap() -> None:
    facts = SharedAudioFacts(False, False, True, True, False)
    brain = VoterResult("brain", [guess("Drums/Short Hits/One Shots", 1)])
    physics = VoterResult("physics", [guess("FX/Textures/Noise/One Shots", 1)])

    decision = finalize_consensus(ConsensusRunner(), brain, physics, facts)

    assert decision.consensus_status == "no_voter_consensus"
    assert decision.final_label == ConsensusPolicy().no_consensus_label
    assert decision.final_top == "_TO_REVIEW"


def test_consensus_reviews_when_shared_candidate_is_too_weak() -> None:
    facts = SharedAudioFacts(False, False, False, False, True)
    shared = "FX/Ambiguous Motion/One Shots"
    brain_guesses = [guess(f"Brain/Other/{index}/One Shots", index) for index in range(1, 15)]
    brain_guesses.append(guess(shared, 15))
    physics_guesses = [guess(f"Physics/Other/{index}/One Shots", index) for index in range(1, 12)]
    physics_guesses.append(guess(shared, 12))

    decision = finalize_consensus(
        ConsensusRunner(policy=ConsensusPolicy(max_combined_rank_score=20)),
        VoterResult("brain", brain_guesses),
        VoterResult("physics", physics_guesses),
        facts,
    )

    assert decision.consensus_status == "weak_voter_consensus"
    assert decision.final_label == ConsensusPolicy().weak_consensus_label
    assert decision.shared_winner == shared


def test_consensus_rehomes_strong_vocal_shape_to_instrument_voice_bucket() -> None:
    facts = SharedAudioFacts(
        False,
        False,
        True,
        True,
        False,
        evidence={
            "measured_roles": {"voiced_one_shot": 1.0, "primary_roles": ["voiced_one_shot"]},
            "shape_vote": {"primary_shape": "vocal_phrase", "confidence": 0.98},
        },
    )
    bad_fx_leaf = "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX"
    brain = VoterResult("brain", [role_guess(bad_fx_leaf, 1, {"voiced_one_shot": 0.0})])
    physics = VoterResult("physics", [role_guess(bad_fx_leaf, 2, {"voiced_one_shot": 0.0})])

    decision = finalize_consensus(ConsensusRunner(), brain, physics, facts)

    assert decision.final_label == "Instruments/Voice/Phrase/One Shots"
    assert "Human and Voice" not in decision.folder_path


def test_consensus_releases_measured_beat_loop_from_fx_ambience() -> None:
    facts = SharedAudioFacts(
        False,
        True,
        False,
        False,
        True,
        evidence={
            "measured_roles": {"percussive_drum_loop": 0.90, "primary_roles": ["percussive_drum_loop"]},
            "shape_vote": {"primary_shape": "beat_loop", "confidence": 0.90},
        },
    )
    fx = "FX/Textures/Natural Ambience/Water/Long FX"
    drum_loop = "Drums/Drum Loops/Loops"
    brain = VoterResult(
        "brain",
        [role_guess(fx, 1, {"percussive_drum_loop": 0.0}), role_guess(drum_loop, 11, {"percussive_drum_loop": 0.9})],
    )
    physics = VoterResult(
        "physics",
        [role_guess(fx, 2, {"percussive_drum_loop": 0.0}), role_guess(drum_loop, 10, {"percussive_drum_loop": 0.9})],
    )

    decision = finalize_consensus(ConsensusRunner(), brain, physics, facts)

    assert decision.final_label == drum_loop
    assert decision.consensus_status in {
        "candidate_true_bucket_rescue",
        "final_shape_review_broad_drum_loop_invariant",
        "measured_drum_loop_claim",
    }


def test_consensus_does_not_promote_drum_candidate_from_percussive_role_only() -> None:
    facts = SharedAudioFacts(
        False,
        False,
        True,
        True,
        False,
        evidence={
            "measured_roles": {"percussive_one_shot": 0.80, "primary_roles": ["percussive_one_shot"]},
            "shape_vote": {"primary_shape": "single_hit", "confidence": 0.83},
        },
    )
    glitch = "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/One Shots"
    rimshot = "Drums/Rims and Sticks/Rimshot/One Shots"
    brain = VoterResult(
        "brain",
        [role_guess(glitch, 3, {"percussive_one_shot": 0.75}), role_guess(rimshot, 10, {"percussive_one_shot": 0.92})],
    )
    physics = VoterResult(
        "physics",
        [role_guess(glitch, 1, {"percussive_one_shot": 0.75}), role_guess(rimshot, 4, {"percussive_one_shot": 0.92})],
    )

    decision = finalize_consensus(ConsensusRunner(), brain, physics, facts)

    assert decision.final_label == glitch
    assert decision.consensus_status == "strong_consensus"


def test_consensus_blocks_formant_light_voice_bucket_without_strong_fit() -> None:
    facts = SharedAudioFacts(
        False,
        False,
        True,
        True,
        False,
        evidence={
            "measured_roles": {"voiced_one_shot": 0.86, "primary_roles": ["voiced_one_shot"]},
            "shape_vote": {"primary_shape": "hit_with_tail", "confidence": 0.73},
        },
    )
    bird = "FX/Animals and Creatures/Bird/One Shots"
    brain = VoterResult("brain", [role_guess(bird, 6, {"voiced_one_shot": 0.61})])
    physics = VoterResult("physics", [role_guess(bird, 3, {"voiced_one_shot": 0.61})])

    decision = finalize_consensus(ConsensusRunner(), brain, physics, facts)

    assert "Human and Voice" not in decision.folder_path
    assert not decision.folder_path.startswith("Instruments/Voice")
    assert decision.final_top in {"FX", "_TO_REVIEW"}


def test_shared_audio_facts_carry_all_named_features() -> None:
    fp = np.linspace(0.0, 1.0, FP_SIZE, dtype=np.float32)
    physics = physics_from_values(fp, duration=2.0, status="ok")
    facts = build_shared_audio_facts(physics)

    assert facts.feature_count == FP_SIZE
    assert len(facts.feature_values_by_name) == FP_SIZE
    assert set(FEATURE_NAMES) == set(facts.feature_values_by_name)
    assert len(facts.feature_vector) == FP_SIZE
    assert facts.evidence["feature_count"] == FP_SIZE
    assert len(facts.evidence["feature_values_by_name"]) == FP_SIZE
    assert "rhythm_loop_structure" in facts.feature_groups
    assert "pitch_harmonic" in facts.feature_groups


def test_broken_or_tiny_is_explicit_review_from_both_voters() -> None:
    fp = np.zeros((FP_SIZE,), dtype=np.float32)
    physics = physics_from_values(fp, duration=0.0, status="read_error:test")
    facts = build_shared_audio_facts(physics)

    brain = BrainVoter().vote(physics, facts, {})
    physics_vote = PhysicsVoter().vote(physics, facts, {})
    decision = finalize_consensus(ConsensusRunner(), brain, physics_vote, facts)

    assert brain.guesses[0].label == ConsensusPolicy().broken_or_tiny_label
    assert physics_vote.guesses[0].label == ConsensusPolicy().broken_or_tiny_label
    assert decision.consensus_status == "broken_or_tiny_review"
    assert decision.final_label == ConsensusPolicy().broken_or_tiny_label


def test_physics_voter_prefers_matching_learned_fact_profile() -> None:
    voice = "Instruments/Voice/Shout/One Shots"
    drum_loop = "Drums/Percussive Loop/Loops"
    feature_values = {
        "f0_voiced_ratio": 0.82,
        "pitch_confidence": 0.76,
        "spectral_flatness_mean": 0.18,
        "temporal_centroid_ratio": 0.24,
        "onset_span_ratio": 0.18,
        "event_rate_hz": 0.20,
    }
    brain = {
        "labels": [drum_loop, voice],
        "top_by_label": {drum_loop: "Drums", voice: "Instruments"},
        "structure_by_label": {drum_loop: "loop", voice: "one_shot"},
        "category_fact_profiles": {
            voice: profile(feature_values),
            drum_loop: profile(
                {
                    "f0_voiced_ratio": 0.05,
                    "pitch_confidence": 0.12,
                    "spectral_flatness_mean": 0.80,
                    "temporal_centroid_ratio": 0.55,
                    "onset_span_ratio": 0.75,
                    "event_rate_hz": 2.00,
                }
            ),
        },
    }
    fp = np.zeros((FP_SIZE,), dtype=np.float32)
    for name, value in feature_values.items():
        fp[FEATURE_NAMES.index(name)] = value
    physics = physics_from_values(fp, duration=0.45, status="ok")
    facts = build_shared_audio_facts(physics)

    result = PhysicsVoter(PhysicsVoterPolicy(top_n=2)).vote(physics, facts, brain)

    assert result.guesses[0].label == voice
    assert result.guesses[0].top_family == "Instruments"
    assert result.guesses[1].label == drum_loop
    assert result.guesses[1].evidence["structure_gate"] == "candidate_pre_filtered"
    assert result.guesses[1].evidence["structure_penalty"] > 0.0


def test_voter_base_class_enforces_vote_contract() -> None:
    class IncompleteVoter(Voter):
        voter_name = "incomplete"

    try:
        IncompleteVoter()
    except TypeError:
        pass
    else:
        raise AssertionError("Voter subclasses must implement vote()")


def test_runner_declares_multi_brain_concrete_voters() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import Aaron_Sound_Sorter as runner

    voters = runner.declare_voters()

    assert [v.voter_name for v in voters] == [
        "brain_full",
        "brain_core_baby",
        "brain_spread_baby",
        "brain_outlier_baby",
        "physics",
        "shape",
    ]
    assert all(hasattr(v, "vote") for v in voters)
    assert len(voters) == 6


def test_recall_lanes_are_diagnostic_by_default_for_product_brain_vote() -> None:
    full = VoterResult("brain_full", [guess("Instruments/Saxophone/Generic Saxophone/Loops", 1)])
    baby = VoterResult("brain_core_baby", [guess("FX/Human and Voice FX/Crowd/Long FX", 1)])

    combined = combine_full_and_balanced_brain_votes(full_result=full, baby_results={}, max_guesses=10)
    labels = [candidate.label for candidate in combined.guesses]

    assert labels == ["Instruments/Saxophone/Generic Saxophone/Loops"]
    assert combined.diagnostics["architecture"] == "full_brain_only_product_vote"

    explicit = combine_full_and_balanced_brain_votes(full_result=full, baby_results={"core_baby": baby}, max_guesses=10)
    explicit_labels = [candidate.label for candidate in explicit.guesses]

    assert "FX/Human and Voice FX/Crowd/Long FX" in explicit_labels
    assert explicit.diagnostics["architecture"] == "weighted_brain_ensemble"


def test_weighted_brain_ensemble_prefers_core_spread_agreement_over_bad_full_lane() -> None:
    full = VoterResult("brain_full", [guess("FX/Designed Noise FX/Alarm/Long FX", 1)])
    core = VoterResult("brain_core_baby", [guess("Instruments/Woodwinds/Saxophone/One Shots", 1)])
    spread = VoterResult("brain_spread_baby", [guess("Instruments/Woodwinds/Saxophone/One Shots", 1)])

    combined = combine_full_and_balanced_brain_votes(
        full_result=full,
        baby_results={"core_baby": core, "spread_baby": spread},
        max_guesses=10,
    )

    assert combined.guesses[0].label == "Instruments/Woodwinds/Saxophone/One Shots"
    assert combined.guesses[0].evidence["brain_candidate_lanes"] == ["core_baby", "spread_baby"]
    assert combined.diagnostics["architecture"] == "weighted_brain_ensemble"


def test_weighted_brain_ensemble_does_not_let_outlier_only_hijack_full_brain() -> None:
    full = VoterResult("brain_full", [guess("Drums/Kick Drums/Generic Kick/One Shots", 1)])
    outlier = VoterResult("brain_outlier_baby", [guess("FX/Animals and Creatures/Cat/Long FX", 1)])

    combined = combine_full_and_balanced_brain_votes(
        full_result=full,
        baby_results={"outlier_baby": outlier},
        max_guesses=10,
    )

    assert combined.guesses[0].label == "Drums/Kick Drums/Generic Kick/One Shots"
    assert combined.guesses[0].evidence["has_full_brain_support"] is True


def test_cli_uses_baby_brain_ensemble_by_default_with_disable_escape_hatch() -> None:
    parser = CommandLineParser()
    default_args = parser.parse(["sort", "input.wav", "out"])
    default_request = request_from_args(default_args)

    assert default_request.use_baby_brains_in_sort is True
    assert default_request.use_harmonic_brains_in_sort is False

    disabled_args = parser.parse(["sort", "input.wav", "out", "--disable-baby-brain-ensemble"])
    disabled_request = request_from_args(disabled_args)

    assert disabled_request.use_baby_brains_in_sort is False

    opt_in_args = parser.parse(["sort", "input.wav", "out", "--use-harmonic-brains-in-sort"])
    opt_in_request = request_from_args(opt_in_args)

    assert opt_in_request.use_baby_brains_in_sort is True
    assert opt_in_request.use_harmonic_brains_in_sort is True


def test_cli_exposes_brain_lab_without_sorting_arguments() -> None:
    args = CommandLineParser().parse(["brain-lab", "sample.wav", "--top-n", "3"])

    assert args.command_kind == "brain-lab"
    assert args.input_path == "sample.wav"
    assert args.top_n == 3


def test_cli_help_is_clean_product_runner() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "Aaron_Sound_Sorter.py"), "--help"],
        cwd=str(root),
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0
    assert "sort" in result.stdout
    assert "build-eval" not in result.stdout
