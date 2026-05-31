#!/usr/bin/env python3
# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Aaron Sound Sorter application runner.

This file is intentionally the visible, breakpoint-friendly application entry
point. It wires the product sorter in plain steps instead of hiding startup in a
second ``main()`` function somewhere else.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aaron_sound_sorter.cli import CommandLineParser, request_from_args
from aaron_sound_sorter.brain_lab import run_brain_lab_from_args
from aaron_sound_sorter.domain.policies import BrainVoterPolicy, ConsensusPolicy, PhysicsVoterPolicy, ShapeVoterPolicy
from aaron_sound_sorter.engine.consensus import ConsensusRunner
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.placement_resolver import PlacementResolver
from aaron_sound_sorter.engine.sorter import SortSamplesUseCase
from aaron_sound_sorter.infrastructure.audio_repository import AudioInputRepository
from aaron_sound_sorter.infrastructure.brain_repository import BrainRepository
from aaron_sound_sorter.voters.base import Voter
from aaron_sound_sorter.voters.brain_recall import BalancedRecallBrainVoter, FullBrainVoter
from aaron_sound_sorter.voters.physics_voter import PhysicsVoter
from aaron_sound_sorter.voters.shape_voter import ShapeVoter


def declare_voters(candidate_count: int = 100) -> list[Voter]:
    """Declare the product voters in one visible place."""
    voter_full_brain = FullBrainVoter(BrainVoterPolicy(top_n=candidate_count))
    voter_core_baby = BalancedRecallBrainVoter(BrainVoterPolicy(top_n=candidate_count), lane_name="core_baby", purpose="precision_clean_center")
    voter_spread_baby = BalancedRecallBrainVoter(BrainVoterPolicy(top_n=candidate_count), lane_name="spread_baby", purpose="balanced_clean_diversity")
    voter_outlier_baby = BalancedRecallBrainVoter(BrainVoterPolicy(top_n=candidate_count), lane_name="outlier_baby", purpose="edge_case_recall_not_final_truth")
    voter_physics = PhysicsVoter(PhysicsVoterPolicy(top_n=candidate_count))
    voter_shape = ShapeVoter(ShapeVoterPolicy())
    return [voter_full_brain, voter_core_baby, voter_spread_baby, voter_outlier_baby, voter_physics, voter_shape]


def build_sorter(voters: list[Voter], candidate_count: int = 100) -> SortSamplesUseCase:
    """Build the user-facing sorter from explicit dependencies."""
    brain_repository = BrainRepository()
    audio_repository = AudioInputRepository()
    consensus_runner = ConsensusRunner(ConsensusPolicy(top_n=candidate_count))
    placement_resolver = PlacementResolver()
    arbiter = FamilyClaimArbiter(placement_resolver=placement_resolver)
    return SortSamplesUseCase(
        brain_repository=brain_repository,
        audio_repository=audio_repository,
        consensus_runner=consensus_runner,
        voters=voters,
        arbiter=arbiter,
    )


def run_self_test(voters: list[Voter]) -> int:
    """Run the smallest app-level wiring test."""
    voter_names = [voter.voter_name for voter in voters]
    expected = ["brain_full", "brain_core_baby", "brain_spread_baby", "brain_outlier_baby", "physics", "shape"]
    if voter_names != expected:
        raise RuntimeError(f"Expected multi-brain voters {expected}, got: {voter_names}")
    print("Self-test passed: app has FullBrainVoter, CoreBabyBrainVoter, SpreadBabyBrainVoter, OutlierBabyBrainVoter, PhysicsVoter, and ShapeVoter")
    return 0


def print_sort_summary(summary) -> None:
    """Print the user-facing sort summary."""
    print(f"Processed: {summary.processed_count}")
    print(f"Needs review: {summary.review_count}")
    print(f"Output: {summary.output_root}")
    print(f"Manifest: {summary.manifest_path}")
    if summary.zip_path:
        print(f"ZIP: {summary.zip_path}")


def main(argv: list[str] | None = None) -> int:
    """Run Aaron Sound Sorter in a small, debuggable sequence.

    Version: v20260512_LONG_FX_LABEL_PRESERVE

    The clean two-voter product runner owns sort/self-test. Developer training
    commands such as train-brain and build-eval still live in the legacy command
    module, so this entry point delegates those commands instead of hiding or
    removing them.
    """
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    legacy_commands = {"train-brain", "train-brain-family", "build-eval"}
    if effective_argv and effective_argv[0] in legacy_commands:
        # Import the component API first. It wires the split modules into the
        # former single-file namespace so legacy developer command handlers can
        # still resolve helpers such as run_build_eval.
        from aaron_sound_sorter import api as legacy_api

        legacy_args = legacy_api.normalize_cli_args(effective_argv)
        legacy_parser = legacy_api.build_argument_parser()
        parsed = legacy_parser.parse_args(legacy_args)
        return int(parsed.func(parsed) or 0)

    parser = CommandLineParser()
    args = parser.parse(effective_argv)

    candidate_count = int(getattr(args, "candidate_count", 100))
    voters = declare_voters(candidate_count=candidate_count)

    if args.command_kind == "self-test":
        return run_self_test(voters)
    if args.command_kind == "brain-lab":
        return run_brain_lab_from_args(args)

    request = request_from_args(args)
    sorter = build_sorter(voters=voters, candidate_count=request.candidate_count)
    summary = sorter.run(request)
    print_sort_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
