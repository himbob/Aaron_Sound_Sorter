"""Application wiring for Aaron Sound Sorter."""

from __future__ import annotations

from dataclasses import dataclass

from aaron_sound_sorter.brain_lab import run_brain_lab_from_args
from aaron_sound_sorter.cli import CommandLineParser, request_from_args
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


@dataclass
class AaronSoundSorterApplication:
    """Readable application object used by the runner."""

    parser: CommandLineParser
    sorter: SortSamplesUseCase

    def run(self, argv: list[str] | None = None) -> int:
        """Run the selected command."""
        args = self.parser.parse(argv)
        if args.command_kind == "self-test":
            return self.run_self_test()
        if args.command_kind == "brain-lab":
            return run_brain_lab_from_args(args)
        request = request_from_args(args)
        summary = self.sorter.run(request)
        print(f"Processed: {summary.processed_count}")
        print(f"Needs review: {summary.review_count}")
        print(f"Output: {summary.output_root}")
        print(f"Manifest: {summary.manifest_path}")
        if summary.zip_path:
            print(f"ZIP: {summary.zip_path}")
        return 0

    def run_self_test(self) -> int:
        """Run a simple wiring self-test."""
        if len(self.sorter.voters) != 6:
            raise RuntimeError("Expected exactly six voters")
        names = [voter.voter_name for voter in self.sorter.voters]
        expected = ["brain_full", "brain_core_baby", "brain_spread_baby", "brain_outlier_baby", "physics", "shape"]
        if names != expected:
            raise RuntimeError(f"Unexpected voter wiring: {names}")
        print(
            "Self-test passed: app has FullBrainVoter, CoreBabyBrainVoter, SpreadBabyBrainVoter, OutlierBabyBrainVoter, PhysicsVoter, and ShapeVoter"
        )
        return 0


def declare_voters(candidate_count: int = 100) -> list[Voter]:
    """Declare the product voters in one visible place."""
    voter_full_brain = FullBrainVoter(BrainVoterPolicy(top_n=candidate_count))
    voter_core_baby = BalancedRecallBrainVoter(
        BrainVoterPolicy(top_n=candidate_count), lane_name="core_baby", purpose="precision_clean_center"
    )
    voter_spread_baby = BalancedRecallBrainVoter(
        BrainVoterPolicy(top_n=candidate_count), lane_name="spread_baby", purpose="balanced_clean_diversity"
    )
    voter_outlier_baby = BalancedRecallBrainVoter(
        BrainVoterPolicy(top_n=candidate_count), lane_name="outlier_baby", purpose="edge_case_recall_not_final_truth"
    )
    voter_physics = PhysicsVoter(PhysicsVoterPolicy(top_n=candidate_count))
    voter_shape = ShapeVoter(ShapeVoterPolicy())
    return [voter_full_brain, voter_core_baby, voter_spread_baby, voter_outlier_baby, voter_physics, voter_shape]


def create_application(candidate_count: int = 100) -> AaronSoundSorterApplication:
    """Create the application with explicit dependencies."""
    voters = declare_voters(candidate_count)
    sorter = SortSamplesUseCase(
        brain_repository=BrainRepository(),
        audio_repository=AudioInputRepository(),
        consensus_runner=ConsensusRunner(ConsensusPolicy(top_n=candidate_count)),
        voters=voters,
        arbiter=FamilyClaimArbiter(placement_resolver=PlacementResolver()),
    )
    return AaronSoundSorterApplication(parser=CommandLineParser(), sorter=sorter)
