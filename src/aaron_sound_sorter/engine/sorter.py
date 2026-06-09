# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Clean product sort use case."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from aaron_sound_sorter.domain.facts import build_shared_audio_facts
from aaron_sound_sorter.domain.models import (
    AudioPhysics,
    CategoryGuess,
    SortFileResult,
    SortRequest,
    SortSummary,
    VoterResult,
)
from aaron_sound_sorter.engine.audio_analysis_cache import AudioAnalysisCache, AudioAnalysisPacket
from aaron_sound_sorter.engine.consensus import ConsensusRunner
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_order_policy import family_order_policy_summary
from aaron_sound_sorter.engine.role_audit import build_parent_role_audit
from aaron_sound_sorter.engine.sort_timing import SortTimingProfiler, write_sort_timing_reports
from aaron_sound_sorter.features import (
    make_direct_body_fingerprint_safe,
    make_fingerprint_safe,
    make_harmonic_core_fingerprint_safe,
    third_party_feature_profile,
    wetness_profile,
)
from aaron_sound_sorter.infrastructure.audio_repository import AudioInputRepository
from aaron_sound_sorter.infrastructure.brain_repository import BrainRepository
from aaron_sound_sorter.infrastructure.report_writer import SortReportWriter
from aaron_sound_sorter.role_gate import dynamic_role_gate
from aaron_sound_sorter.voters.base import Voter
from aaron_sound_sorter.voters.brain_recall import combine_full_and_balanced_brain_votes, make_disabled_balanced_result


class SortSamplesUseCase:
    """Orchestrate one user-facing sort run."""

    def __init__(
        self,
        brain_repository: BrainRepository,
        audio_repository: AudioInputRepository,
        consensus_runner: ConsensusRunner,
        voters: list[Voter],
        arbiter: FamilyClaimArbiter,
    ) -> None:
        self.brain_repository = brain_repository
        self.audio_repository = audio_repository
        self.consensus_runner = consensus_runner
        self.decision_core = DecisionCoreV2(raw_consensus=consensus_runner, arbiter=arbiter)
        self.voters = voters
        self.analysis_cache = AudioAnalysisCache()
        self.sort_timing = SortTimingProfiler()

    def run(self, request: SortRequest) -> SortSummary:
        """Sort audio files using the configured brain family and voters.

        Args:
            request: User-facing sort request.

        Returns:
            Sort summary with manifest, summary, optional ZIP, and file results.

        Side Effects:
            Creates output folders/reports under ``request.output_dir`` and
            writes timing diagnostics there. It does not write cache data to the
            repository.

        Raises:
            Propagates repository, analysis, voter, or writer exceptions.

        Important Constraints:
            Timing and cache data are diagnostics only and must not change
            category decisions.
        """
        self.analysis_cache.reset()
        self.sort_timing.reset()
        with self.sort_timing.stage("load_brains"):
            brain = self.brain_repository.load(request.brain_path)
            baby_brains = self.load_baby_brains_if_available(request)
            harmonic_baby_brains = self.load_harmonic_baby_brains_if_available(request)
        if any(baby_brains.values()):
            brain["_multi_baby_brains_loaded"] = True
        if any(harmonic_baby_brains.values()):
            brain["_harmonic_baby_brains_loaded"] = True
        with self.sort_timing.stage("prepare_input"):
            prepared_input = self.audio_repository.prepare(request.input_path, request.output_dir)
        writer = SortReportWriter(request.output_dir, write_zip=request.write_zip)
        with self.sort_timing.stage("writer_start"):
            writer.start()
        with self.sort_timing.stage("classify_files"):
            classified_results = self.classify_audio_files(
                prepared_input.audio_files,
                brain,
                baby_brains,
                harmonic_baby_brains,
                use_baby_brains_in_sort=request.use_baby_brains_in_sort,
                use_harmonic_brains_in_sort=request.use_harmonic_brains_in_sort,
                max_workers=request.sort_workers,
            )
        with self.sort_timing.stage("place_files"):
            results = [self.place_classified_file(writer, result) for result in classified_results]
        with self.sort_timing.stage("write_reports"):
            summary = writer.finish(results)
        with self.sort_timing.stage("write_timing_profile"):
            write_sort_timing_reports(request.output_dir, self.sort_timing.snapshot())
        return summary

    def load_baby_brains_if_available(self, request: SortRequest) -> dict[str, dict[str, Any] | None]:
        """Load optional baby brains by job. Missing baby brains are allowed."""
        full_path = Path(request.brain_path).expanduser()
        default_core = full_path.with_name("stage4_folder_brain_core_baby.json")
        default_spread = full_path.with_name("stage4_folder_brain_spread_baby.json")
        default_outlier = full_path.with_name("stage4_folder_brain_outlier_baby.json")
        legacy = request.baby_brain_path or full_path.with_name("stage4_folder_brain_baby.json")
        paths = {
            "core_baby": request.core_baby_brain_path or default_core,
            "spread_baby": request.spread_baby_brain_path
            or (legacy if Path(legacy).expanduser().exists() else default_spread),
            "outlier_baby": request.outlier_baby_brain_path or default_outlier,
        }
        loaded: dict[str, dict[str, Any] | None] = {}
        for lane, candidate in paths.items():
            candidate = Path(candidate).expanduser()
            loaded[lane] = self.brain_repository.load(candidate) if candidate.exists() else None
        return loaded

    def load_harmonic_baby_brains_if_available(self, request: SortRequest) -> dict[str, dict[str, Any] | None]:
        """Load optional harmonic-trained baby brains.

        These must be trained on harmonic-core fingerprints.  Raw baby brains
        are intentionally not reused here because that compares transformed
        query data to a different training distribution.
        """
        full_path = Path(request.brain_path).expanduser()
        paths = {
            "harmonic_core_baby": request.harmonic_core_baby_brain_path
            or full_path.with_name("stage4_folder_brain_harmonic_core_baby.json"),
            "harmonic_spread_baby": request.harmonic_spread_baby_brain_path
            or full_path.with_name("stage4_folder_brain_harmonic_spread_baby.json"),
            "harmonic_outlier_baby": request.harmonic_outlier_baby_brain_path
            or full_path.with_name("stage4_folder_brain_harmonic_outlier_baby.json"),
        }
        loaded: dict[str, dict[str, Any] | None] = {}
        for lane, candidate in paths.items():
            candidate = Path(candidate).expanduser()
            loaded[lane] = self.brain_repository.load(candidate) if candidate.exists() else None
        return loaded

    def sort_one_file(
        self,
        audio_file: Path,
        brain: dict[str, Any],
        writer: SortReportWriter,
        baby_brains: dict[str, dict[str, Any] | None] | None = None,
        harmonic_baby_brains: dict[str, dict[str, Any] | None] | None = None,
        *,
        use_baby_brains_in_sort: bool = True,
        use_harmonic_brains_in_sort: bool = False,
    ) -> SortFileResult:
        """Sort one audio file and write it to the output tree."""
        result = self.classify_one_file(
            audio_file,
            brain,
            baby_brains,
            harmonic_baby_brains,
            use_baby_brains_in_sort=use_baby_brains_in_sort,
            use_harmonic_brains_in_sort=use_harmonic_brains_in_sort,
        )
        return self.place_classified_file(writer, result)

    def classify_audio_files(
        self,
        audio_files: list[Path],
        brain: dict[str, Any],
        baby_brains: dict[str, dict[str, Any] | None] | None = None,
        harmonic_baby_brains: dict[str, dict[str, Any] | None] | None = None,
        *,
        use_baby_brains_in_sort: bool = True,
        use_harmonic_brains_in_sort: bool = False,
        max_workers: int = 1,
    ) -> list[SortFileResult]:
        """Classify files in input order, optionally parallelizing analysis/votes.

        Report writing and file placement stay sequential.  The expensive work
        here is audio analysis plus independent voter scoring, both of which are
        per-file and mostly numpy/IO heavy.
        """
        if max_workers <= 1 or len(audio_files) <= 1:
            return [
                self.classify_one_file(
                    audio_file,
                    brain,
                    baby_brains,
                    harmonic_baby_brains,
                    use_baby_brains_in_sort=use_baby_brains_in_sort,
                    use_harmonic_brains_in_sort=use_harmonic_brains_in_sort,
                )
                for audio_file in audio_files
            ]
        results: list[SortFileResult | None] = [None] * len(audio_files)
        worker_count = max(1, min(int(max_workers), len(audio_files)))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="aaron-sort") as executor:
            futures = {
                executor.submit(
                    self.classify_one_file,
                    audio_file,
                    brain,
                    baby_brains,
                    harmonic_baby_brains,
                    use_baby_brains_in_sort=use_baby_brains_in_sort,
                    use_harmonic_brains_in_sort=use_harmonic_brains_in_sort,
                ): index
                for index, audio_file in enumerate(audio_files)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [result for result in results if result is not None]

    def classify_one_file(
        self,
        audio_file: Path,
        brain: dict[str, Any],
        baby_brains: dict[str, dict[str, Any] | None] | None = None,
        harmonic_baby_brains: dict[str, dict[str, Any] | None] | None = None,
        *,
        use_baby_brains_in_sort: bool = True,
        use_harmonic_brains_in_sort: bool = False,
    ) -> SortFileResult:
        """Classify one audio file without placing it on disk."""
        with self.sort_timing.file_stage(audio_file, "analysis_packet"):
            analysis_packet = self.build_audio_analysis_packet(
                audio_file=audio_file,
                harmonic_baby_brains=harmonic_baby_brains or {},
            )
        physics = analysis_packet.audio_physics
        wet_profile = analysis_packet.wetness_profile
        harmonic_physics = analysis_packet.harmonic_physics
        facts = build_shared_audio_facts(physics)
        facts.evidence["wetness_profile"] = wet_profile
        facts.evidence["harmonic_core_status"] = (
            harmonic_physics.read_status if harmonic_physics is not None else "not_computed"
        )
        facts.evidence["harmonic_core_duration_sec"] = (
            harmonic_physics.duration_sec if harmonic_physics is not None else 0.0
        )
        facts.evidence["audio_analysis_packet_summary"] = analysis_packet.summary()
        facts.evidence["audio_analysis_cache_stats"] = dict(analysis_packet.cache_stats_after)
        facts.evidence["top_family_order_policy"] = family_order_policy_summary()
        with self.sort_timing.file_stage(audio_file, "dynamic_role_gate"):
            role_gate = dynamic_role_gate(physics, facts, brain)
        facts.evidence["dynamic_role_gate"] = role_gate.evidence()
        with self.sort_timing.file_stage(audio_file, "collect_votes"):
            brain_votes, physics_votes = self.collect_votes(
                physics,
                facts,
                brain,
                baby_brains or {},
                harmonic_baby_brains or {},
                harmonic_physics=harmonic_physics,
                use_baby_brains_in_sort=use_baby_brains_in_sort,
                use_harmonic_brains_in_sort=use_harmonic_brains_in_sort,
            )
        with self.sort_timing.file_stage(audio_file, "decision_core"):
            decision = self.decision_core.choose(brain_votes, physics_votes, facts)
        facts.evidence["parent_role_audit"] = build_parent_role_audit(
            facts=facts,
            brain_votes=brain_votes,
            physics_votes=physics_votes,
            decision=decision,
        )
        facts.evidence["sort_timing_profile"] = self.sort_timing.file_summary(audio_file)
        initial = SortFileResult(
            source_path=audio_file,
            placed_path=None,
            physics=physics,
            facts=facts,
            brain_votes=brain_votes,
            physics_votes=physics_votes,
            decision=decision,
        )
        return initial

    def build_audio_analysis_packet(
        self,
        *,
        audio_file: Path,
        harmonic_baby_brains: dict[str, dict[str, Any] | None],
    ) -> AudioAnalysisPacket:
        """Build one shared measured-evidence packet for the current file.

        The packet is per-run and in memory only.  It reuses expensive analysis
        through ``AudioAnalysisCache`` but never caches final placement labels,
        voter winners, training data, or manual corrections.
        """
        stats_before = self.analysis_cache.stats()
        physics = self.analysis_cache.audio_physics(audio_file, analyze_audio_file)
        wet_profile = self.analysis_cache.wetness_profile(audio_file, analyze_wetness_profile)
        harmonic_physics = self.analyze_harmonic_core_if_needed(
            audio_file=audio_file,
            wet_profile=wet_profile,
            harmonic_baby_brains=harmonic_baby_brains or {},
            analysis_cache=self.analysis_cache,
        )
        stats_after = self.analysis_cache.stats()
        return AudioAnalysisPacket(
            audio_physics=physics,
            wetness_profile=wet_profile,
            harmonic_physics=harmonic_physics,
            cache_stats_before=stats_before,
            cache_stats_after=stats_after,
        )

    @staticmethod
    def analyze_harmonic_core_if_needed(
        *,
        audio_file: Path,
        wet_profile: dict[str, Any],
        harmonic_baby_brains: dict[str, dict[str, Any] | None],
        analysis_cache: AudioAnalysisCache | None = None,
    ) -> AudioPhysics | None:
        """Compute harmonic-core only when a downstream consumer can use it."""
        wetness_score = wetness_score_from_profile(wet_profile)
        has_harmonic_brain = any(harmonic_baby_brains.values())
        harmonic_recall_needed = bool(has_harmonic_brain and wetness_score >= 0.35)
        dry_probe_may_need_harmonic = bool(wetness_score >= 0.55)
        if not (harmonic_recall_needed or dry_probe_may_need_harmonic):
            return None
        if analysis_cache is not None:
            return analysis_cache.harmonic_physics(
                audio_file,
                lambda path: analyze_harmonic_core_with_optional_wet_profile(path, wet_profile),
            )
        return analyze_harmonic_core_with_optional_wet_profile(audio_file, wet_profile)

    @staticmethod
    def place_classified_file(writer: SortReportWriter, result: SortFileResult) -> SortFileResult:
        """Place one already classified file and attach the destination path."""
        placed_path = writer.place_file(result)
        return SortFileResult(
            source_path=result.source_path,
            placed_path=placed_path,
            physics=result.physics,
            facts=result.facts,
            brain_votes=result.brain_votes,
            physics_votes=result.physics_votes,
            decision=result.decision,
        )

    def collect_votes(
        self,
        physics: AudioPhysics,
        facts,
        brain: dict[str, Any],
        baby_brains: dict[str, dict[str, Any] | None] | None = None,
        harmonic_baby_brains: dict[str, dict[str, Any] | None] | None = None,
        *,
        harmonic_physics: AudioPhysics | None = None,
        use_baby_brains_in_sort: bool = True,
        use_harmonic_brains_in_sort: bool = False,
    ):
        """Return combined BrainVoter result, PhysicsVoter result, and diagnostics.

        Voter order is intentionally visible:
          0 full brain        stability lane
          1 core baby         precision clean-center lane
          2 spread baby       balanced clean-diversity lane
          3 outlier baby      edge-case recall lane, never final truth alone
          4 physics           full-profile plausibility lane
          5+ diagnostics      shape / future voters
        """
        if len(self.voters) < 5:
            raise ValueError("Product sorter expects FullBrain, three BabyBrains, and PhysicsVoter")
        baby_brains = baby_brains or {}
        harmonic_baby_brains = harmonic_baby_brains or {}
        full_brain_votes = self.voters[0].vote(physics, facts, brain)
        lane_to_index = {"core_baby": 1, "spread_baby": 2, "outlier_baby": 3}
        baby_vote_results: dict[str, VoterResult] = {}
        for lane, voter_index in lane_to_index.items():
            lane_brain = baby_brains.get(lane)
            if lane_brain is not None:
                lane_result = self.voters[voter_index].vote(physics, facts, lane_brain)
            else:
                lane_result = make_disabled_balanced_result(f"no {lane} brain file provided or found", lane)
            baby_vote_results[lane] = lane_result
            facts.evidence[f"{lane}_vote_result"] = voter_result_digest(lane_result)
        # Optional harmonic-core recall lane for wet/smeared pitched material.
        # It is diagnostic by default and must use separately trained harmonic
        # baby brains.  Reusing raw baby brains here mixes feature domains and
        # makes the voter stack look more certain than it really is.
        harmonic_vote_results: dict[str, VoterResult] = {}
        wet_profile = facts.evidence.get("wetness_profile", {}) if isinstance(facts.evidence, dict) else {}
        wetness_score = float(wet_profile.get("wetness_score", 0.0) or 0.0) if isinstance(wet_profile, dict) else 0.0
        harmonic_ok = bool(harmonic_physics is not None and harmonic_physics.read_status == "ok")
        has_harmonic_brain = any(harmonic_baby_brains.values())
        harmonic_enabled = bool(harmonic_ok and wetness_score >= 0.35 and has_harmonic_brain)
        facts.evidence["harmonic_core_recall_enabled"] = harmonic_enabled
        if not has_harmonic_brain:
            harmonic_reason = "harmonic core recall disabled: no harmonic-trained baby brain files provided or found"
        else:
            harmonic_reason = f"wetness_score={wetness_score:.3f}; harmonic_status={getattr(harmonic_physics, 'read_status', 'missing')}"
        facts.evidence["harmonic_core_recall_reason"] = harmonic_reason
        facts.evidence["harmonic_core_recall_training_domain"] = "harmonic_core" if has_harmonic_brain else "missing"
        if harmonic_enabled and harmonic_physics is not None:
            harmonic_lane_map = {
                "harmonic_core_baby": 1,
                "harmonic_spread_baby": 2,
                "harmonic_outlier_baby": 3,
            }
            for harmonic_lane, voter_index in harmonic_lane_map.items():
                lane_brain = harmonic_baby_brains.get(harmonic_lane)
                if lane_brain is None:
                    lane_result = make_disabled_balanced_result(
                        f"no {harmonic_lane} brain file provided or found", harmonic_lane
                    )
                else:
                    lane_result = self.voters[voter_index].vote(harmonic_physics, facts, lane_brain)
                    lane_result = relabel_voter_result(lane_result, harmonic_lane, prefix_reason="harmonic_core")
                harmonic_vote_results[harmonic_lane] = lane_result
                facts.evidence[f"{harmonic_lane}_vote_result"] = voter_result_digest(lane_result)
        else:
            for harmonic_lane in ["harmonic_core_baby", "harmonic_spread_baby", "harmonic_outlier_baby"]:
                lane_result = make_disabled_balanced_result(harmonic_reason, harmonic_lane)
                harmonic_vote_results[harmonic_lane] = lane_result
                facts.evidence[f"{harmonic_lane}_vote_result"] = voter_result_digest(lane_result)

        product_baby_results: dict[str, VoterResult] = {
            lane: result for lane, result in baby_vote_results.items() if result.guesses
        }
        if use_harmonic_brains_in_sort:
            product_baby_results.update(
                {lane: result for lane, result in harmonic_vote_results.items() if result.guesses}
            )
        facts.evidence["baby_brains_affect_product_vote"] = bool(product_baby_results)
        facts.evidence["baby_brains_affect_product_vote_reason"] = (
            "default_weighted_brain_ensemble_uses_loaded_raw_baby_lanes"
            if product_baby_results
            else "no_loaded_raw_baby_lanes_available"
        )
        facts.evidence["harmonic_brains_affect_product_vote"] = bool(
            use_harmonic_brains_in_sort and any(result.guesses for result in harmonic_vote_results.values())
        )
        facts.evidence["brain_ensemble_product_policy"] = "full_plus_loaded_raw_baby_rank_fusion_default"

        # Legacy field names remain so older review scripts keep working.
        facts.evidence["full_brain_vote_result"] = voter_result_digest(full_brain_votes)
        facts.evidence["baby_brain_vote_result"] = voter_result_digest(baby_vote_results.get("spread_baby"))
        brain_votes = combine_full_and_balanced_brain_votes(
            full_result=full_brain_votes,
            baby_results=product_baby_results,
            max_guesses=max(getattr(self.consensus_runner.policy, "top_n", 100), 120),
            facts=facts,
        )
        facts.evidence["brain_ensemble_vote_result"] = voter_result_digest(brain_votes)
        facts.evidence["brain_ensemble_lane_weights"] = brain_votes.diagnostics.get("lane_weights", {})
        facts.evidence["brain_ensemble_lanes_affecting_product_vote"] = brain_votes.diagnostics.get(
            "lanes_affecting_product_vote", []
        )
        physics_votes = self.voters[4].vote(physics, facts, brain)
        for extra_voter in self.voters[5:]:
            extra_result = extra_voter.vote(physics, facts, brain)
            if extra_result.voter_name == "shape":
                shape_vote = extra_result.diagnostics.get("shape_vote", {})
                facts.evidence["shape_vote"] = shape_vote
                facts.evidence["shape_vote_result"] = {
                    "voter_name": extra_result.voter_name,
                    "guesses": [guess.evidence for guess in extra_result.guesses[:3]],
                }

        self._attach_dry_wet_conflict_probe(
            facts=facts,
            full_brain_votes=full_brain_votes,
            brain_votes=brain_votes,
            physics_votes=physics_votes,
            harmonic_physics=harmonic_physics,
            brain=brain,
            baby_brains=baby_brains,
        )
        return brain_votes, physics_votes

    def _attach_dry_wet_conflict_probe(
        self,
        *,
        facts,
        full_brain_votes: VoterResult,
        brain_votes: VoterResult,
        physics_votes: VoterResult,
        harmonic_physics: AudioPhysics | None,
        brain: dict[str, Any],
        baby_brains: dict[str, dict[str, Any] | None],
    ) -> None:
        """Attach a non-destructive dry-core probe for wet unresolved conflicts.

        This is intentionally diagnostic evidence.  It does not replace the normal
        full-file brain or physics voters, and it does not affect files where the
        committee is already clear.  The arbiter may use this witness only as a
        tie-breaker/conflict probe to stop wet/reverb tails from over-promoting
        Voice, Strings, Harmonica, or generic instrument buckets.
        """
        if not isinstance(getattr(facts, "evidence", None), dict):
            return
        probe = {
            "enabled": False,
            "reason": "not_needed",
        }
        facts.evidence["dry_wet_conflict_probe"] = probe
        wet_profile = facts.evidence.get("wetness_profile", {})
        wetness_score = 0.0
        if isinstance(wet_profile, dict):
            try:
                wetness_score = float(wet_profile.get("wetness_score", 0.0) or 0.0)
            except Exception:
                wetness_score = 0.0
        if harmonic_physics is None or harmonic_physics.read_status != "ok":
            probe.update({"reason": "harmonic_core_not_available", "wetness_score": wetness_score})
            return
        if wetness_score < 0.55:
            probe.update({"reason": "wetness_below_probe_threshold", "wetness_score": wetness_score})
            return
        if not self._voter_results_show_source_disagreement(brain_votes, physics_votes, facts):
            probe.update({"reason": "committee_source_identity_already_clear", "wetness_score": wetness_score})
            return

        dry_full = relabel_voter_result(
            self.voters[0].vote(harmonic_physics, facts, brain),
            "dry_core_full_brain",
            prefix_reason="dry_core",
        )
        dry_babies: dict[str, VoterResult] = {}
        lane_to_index = {"dry_core_core_baby": 1, "dry_core_spread_baby": 2, "dry_core_outlier_baby": 3}
        raw_to_dry = {
            "dry_core_core_baby": "core_baby",
            "dry_core_spread_baby": "spread_baby",
            "dry_core_outlier_baby": "outlier_baby",
        }
        for dry_lane, voter_index in lane_to_index.items():
            raw_lane = raw_to_dry[dry_lane]
            lane_brain = baby_brains.get(raw_lane)
            if lane_brain is None:
                dry_babies[dry_lane] = make_disabled_balanced_result(
                    f"no {raw_lane} brain file provided or found for dry-core conflict probe",
                    dry_lane,
                )
            else:
                dry_babies[dry_lane] = relabel_voter_result(
                    self.voters[voter_index].vote(harmonic_physics, facts, lane_brain),
                    dry_lane,
                    prefix_reason="dry_core",
                )
        dry_product_babies = {lane: result for lane, result in dry_babies.items() if result.guesses}
        dry_ensemble = combine_full_and_balanced_brain_votes(
            full_result=dry_full,
            baby_results=dry_product_babies,
            max_guesses=max(getattr(self.consensus_runner.policy, "top_n", 100), 120),
            facts=facts,
        )
        dry_physics = self.voters[4].vote(harmonic_physics, facts, brain)
        facts.evidence["dry_core_full_brain_vote_result"] = voter_result_digest(dry_full)
        for lane, result in dry_babies.items():
            facts.evidence[f"{lane}_vote_result"] = voter_result_digest(result)
        facts.evidence["dry_core_brain_ensemble_vote_result"] = voter_result_digest(dry_ensemble)
        facts.evidence["dry_core_physics_vote_result"] = voter_result_digest(dry_physics)
        dry_paths = self._top_paths_from_results(dry_ensemble, dry_physics, limit=8)
        wet_paths = self._top_paths_from_results(brain_votes, physics_votes, limit=8)
        dry_buckets = [self._source_bucket_from_path(path) for path in dry_paths]
        wet_buckets = [self._source_bucket_from_path(path) for path in wet_paths]
        probe.update(
            {
                "enabled": True,
                "reason": "wet_source_identity_disagreement",
                "wetness_score": wetness_score,
                "wet_top_paths": wet_paths,
                "wet_source_buckets": wet_buckets,
                "dry_top_paths": dry_paths,
                "dry_source_buckets": dry_buckets,
                "dry_has_voice": "voice" in dry_buckets,
                "dry_has_sax_or_reed": any(bucket in {"sax_reed", "brass_woodwind"} for bucket in dry_buckets),
                "dry_has_synth_or_bells": any(bucket in {"synth", "mallet_bells"} for bucket in dry_buckets),
                "dry_has_non_voice_instrument": any(
                    bucket
                    in {
                        "sax_reed",
                        "brass_woodwind",
                        "synth",
                        "mallet_bells",
                        "bass",
                        "guitar",
                        "keys",
                        "strings",
                        "instrument_loop",
                    }
                    for bucket in dry_buckets
                ),
            }
        )

    @staticmethod
    def _top_paths_from_results(*results: VoterResult, limit: int = 8) -> list[str]:
        """Return unique top folder paths from voter results."""
        paths: list[str] = []
        seen: set[str] = set()
        for result in results:
            for guess in result.guesses[:limit]:
                path = str(guess.folder_path or guess.label or "").replace("\\", "/").strip("/")
                if path and path not in seen:
                    paths.append(path)
                    seen.add(path)
                if len(paths) >= limit:
                    return paths
        return paths

    @staticmethod
    def _source_bucket_from_path(path: str) -> str:
        """Map an internal candidate path to a broad source identity bucket."""
        normalized = str(path or "").lower().replace("\\", "/")
        if "voice" in normalized or "vocal" in normalized or "spoken" in normalized:
            return "voice"
        if (
            "sax" in normalized
            or "saxophone" in normalized
            or "harmonica" in normalized
            or "clarinet" in normalized
            or "bassoon" in normalized
        ):
            return "sax_reed"
        if "brass" in normalized or "woodwind" in normalized or "flute" in normalized or "horn" in normalized:
            return "brass_woodwind"
        if "synth" in normalized or "lead" in normalized or "pad" in normalized or "pluck" in normalized:
            return "synth"
        if "mallet" in normalized or "bell" in normalized or "vibraphone" in normalized or "marimba" in normalized:
            return "mallet_bells"
        if "/bass" in normalized or "bass/" in normalized or "808" in normalized:
            return "bass"
        if "guitar" in normalized:
            return "guitar"
        if "keys" in normalized or "piano" in normalized or "organ" in normalized:
            return "keys"
        if "string" in normalized or "violin" in normalized or "cello" in normalized:
            return "strings"
        if "instrument loops" in normalized or "mixed musical" in normalized:
            return "instrument_loop"
        if normalized.startswith("drums/"):
            return "drums"
        if normalized.startswith("fx/"):
            return "fx"
        if normalized.startswith("textures/"):
            return "textures"
        return "unknown"

    def _voter_results_show_source_disagreement(
        self,
        brain_votes: VoterResult,
        physics_votes: VoterResult,
        facts,
    ) -> bool:
        """Return True when wet full-file lanes disagree enough to justify dry probe."""
        paths = self._top_paths_from_results(brain_votes, physics_votes, limit=8)
        buckets = [self._source_bucket_from_path(path) for path in paths]
        concrete = [bucket for bucket in buckets if bucket not in {"unknown"}]
        if len(set(concrete[:4])) >= 2:
            return True
        if concrete and concrete[0] in {"voice", "strings", "sax_reed", "synth", "mallet_bells", "instrument_loop"}:
            if any(bucket != concrete[0] for bucket in concrete[1:6]):
                return True
        shape = facts.evidence.get("shape_vote", {}) if isinstance(getattr(facts, "evidence", None), dict) else {}
        if isinstance(shape, dict):
            primary_shape = str(shape.get("primary_shape", ""))
            secondary_shape = str(shape.get("secondary_shape", ""))
            if primary_shape == "vocal_phrase" and secondary_shape in {
                "pitched_phrase",
                "sustained_pad",
                "bass_phrase",
            }:
                return True
        return False


def analyze_audio_file(audio_file: Path) -> AudioPhysics:
    """Extract full and direct/body physics for one audio file.

    The direct/body view is tail-reduced analysis evidence. It is computed for
    every sorted file so voters can compare the full recording against the
    source body before reverb, delay, or long decay dominates.
    """
    fingerprint, duration, read_status = make_fingerprint_safe(audio_file)
    direct_fp, direct_duration, direct_status, direct_meta = make_direct_body_fingerprint_safe(audio_file)
    third_party_profile = third_party_feature_profile(audio_file)
    physics = AudioPhysics(
        source_path=audio_file,
        fingerprint=fingerprint,
        duration_sec=float(duration),
        read_status=str(read_status),
        direct_body_fingerprint=direct_fp,
        direct_body_duration_sec=float(direct_duration),
        direct_body_status=str(direct_status),
    )
    object.__setattr__(physics, "direct_body_profile", direct_meta)
    object.__setattr__(physics, "third_party_feature_profile", third_party_profile)
    return physics


def analyze_harmonic_core_with_optional_wet_profile(
    audio_file: Path,
    wet_profile: dict[str, Any] | None,
) -> AudioPhysics:
    """Call harmonic-core analysis while preserving older test monkeypatches.

    Args:
        audio_file: Staged audio file to analyze.
        wet_profile: Optional wetness profile already measured during this run.

    Returns:
        Harmonic-core physics.

    Side Effects:
        Reads audio through the configured harmonic analyzer.

    Raises:
        Re-raises analyzer exceptions other than the compatibility ``TypeError``
        caused by older monkeypatched functions that do not accept the new
        keyword argument.

    Important Constraints:
        The fallback exists for test compatibility. Product code should pass the
        wetness profile through the keyword path so duplicate wetness work is
        avoided.
    """
    try:
        return analyze_harmonic_core_audio_file(audio_file, precomputed_wet_profile=wet_profile)
    except TypeError as error:
        if "precomputed_wet_profile" not in str(error):
            raise
        return analyze_harmonic_core_audio_file(audio_file)


def analyze_harmonic_core_audio_file(
    audio_file: Path,
    *,
    precomputed_wet_profile: dict[str, Any] | None = None,
) -> AudioPhysics:
    """Extract harmonic-core recall fingerprint and attach wetness diagnostics.

    Args:
        audio_file: Staged audio file to analyze.
        precomputed_wet_profile: Optional wetness profile already measured for
            this file during the same run. Passing it avoids one duplicate
            wetness pass while preserving harmonic-core output.

    Returns:
        Harmonic-core ``AudioPhysics`` object with wetness diagnostics attached.

    Side Effects:
        Reads audio from disk. Does not write cache data.

    Raises:
        No intentional exceptions; feature helpers convert failures to status
        strings.

    Important Constraints:
        The precomputed profile is measured evidence only. It must not contain
        final labels or voter decisions.
    """
    fingerprint, duration, read_status, wet_profile = make_harmonic_core_fingerprint_safe(
        audio_file,
        wet_profile=precomputed_wet_profile,
    )
    physics = AudioPhysics(
        source_path=audio_file,
        fingerprint=fingerprint,
        duration_sec=float(duration),
        read_status=str(read_status),
    )
    object.__setattr__(physics, "wetness_profile", wet_profile)
    return physics


def analyze_wetness_profile(audio_file: Path) -> dict[str, Any]:
    """Return wet/smeared diagnostics without computing a harmonic fingerprint."""
    try:
        profile = wetness_profile(audio_file)
        return profile if isinstance(profile, dict) else {"status": "missing", "wetness_score": 0.0}
    except Exception as exc:
        return {"status": "wetness_error:" + str(exc)[:120], "wetness_score": 0.0}


def wetness_score_from_profile(profile: dict[str, Any] | None) -> float:
    """Read wetness score from a diagnostics dict."""
    if not isinstance(profile, dict):
        return 0.0
    try:
        return float(profile.get("wetness_score", 0.0) or 0.0)
    except Exception:
        return 0.0


def relabel_voter_result(result: VoterResult, lane_name: str, prefix_reason: str = "") -> VoterResult:
    """Rename a baby-lane result produced from a transformed fingerprint."""
    guesses: list[CategoryGuess] = []
    for guess in result.guesses:
        evidence = dict(guess.evidence)
        evidence["brain_lane"] = lane_name
        evidence["harmonic_core_recall_lane"] = True
        evidence["harmonic_core_is_recall_not_final_truth"] = True
        guesses.append(
            CategoryGuess(
                label=guess.label,
                folder_path=guess.folder_path,
                top_family=guess.top_family,
                score=guess.score,
                confidence=guess.confidence,
                rank=guess.rank,
                reason=(prefix_reason + "_" + str(guess.reason)).strip("_"),
                evidence=evidence,
            )
        )
    diagnostics = dict(result.diagnostics)
    diagnostics.update({"enabled": True, "lane_name": lane_name, "purpose": "wet_smeared_harmonic_core_recall"})
    return VoterResult(voter_name=f"brain_{lane_name}", guesses=guesses, diagnostics=diagnostics)


def voter_result_digest(result: VoterResult) -> dict[str, Any]:
    """Compact voter result for manifest diagnostics."""
    return {
        "voter_name": result.voter_name,
        "diagnostics": result.diagnostics,
        "top_guesses": [
            {
                "rank": guess.rank,
                "label": guess.label,
                "folder_path": guess.folder_path,
                "top_family": guess.top_family,
                "score": guess.score,
                "confidence": guess.confidence,
                "reason": guess.reason,
                "evidence": guess.evidence,
            }
            for guess in result.guesses[:10]
        ],
    }
