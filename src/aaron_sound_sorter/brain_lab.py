"""Brain Lab diagnostics for voter-lane evidence.

This command is intentionally not a sorter. It runs the voters in isolation and
prints their raw evidence so bad families can be investigated before another
gate, rescue, or combiner rule is allowed to influence placement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aaron_sound_sorter.domain.facts import build_shared_audio_facts
from aaron_sound_sorter.domain.policies import BrainVoterPolicy, PhysicsVoterPolicy, ShapeVoterPolicy
from aaron_sound_sorter.engine.sorter import (
    analyze_audio_file,
    analyze_harmonic_core_audio_file,
    relabel_voter_result,
    voter_result_digest,
)
from aaron_sound_sorter.infrastructure.brain_repository import BrainRepository
from aaron_sound_sorter.role_gate import dynamic_role_gate
from aaron_sound_sorter.voters.brain_recall import (
    BalancedRecallBrainVoter,
    FullBrainVoter,
    combine_full_and_balanced_brain_votes,
    make_disabled_balanced_result,
)
from aaron_sound_sorter.voters.physics_voter import PhysicsVoter
from aaron_sound_sorter.voters.shape_voter import ShapeVoter

RAW_BABY_DEFAULTS = {
    "core_baby": "stage4_folder_brain_core_baby.json",
    "spread_baby": "stage4_folder_brain_spread_baby.json",
    "outlier_baby": "stage4_folder_brain_outlier_baby.json",
}
HARMONIC_DEFAULTS = {
    "harmonic_full": "stage4_folder_brain_harmonic.json",
    "harmonic_core_baby": "stage4_folder_brain_harmonic_core_baby.json",
    "harmonic_spread_baby": "stage4_folder_brain_harmonic_spread_baby.json",
    "harmonic_outlier_baby": "stage4_folder_brain_harmonic_outlier_baby.json",
}


class BrainLabRunner:
    """Run diagnostic voter lanes without consensus or file placement."""

    def __init__(self, brain_repository: BrainRepository | None = None) -> None:
        self.brain_repository = brain_repository or BrainRepository()

    def run(
        self,
        *,
        input_path: Path,
        brain_path: Path,
        raw_baby_paths: dict[str, Path | None],
        harmonic_paths: dict[str, Path | None],
        top_n: int = 10,
    ) -> dict[str, Any]:
        """Return a JSON-friendly diagnostic report for one audio file."""
        top_n = max(1, int(top_n or 10))
        full_brain = self.brain_repository.load(brain_path)
        physics = analyze_audio_file(input_path)
        facts = build_shared_audio_facts(physics)
        harmonic_physics = analyze_harmonic_core_audio_file(input_path)
        facts.evidence["wetness_profile"] = getattr(harmonic_physics, "wetness_profile", {})
        facts.evidence["harmonic_core_status"] = harmonic_physics.read_status
        facts.evidence["harmonic_core_duration_sec"] = harmonic_physics.duration_sec
        role_gate = dynamic_role_gate(physics, facts, full_brain)
        facts.evidence["dynamic_role_gate"] = role_gate.evidence()

        policy_n = max(top_n, 20)
        lanes: dict[str, dict[str, Any]] = {}
        raw_full = FullBrainVoter(BrainVoterPolicy(top_n=policy_n)).vote(physics, facts, full_brain)
        lanes["raw_full_brain"] = voter_result_digest(raw_full)

        raw_baby_results = {}
        for lane, purpose in [
            ("core_baby", "precision_clean_center"),
            ("spread_baby", "balanced_clean_diversity"),
            ("outlier_baby", "edge_case_recall_not_final_truth"),
        ]:
            brain = self._load_optional(raw_baby_paths.get(lane))
            voter = BalancedRecallBrainVoter(BrainVoterPolicy(top_n=policy_n), lane_name=lane, purpose=purpose)
            result = (
                voter.vote(physics, facts, brain)
                if brain is not None
                else make_disabled_balanced_result(f"no {lane} brain file provided or found", lane)
            )
            raw_baby_results[lane] = result
            lanes[f"raw_{lane}"] = voter_result_digest(result)

        enabled_babies = {lane: result for lane, result in raw_baby_results.items() if result.guesses}
        lanes["brain_ensemble"] = voter_result_digest(
            combine_full_and_balanced_brain_votes(
                full_result=raw_full,
                baby_results=enabled_babies,
                max_guesses=max(policy_n, 50),
            )
        )

        lanes["physics"] = voter_result_digest(
            PhysicsVoter(PhysicsVoterPolicy(top_n=policy_n)).vote(physics, facts, full_brain)
        )
        shape_result = ShapeVoter(ShapeVoterPolicy()).vote(physics, facts, full_brain)
        lanes["shape"] = voter_result_digest(shape_result)

        harmonic_ready = bool(harmonic_physics.read_status == "ok")
        harmonic_full = self._load_optional(harmonic_paths.get("harmonic_full"))
        if harmonic_ready and harmonic_full is not None:
            result = FullBrainVoter(BrainVoterPolicy(top_n=policy_n)).vote(harmonic_physics, facts, harmonic_full)
        else:
            reason = "no harmonic-trained full brain file provided or found"
            if not harmonic_ready:
                reason = f"harmonic transform failed: {harmonic_physics.read_status}"
            result = make_disabled_balanced_result(reason, "harmonic_full")
        lanes["harmonic_full"] = voter_result_digest(result)

        for lane, source_lane, purpose in [
            ("harmonic_core_baby", "core_baby", "harmonic_precision_clean_center"),
            ("harmonic_spread_baby", "spread_baby", "harmonic_balanced_clean_diversity"),
            ("harmonic_outlier_baby", "outlier_baby", "harmonic_edge_case_recall_not_final_truth"),
        ]:
            brain = self._load_optional(harmonic_paths.get(lane))
            voter = BalancedRecallBrainVoter(BrainVoterPolicy(top_n=policy_n), lane_name=source_lane, purpose=purpose)
            if harmonic_ready and brain is not None:
                result = relabel_voter_result(
                    voter.vote(harmonic_physics, facts, brain), lane, prefix_reason="harmonic_core"
                )
            else:
                reason = f"no {lane} brain file provided or found"
                if not harmonic_ready:
                    reason = f"harmonic transform failed: {harmonic_physics.read_status}"
                result = make_disabled_balanced_result(reason, lane)
            lanes[lane] = voter_result_digest(result)

        return {
            "input_path": str(Path(input_path).expanduser()),
            "brain_path": str(Path(brain_path).expanduser()),
            "top_n": top_n,
            "audio": {
                "duration_sec": physics.duration_sec,
                "read_status": physics.read_status,
                "is_loop_like": facts.is_loop_like,
                "is_single_event_like": facts.is_single_event_like,
                "is_short_hit_like": facts.is_short_hit_like,
                "is_long": facts.is_long,
                "feature_count": facts.feature_count,
            },
            "wetness_profile": facts.evidence.get("wetness_profile", {}),
            "dynamic_role_gate": facts.evidence.get("dynamic_role_gate", {}),
            "shape_vote": shape_result.diagnostics.get("shape_vote", {}),
            "lanes": lanes,
        }

    def _load_optional(self, path: Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        resolved = Path(path).expanduser()
        if not resolved.exists():
            return None
        return self.brain_repository.load(resolved)


def run_brain_lab_from_args(args) -> int:
    """CLI entry point for the diagnostic lab command."""
    brain_path = Path(args.brain).expanduser()
    raw_paths = resolve_raw_baby_paths(brain_path, args)
    harmonic_paths = resolve_harmonic_paths(brain_path, args)
    report = BrainLabRunner().run(
        input_path=Path(args.input_path).expanduser(),
        brain_path=brain_path,
        raw_baby_paths=raw_paths,
        harmonic_paths=harmonic_paths,
        top_n=int(getattr(args, "top_n", 10)),
    )
    json_out = str(getattr(args, "json_out", "") or "").strip()
    if json_out:
        Path(json_out).expanduser().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(format_brain_lab_report(report))
    return 0


def resolve_raw_baby_paths(brain_path: Path, args) -> dict[str, Path | None]:
    """Resolve raw baby brain paths without touching harmonic defaults."""
    base = Path(brain_path).expanduser()
    legacy = optional_path(getattr(args, "baby_brain", "")) or base.with_name("stage4_folder_brain_baby.json")
    default_spread = base.with_name(RAW_BABY_DEFAULTS["spread_baby"])
    return {
        "core_baby": optional_path(getattr(args, "core_baby_brain", ""))
        or base.with_name(RAW_BABY_DEFAULTS["core_baby"]),
        "spread_baby": optional_path(getattr(args, "spread_baby_brain", ""))
        or (legacy if legacy.exists() else default_spread),
        "outlier_baby": optional_path(getattr(args, "outlier_baby_brain", ""))
        or base.with_name(RAW_BABY_DEFAULTS["outlier_baby"]),
    }


def resolve_harmonic_paths(brain_path: Path, args) -> dict[str, Path | None]:
    """Resolve harmonic-domain brain paths; raw brains are never substituted."""
    base = Path(brain_path).expanduser()
    return {
        "harmonic_full": optional_path(getattr(args, "harmonic_brain", ""))
        or base.with_name(HARMONIC_DEFAULTS["harmonic_full"]),
        "harmonic_core_baby": optional_path(getattr(args, "harmonic_core_baby_brain", ""))
        or base.with_name(HARMONIC_DEFAULTS["harmonic_core_baby"]),
        "harmonic_spread_baby": optional_path(getattr(args, "harmonic_spread_baby_brain", ""))
        or base.with_name(HARMONIC_DEFAULTS["harmonic_spread_baby"]),
        "harmonic_outlier_baby": optional_path(getattr(args, "harmonic_outlier_baby_brain", ""))
        or base.with_name(HARMONIC_DEFAULTS["harmonic_outlier_baby"]),
    }


def optional_path(value: str | None) -> Path | None:
    """Convert an optional CLI path."""
    text = str(value or "").strip()
    return Path(text).expanduser() if text else None


def format_brain_lab_report(report: dict[str, Any]) -> str:
    """Format the diagnostic report for humans."""
    lines = [
        f"Brain Lab: {report.get('input_path', '')}",
        f"Read: {report.get('audio', {}).get('read_status', '')}; duration={float(report.get('audio', {}).get('duration_sec', 0.0)):.3f}s; features={report.get('audio', {}).get('feature_count', '')}",
    ]
    wetness = report.get("wetness_profile", {})
    if isinstance(wetness, dict):
        lines.append(f"Wetness: {float(wetness.get('wetness_score', 0.0) or 0.0):.3f}")
    role_gate = report.get("dynamic_role_gate", {})
    if isinstance(role_gate, dict):
        lines.append(
            f"Role gate: selected={role_gate.get('selected_top_families', '')}; enabled={role_gate.get('enabled', '')}; reason={role_gate.get('reason', '')}"
        )
    shape_vote = report.get("shape_vote", {})
    if isinstance(shape_vote, dict):
        lines.append(f"Shape: {shape_vote.get('primary_shape', '')}; confidence={shape_vote.get('confidence', '')}")
    lines.append("")

    lane_order = [
        "raw_full_brain",
        "raw_core_baby",
        "raw_spread_baby",
        "raw_outlier_baby",
        "brain_ensemble",
        "harmonic_full",
        "harmonic_core_baby",
        "harmonic_spread_baby",
        "harmonic_outlier_baby",
        "physics",
        "shape",
    ]
    top_n = int(report.get("top_n", 10) or 10)
    lanes = report.get("lanes", {})
    for lane in lane_order:
        digest = lanes.get(lane, {}) if isinstance(lanes, dict) else {}
        diagnostics = digest.get("diagnostics", {}) if isinstance(digest, dict) else {}
        lines.append(f"{lane}:")
        if diagnostics.get("enabled") is False:
            lines.append(f"  disabled: {diagnostics.get('reason', '')}")
            lines.append("")
            continue
        guesses = digest.get("top_guesses", []) if isinstance(digest, dict) else []
        if not guesses:
            lines.append("  no guesses")
            lines.append("")
            continue
        for guess in guesses[:top_n]:
            label = guess.get("label", "")
            score = float(guess.get("score", 0.0) or 0.0)
            confidence = float(guess.get("confidence", 0.0) or 0.0)
            rank = int(guess.get("rank", 0) or 0)
            lines.append(f"  {rank:>2}. {label}  score={score:.4f} confidence={confidence:.3f}")
        lines.append("")
    return "\n".join(lines).rstrip()
