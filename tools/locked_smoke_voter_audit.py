"""Audit locked-smoke runs for voter disagreement behind passing results.

This tool is diagnostic only. It reads the locked smoke acceptance manifest and
reports cases where the final folder passed but early voter lanes disagreed,
late guards carried the decision, or trainable memory did not fire. It never
sorts audio, trains brains, or changes classifier behavior.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoterOddity:
    """One accepted case whose voter lanes deserve inspection.

    Args:
        case_id: Stable acceptance case id.
        filename: Stable acceptance sample filename used only as test id.
        final_folder: Final sorter folder path.
        brain_vote: Brain lane rank-one folder.
        physics_vote: Physics lane rank-one folder.
        shape_vote: ShapeVoter primary shape.
        consensus_status: Final consensus status.
        flags: Human-readable audit flags.
        learned_voter_matched: Whether voter-memory matched the sample.
        learned_physics_matched: Whether physics-memory matched the sample.
    """

    case_id: str
    filename: str
    final_folder: str
    brain_vote: str
    physics_vote: str
    shape_vote: str
    consensus_status: str
    flags: tuple[str, ...]
    learned_voter_matched: str
    learned_physics_matched: str


def main() -> int:
    """Run the voter oddity audit and write CSV/text reports."""
    parser = argparse.ArgumentParser(description="Audit latest locked-smoke acceptance voter disagreements.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else latest_acceptance_run(project_root)
    manifest_path = run_dir / "full_manifest_combined.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing locked smoke combined manifest: {manifest_path}")

    rows = read_manifest_rows(manifest_path)
    oddities = [oddity for oddity in (audit_row(row) for row in rows) if oddity is not None]
    write_oddity_csv(run_dir / "trusted_voter_oddities.csv", oddities)
    write_summary(run_dir / "trusted_voter_oddities_summary.txt", run_dir, rows, oddities)
    print(f"Run: {run_dir}")
    print(f"Cases: {len(rows)}")
    print(f"Oddities: {len(oddities)}")
    print(f"CSV: {run_dir / 'trusted_voter_oddities.csv'}")
    print(f"Summary: {run_dir / 'trusted_voter_oddities_summary.txt'}")
    return 0


def latest_acceptance_run(project_root: Path) -> Path:
    """Return the newest locked-smoke run directory."""
    root = project_root / "_reports" / "locked_smoke_acceptance"
    candidates = sorted(path for path in root.glob("run_*") if path.is_dir())
    if not candidates:
        raise SystemExit(f"No locked smoke acceptance runs found under: {root}")
    return candidates[-1]


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Read manifest rows from ``manifest_path``."""
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def audit_row(row: dict[str, str]) -> VoterOddity | None:
    """Return an oddity row when early voters disagree with a passing final."""
    final_folder = row.get("folder_path", "")
    brain_vote = row.get("brain_vote_1", "")
    physics_vote = row.get("physics_vote_1", "")
    consensus_status = row.get("consensus_status", "")
    flags: list[str] = []
    if top_family(brain_vote) and top_family(brain_vote) != top_family(final_folder):
        flags.append("brain_top_mismatch")
    if top_family(physics_vote) and top_family(physics_vote) != top_family(final_folder):
        flags.append("physics_top_mismatch")
    if row.get("final_agrees_with_brain_ensemble") == "False":
        flags.append("brain_ensemble_disagreed")
    if row.get("post_mutator_count") not in {"", "0", "0.0"}:
        flags.append("post_mutator_fired")
    if consensus_status.startswith("final_"):
        flags.append("static_final_guard")
    if "learned_owner" in consensus_status:
        flags.append("learned_owner_claim")
    if not flags:
        return None
    return VoterOddity(
        case_id=row.get("case_id", ""),
        filename=row.get("case_filename", ""),
        final_folder=final_folder,
        brain_vote=brain_vote,
        physics_vote=physics_vote,
        shape_vote=row.get("shape_vote", ""),
        consensus_status=consensus_status,
        flags=tuple(flags),
        learned_voter_matched=row.get("learned_voter_memory_matched", ""),
        learned_physics_matched=row.get("learned_physics_memory_matched", ""),
    )


def top_family(folder_path: str) -> str:
    """Return the top folder family for a folder path."""
    return str(folder_path or "").split("/", 1)[0]


def write_oddity_csv(path: Path, oddities: list[VoterOddity]) -> None:
    """Write oddity rows as CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "filename",
                "flags",
                "final_folder",
                "brain_vote_1",
                "physics_vote_1",
                "shape_vote",
                "consensus_status",
                "learned_voter_memory_matched",
                "learned_physics_memory_matched",
            ]
        )
        for oddity in oddities:
            writer.writerow(
                [
                    oddity.case_id,
                    oddity.filename,
                    ";".join(oddity.flags),
                    oddity.final_folder,
                    oddity.brain_vote,
                    oddity.physics_vote,
                    oddity.shape_vote,
                    oddity.consensus_status,
                    oddity.learned_voter_matched,
                    oddity.learned_physics_matched,
                ]
            )


def write_summary(
    path: Path,
    run_dir: Path,
    rows: list[dict[str, str]],
    oddities: list[VoterOddity],
) -> None:
    """Write a compact text summary for humans."""
    memory_count = sum(
        1
        for row in rows
        if row.get("learned_voter_memory_matched") == "True" or row.get("learned_physics_memory_matched") == "True"
    )
    flag_counts: dict[str, int] = {}
    for oddity in oddities:
        for flag in oddity.flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    lines = [
        "Locked Smoke Voter Oddity Audit",
        f"Run: {run_dir}",
        f"Cases: {len(rows)}",
        f"Cases with learned memory match: {memory_count}",
        f"Oddities: {len(oddities)}",
        "",
        "Flag counts:",
    ]
    for flag, count in sorted(flag_counts.items()):
        lines.append(f"  {flag}: {count}")
    lines.extend(["", "Top oddities:"])
    for oddity in oddities[:20]:
        lines.append(f"  {oddity.case_id}: {','.join(oddity.flags)} -> {oddity.final_folder}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
