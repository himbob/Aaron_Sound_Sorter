#!/usr/bin/env python3
"""Build a read-only cluster-first audio review pack with every member."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.domain.facts import build_shared_audio_facts  # noqa: E402
from aaron_sound_sorter.engine.sorter import analyze_audio_file  # noqa: E402
from aaron_sound_sorter.neural_audio.active_learning_queue import (  # noqa: E402
    ActiveLearningCluster,
    ActiveLearningMember,
    create_active_learning_clusters,
    measured_structure_bucket,
    semantic_partition_family,
)
from aaron_sound_sorter.neural_audio.dataset import AUDIO_SUFFIXES  # noqa: E402
from aaron_sound_sorter.neural_audio.gui_training import configured_clap_trainer  # noqa: E402
from aaron_sound_sorter.neural_audio.hashing import sha256_file  # noqa: E402
from aaron_sound_sorter.neural_audio.prototype_index import PrototypeIndex  # noqa: E402
from aaron_sound_sorter.neural_audio.semantic_panel import (  # noqa: E402
    flattened_semantic_prompts,
    predict_semantic_family,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sample-library-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.84)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-clusters", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260725)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Collect, partition, cluster, and copy a source-name-blind review pack."""
    args = build_parser().parse_args(argv)
    if args.max_files < 0 or args.max_clusters < 0:
        raise ValueError("maximum counts cannot be negative")
    root = args.project_root.expanduser().resolve()
    library_root = args.sample_library_root.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    trainer = configured_clap_trainer(root)
    index_path = active_index_path(root, trainer.pointer_path)
    index = PrototypeIndex.load(index_path)
    members = collect_members(
        library_root,
        trainer=trainer,
        index=index,
        max_files=args.max_files,
        random_seed=args.random_seed,
    )
    clusters = create_active_learning_clusters(
        members,
        model_id=trainer.provider.model_id,
        training_run_id=index_path.name,
        similarity_threshold=args.similarity_threshold,
    )
    if args.max_clusters:
        clusters = tuple(
            sorted(clusters, key=lambda cluster: (-cluster.review_priority, cluster.cluster_id))[: args.max_clusters]
        )
    manifest_rows = copy_review_pack(destination, clusters)
    output_payload = {
        "schema_version": 1,
        "status": "built",
        "destination": str(destination),
        "model_id": trainer.provider.model_id,
        "training_run_id": index_path.name,
        "cluster_count": len(clusters),
        "member_count": sum(len(cluster.member_hashes) for cluster in clusters),
        "source_name_policy": "decoded audio and content hashes only; source names forbidden as evidence",
        "production_training_enabled": False,
        "clusters": manifest_rows,
    }
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Clusters: {len(clusters)}")
    print(f"Members retained: {sum(len(cluster.member_hashes) for cluster in clusters)}")
    print(f"Review pack: {destination}")
    print(f"Build report: {output_path}")
    return 0


def collect_members(
    library_root: Path,
    *,
    trainer: Any,
    index: PrototypeIndex,
    max_files: int = 0,
    random_seed: int = 20260725,
) -> list[ActiveLearningMember]:
    """Collect deduplicated measured members without inspecting source names."""
    active_hashes = set(index.metadata.training_hashes)
    semantic_prompts, semantic_families = flattened_semantic_prompts()
    semantic_text_embeddings = trainer.provider.embed_texts(semantic_prompts)
    members: list[ActiveLearningMember] = []
    seen_hashes: set[str] = set()
    audio_paths = [
        audio_path
        for audio_path in library_root.rglob("*")
        if audio_path.is_file() and audio_path.suffix.casefold() in AUDIO_SUFFIXES
    ]
    random.Random(random_seed).shuffle(audio_paths)
    for audio_path in audio_paths:
        try:
            file_sha256 = sha256_file(audio_path)
            if file_sha256 in active_hashes or file_sha256 in seen_hashes:
                continue
            physics = analyze_audio_file(audio_path)
            facts = build_shared_audio_facts(physics)
            if facts.is_broken_or_tiny:
                continue
            record = trainer.cache.get_or_compute(audio_path, trainer.provider)
            prediction = index.predict(record)
            semantic = predict_semantic_family(record, semantic_text_embeddings, semantic_families)
        except (OSError, RuntimeError, ValueError):
            continue
        seen_hashes.add(file_sha256)
        members.append(
            ActiveLearningMember(
                audio_path=audio_path,
                file_sha256=file_sha256,
                vector=record.vector,
                structure_bucket=measured_structure_bucket(facts),
                broad_family=semantic_partition_family(semantic.predicted_family),
                predicted_label=prediction.predicted_label,
                second_label=prediction.second_label,
                top_similarity=prediction.top_similarity,
                margin=prediction.margin,
                known_distribution=prediction.known_distribution,
            )
        )
        if max_files and len(members) >= max_files:
            break
    return members


def copy_review_pack(destination: Path, clusters: tuple[ActiveLearningCluster, ...]) -> list[dict[str, Any]]:
    """Copy all cluster members and representatives under hash-only names."""
    if destination.exists():
        raise FileExistsError(f"review destination already exists: {destination}")
    destination.mkdir(parents=True)
    manifest_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, str]] = []
    for cluster in clusters:
        cluster_dir = destination / cluster.cluster_id
        members_dir = cluster_dir / "members"
        representatives_dir = cluster_dir / "representatives"
        members_dir.mkdir(parents=True)
        representatives_dir.mkdir()
        member_path_by_hash: dict[str, str] = {}
        for member in cluster.members:
            suffix = member.audio_path.suffix.casefold()
            copied_name = f"audio_{member.file_sha256}{suffix}"
            copied_path = members_dir / copied_name
            shutil.copy2(member.audio_path, copied_path)
            relative_path = str(copied_path.relative_to(destination))
            member_path_by_hash[member.file_sha256] = relative_path
            csv_rows.append(member_csv_row(cluster, member, relative_path))
        copy_representative(
            cluster,
            member_path_by_hash,
            destination,
            representatives_dir,
            "center",
            cluster.center_hash,
        )
        for position, file_sha256 in enumerate(cluster.typical_hashes, start=1):
            copy_representative(
                cluster,
                member_path_by_hash,
                destination,
                representatives_dir,
                f"typical_{position:02d}",
                file_sha256,
            )
        copy_representative(
            cluster,
            member_path_by_hash,
            destination,
            representatives_dir,
            "boundary",
            cluster.boundary_representative_hash,
        )
        copy_representative(
            cluster,
            member_path_by_hash,
            destination,
            representatives_dir,
            "outlier",
            cluster.outlier_representative_hash,
        )
        manifest_rows.append(cluster_payload(cluster, member_path_by_hash))
    write_member_csv(destination / "cluster_manifest.csv", csv_rows)
    (destination / "cluster_manifest.json").write_text(
        json.dumps({"schema_version": 1, "clusters": manifest_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "README.txt").write_text(review_readme(), encoding="utf-8")
    return manifest_rows


def copy_representative(
    cluster: ActiveLearningCluster,
    member_path_by_hash: dict[str, str],
    destination: Path,
    representatives_dir: Path,
    role: str,
    file_sha256: str,
) -> None:
    """Copy one representative from the already hash-named member file."""
    if not file_sha256:
        return
    member_relative = member_path_by_hash[file_sha256]
    member_path = destination / member_relative
    target = representatives_dir / f"{role}_{file_sha256}{member_path.suffix}"
    shutil.copy2(member_path, target)


def cluster_payload(cluster: ActiveLearningCluster, member_path_by_hash: dict[str, str]) -> dict[str, Any]:
    """Return the required source-name-blind cluster manifest record."""
    return {
        "cluster_id": cluster.cluster_id,
        "model_id": cluster.model_id,
        "model_version": cluster.model_id,
        "member_count": len(cluster.member_hashes),
        "safe_core_count": len(cluster.safe_core_hashes),
        "boundary_count": len(cluster.boundary_hashes),
        "outlier_count": len(cluster.outlier_hashes),
        "structure_bucket": cluster.structure_bucket,
        "broad_measured_family": cluster.broad_family,
        "center_hash": cluster.center_hash,
        "typical_hashes": cluster.typical_hashes,
        "boundary_hash": cluster.boundary_representative_hash,
        "outlier_hash": cluster.outlier_representative_hash,
        "member_hashes": cluster.member_hashes,
        "safe_core_hashes": cluster.safe_core_hashes,
        "boundary_hashes": cluster.boundary_hashes,
        "outlier_hashes": cluster.outlier_hashes,
        "member_paths_by_hash": member_path_by_hash,
        "mean_similarity": cluster.mean_similarity,
        "radius_p95": cluster.radius_p95,
        "novelty_score": cluster.novelty_score,
        "conflict_score": cluster.conflict_score,
        "suggested_parent_category": cluster.suggested_parent_category,
        "suggested_detailed_categories": cluster.suggested_detailed_categories,
        "review_priority": cluster.review_priority,
        "source_pack_id": cluster.source_pack_id,
        "training_run_id": cluster.training_run_id,
    }


def member_csv_row(
    cluster: ActiveLearningCluster,
    member: ActiveLearningMember,
    relative_path: str,
) -> dict[str, str]:
    """Return one retained-member CSV row without a source locator."""
    if member.file_sha256 in cluster.safe_core_hashes:
        membership = "safe_core"
    elif member.file_sha256 in cluster.outlier_hashes:
        membership = "outlier"
    else:
        membership = "boundary"
    return {
        "cluster_id": cluster.cluster_id,
        "file_sha256": member.file_sha256,
        "review_file": relative_path,
        "membership": membership,
        "structure_bucket": member.structure_bucket,
        "broad_measured_family": member.broad_family,
        "suggested_category": member.predicted_label,
        "second_suggestion": member.second_label,
        "top_similarity": f"{member.top_similarity:.8f}",
        "margin": f"{member.margin:.8f}",
        "known_distribution": "1" if member.known_distribution else "0",
    }


def write_member_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the retained-member review manifest."""
    fieldnames = [
        "cluster_id",
        "file_sha256",
        "review_file",
        "membership",
        "structure_bucket",
        "broad_measured_family",
        "suggested_category",
        "second_suggestion",
        "top_similarity",
        "margin",
        "known_distribution",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def review_readme() -> str:
    """Return concise human instructions for the read-only cluster pack."""
    return (
        "Aaron cluster review pack\n\n"
        "- Listen to center, typical, boundary, and outlier representatives first.\n"
        "- Every member is retained under its cluster's members folder.\n"
        "- Suggestions are neural opinions, not training truth.\n"
        "- Approve the safe core only when the representatives genuinely belong together.\n"
        "- Review boundaries individually.\n"
        "- This pack does not modify or train any brain.\n"
        "- Hash-only filenames prevent source names from becoming evidence.\n"
    )


def active_index_path(project_root: Path, pointer_path: Path) -> Path:
    """Resolve the active prototype index pointer."""
    raw_path = Path(pointer_path.read_text(encoding="utf-8").strip()).expanduser()
    return raw_path.resolve() if raw_path.is_absolute() else (project_root / raw_path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
