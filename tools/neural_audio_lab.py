#!/usr/bin/env python3
"""Shadow-only neural audio lab.

This tool builds and evaluates neural prototype indexes without changing the
legacy sorter's production routing.  It intentionally separates preview
training from held-out evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.cache import EmbeddingCache
from aaron_sound_sorter.neural_audio.contracts import EmbeddingRecord, LabeledAudioExample
from aaron_sound_sorter.neural_audio.dataset import (
    AUDIO_SUFFIXES,
    build_evaluation_split,
    build_explicit_evaluation_split,
    discover_curated_examples,
    write_split_manifest,
)
from aaron_sound_sorter.neural_audio.evaluation import evaluate_heldout
from aaron_sound_sorter.neural_audio.legacy_brain_audit import (
    audit_legacy_brain_trainers,
    write_legacy_brain_audit,
)
from aaron_sound_sorter.neural_audio.legacy_manifest import read_legacy_folder_map_by_hash
from aaron_sound_sorter.neural_audio.prototype_index import PrototypeIndex, PrototypeIndexBuilder
from aaron_sound_sorter.neural_audio.providers import (
    DEFAULT_CLAP_MODEL,
    HuggingFaceAudioEncoderProvider,
    HuggingFaceClapProvider,
)
from aaron_sound_sorter.neural_audio.shadow import build_shadow_rows, write_shadow_csv
from aaron_sound_sorter.neural_audio.trainer_audit import (
    audit_labeled_embeddings,
    trainer_audit_status_counts,
    write_trainer_audit_csv,
)


def provider_from_args(args: argparse.Namespace):
    if args.provider == "clap":
        return HuggingFaceClapProvider(
            model_name_or_path=args.model,
            source_model_id=args.model_source_id or DEFAULT_CLAP_MODEL,
            model_revision=args.model_revision,
            device=args.device,
            allow_network=args.allow_network,
        )
    if args.provider in {"mert", "beats"}:
        return HuggingFaceAudioEncoderProvider(
            provider_name=f"hf_{args.provider}",
            model_name_or_path=args.model,
            source_model_id=args.model_source_id or args.model,
            model_revision=args.model_revision,
            layer=args.layer,
            device=args.device,
            allow_network=args.allow_network,
            allow_remote_code=args.allow_remote_code,
        )
    raise SystemExit(f"unsupported provider: {args.provider}")


def embed_examples(examples, provider, cache: EmbeddingCache):
    by_label = defaultdict(list)
    total = len(examples)
    for index, example in enumerate(examples, start=1):
        print(f"[{index}/{total}] embedding {example.path.name}", flush=True)
        by_label[example.label].append(cache.get_or_compute(example.path, provider))
    return dict(by_label)


def safe_extract_audio_zip(zip_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = [part for part in info.filename.replace("\\", "/").split("/") if part]
            if not parts or any(part in {"..", "."} or part.startswith(".") for part in parts):
                continue
            if Path(parts[-1]).suffix.lower() not in AUDIO_SUFFIXES:
                continue
            target = output_dir.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    return output_dir


def audio_paths(input_path: Path, temp_root: Path) -> list[Path]:
    input_path = input_path.resolve()
    if input_path.is_dir():
        root = input_path
    elif input_path.suffix.lower() == ".zip":
        root = safe_extract_audio_zip(input_path, temp_root / "input")
    else:
        raise SystemExit(f"expected audio folder or ZIP: {input_path}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)


def command_build(args: argparse.Namespace) -> int:
    provider = provider_from_args(args)
    cache = EmbeddingCache(args.cache)
    discovered = discover_curated_examples(args.training_root)
    if args.heldout_root is not None:
        heldout_discovered = discover_curated_examples(args.heldout_root)
        split = build_explicit_evaluation_split(discovered, heldout_discovered)
    else:
        split = build_evaluation_split(discovered, holdout_fraction=args.holdout_fraction, seed=args.seed)
    write_split_manifest(split, args.output / "splits")

    preview_embeddings = embed_examples(split.review_preview, provider, cache)
    index = PrototypeIndexBuilder(
        max_prototypes_per_label=args.max_prototypes,
        keep_fraction=args.keep_fraction,
    ).build(preview_embeddings)
    index.save(args.output / "index")

    heldout_embeddings = embed_examples(split.heldout_eval, provider, cache)
    evaluations = evaluate_heldout(index, heldout_embeddings)
    write_heldout_predictions(
        index,
        split.heldout_eval,
        heldout_embeddings,
        args.output / "heldout_predictions.csv",
    )
    report = args.output / "heldout_evaluation.csv"
    with report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "provider_id",
                "label",
                "heldout_count",
                "correct_count",
                "top1_accuracy",
                "mean_correct_similarity",
                "mean_margin",
                "known_distribution_rate",
            ]
        )
        for row in evaluations:
            writer.writerow(
                [
                    row.provider_id,
                    row.label,
                    row.heldout_count,
                    row.correct_count,
                    f"{row.top1_accuracy:.8f}",
                    f"{row.mean_correct_similarity:.8f}",
                    f"{row.mean_margin:.8f}",
                    f"{row.known_distribution_rate:.8f}",
                ]
            )
    print(f"Index: {args.output / 'index'}")
    print(f"Held-out report: {report}")
    return 0


def write_heldout_predictions(
    index: PrototypeIndex,
    examples: Sequence[LabeledAudioExample],
    embedded_by_label: Mapping[str, Sequence[EmbeddingRecord]],
    output_path: Path,
) -> None:
    """Write per-file held-out evidence without turning it into routing."""
    examples_by_hash = {example.file_sha256: example for example in examples}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file_sha256",
                "display_name",
                "expected_label",
                "predicted_label",
                "second_label",
                "top_similarity",
                "second_similarity",
                "margin",
                "known_distribution",
                "prototype_id",
                "prototype_support_count",
                "prototype_radius_p95",
                "category_spread_p95",
                "distance_to_prototype",
                "radius_ratio",
                "provider_id",
                "model_id",
                "human_verdict",
            ]
        )
        rows = []
        for expected_label, records in embedded_by_label.items():
            for record in records:
                prediction = index.predict(record)
                example = examples_by_hash[record.file_sha256]
                rows.append((expected_label, example, prediction))
        for expected_label, example, prediction in sorted(
            rows,
            key=lambda row: (
                row[0] == row[2].predicted_label,
                row[0],
                row[1].path.name,
            ),
        ):
            writer.writerow(
                [
                    example.file_sha256,
                    example.path.name,
                    expected_label,
                    prediction.predicted_label,
                    prediction.second_label,
                    f"{prediction.top_similarity:.8f}",
                    f"{prediction.second_similarity:.8f}",
                    f"{prediction.margin:.8f}",
                    int(prediction.known_distribution),
                    prediction.prototype_id,
                    int(prediction.evidence.get("prototype_support_count", 0)),
                    f"{prediction.prototype_radius_p95:.8f}",
                    f"{float(prediction.evidence.get('category_spread_p95', 0.0)):.8f}",
                    f"{prediction.distance_to_prototype:.8f}",
                    f"{prediction.radius_ratio:.8f}",
                    prediction.provider_id,
                    prediction.model_id,
                    "",
                ]
            )


def command_shadow(args: argparse.Namespace) -> int:
    provider = provider_from_args(args)
    cache = EmbeddingCache(args.cache)
    index = PrototypeIndex.load(args.index)
    legacy = read_legacy_folder_map_by_hash(args.legacy_manifest) if args.legacy_manifest else {}
    with tempfile.TemporaryDirectory(prefix="aaron_neural_shadow_") as temp:
        paths = audio_paths(args.input, Path(temp))
        records = []
        for position, path in enumerate(paths, start=1):
            print(f"[{position}/{len(paths)}] embedding {path.name}", flush=True)
            records.append((path, cache.get_or_compute(path, provider)))
        rows = build_shadow_rows(index, records, legacy_folders_by_hash=legacy)
        write_shadow_csv(rows, args.output)
    disagreements = sum(1 for row in rows if row.legacy_folder and not row.agreement)
    unknown = sum(1 for row in rows if not row.neural_known_distribution)
    print(f"Files: {len(rows)}")
    print(f"Legacy/neural disagreements: {disagreements}")
    print(f"Neural out-of-distribution: {unknown}")
    print(f"Report: {args.output}")
    return 0


def command_audit_trainers(args: argparse.Namespace) -> int:
    provider = provider_from_args(args)
    cache = EmbeddingCache(args.cache)
    discovered = discover_curated_examples(args.training_root, allow_label_conflicts=True)
    embedded = defaultdict(list)
    display_names = {}
    total = sum(len(rows) for rows in discovered.values())
    position = 0
    for label, rows in sorted(discovered.items()):
        for path, digest, _ in sorted(rows, key=lambda item: item[1]):
            position += 1
            print(f"[{position}/{total}] auditing trainer {path.name}", flush=True)
            embedded[label].append(cache.get_or_compute(path, provider))
            display_names[digest] = path.name
    audit_rows = audit_labeled_embeddings(dict(embedded))
    write_trainer_audit_csv(audit_rows, args.output, display_names_by_hash=display_names)
    counts = trainer_audit_status_counts(audit_rows)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider_id": provider.provider_id,
                "model_id": provider.model_id,
                "trainer_count": len(audit_rows),
                "status_counts": counts,
                "warning": "This is leave-one-out contamination evidence, not held-out accuracy.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Trainer audit: {args.output}")
    print(f"Summary: {summary_path}")
    print(f"Status counts: {counts}")
    if args.focus_output:
        if not args.focus_label_prefix:
            raise ValueError("--focus-output requires at least one --focus-label-prefix")
        focus_rows = tuple(
            row for row in audit_rows if any(row.label.startswith(prefix) for prefix in args.focus_label_prefix)
        )
        write_trainer_audit_csv(focus_rows, args.focus_output, display_names_by_hash=display_names)
        focus_summary_path = args.focus_output.with_suffix(".summary.json")
        focus_summary_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "label_prefixes": args.focus_label_prefix,
                    "trainer_count": len(focus_rows),
                    "status_counts": trainer_audit_status_counts(focus_rows),
                    "warning": "Filtered leave-one-out contamination evidence; not held-out accuracy.",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Focused trainer audit: {args.focus_output}")
        print(f"Focused summary: {focus_summary_path}")
    return 0


def command_audit_legacy_brains(args: argparse.Namespace) -> int:
    brain_paths = args.brain or sorted(PROJECT_ROOT.glob("stage4*_brain*.json"))
    rows = audit_legacy_brain_trainers(brain_paths)
    write_legacy_brain_audit(rows, args.output, args.summary)
    status_counts = Counter(row.status for row in rows)
    print(f"Brain files: {len({row.brain_path for row in rows})}")
    print(f"Trainer occurrences: {len(rows)}")
    print(f"Status counts: {dict(sorted(status_counts.items()))}")
    print(f"Audit: {args.output}")
    print(f"Summary: {args.summary}")
    return 0


def add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["clap", "mert", "beats"], default="clap")
    parser.add_argument("--model", required=True, help="Pinned local model path or reviewed Hugging Face model id")
    parser.add_argument("--model-source-id", default="", help="Stable upstream model ID for a local snapshot")
    parser.add_argument("--model-revision", default="", help="Pinned upstream commit hash")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-network", action="store_true", help="Allow model download; disabled by default")
    parser.add_argument(
        "--allow-remote-code", action="store_true", help="Required by some experimental models such as MERT"
    )
    parser.add_argument("--layer", type=int, default=-1, help="Hidden-state layer for generic audio encoders")
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "_reports" / "neural_audio" / "embedding_cache",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build preview-only prototypes and evaluate held-out examples")
    build.add_argument("--training-root", type=Path, required=True)
    build.add_argument(
        "--heldout-root",
        type=Path,
        help="Optional deliberately separate held-out root; disables automatic splitting",
    )
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--holdout-fraction", type=float, default=0.20)
    build.add_argument("--seed", default="aaron-neural-v1")
    build.add_argument("--max-prototypes", type=int, default=6)
    build.add_argument("--keep-fraction", type=float, default=0.95)
    add_provider_arguments(build)
    build.set_defaults(handler=command_build)

    shadow = subparsers.add_parser("shadow", help="compare a neural index with optional legacy manifest results")
    shadow.add_argument("--index", type=Path, required=True)
    shadow.add_argument("--input", type=Path, required=True)
    shadow.add_argument("--legacy-manifest", type=Path)
    shadow.add_argument("--output", type=Path, required=True)
    add_provider_arguments(shadow)
    shadow.set_defaults(handler=command_shadow)

    audit = subparsers.add_parser(
        "audit-trainers",
        help="find unsupported, outlier, and cross-label human trainers",
    )
    audit.add_argument("--training-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--focus-label-prefix", action="append", default=[])
    audit.add_argument("--focus-output", type=Path)
    add_provider_arguments(audit)
    audit.set_defaults(handler=command_audit_trainers)

    legacy_audit = subparsers.add_parser(
        "audit-legacy-brains",
        help="inventory trainer conflicts across legacy folder and dedicated memory brains",
    )
    legacy_audit.add_argument("--brain", type=Path, action="append", help="brain JSON; repeat to select several")
    legacy_audit.add_argument("--output", type=Path, required=True)
    legacy_audit.add_argument("--summary", type=Path, required=True)
    legacy_audit.set_defaults(handler=command_audit_legacy_brains)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
