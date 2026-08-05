#!/usr/bin/env python3
"""Predict a batch with the active source-name-blind neural prototype index."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.contracts import EmbeddingRecord, LabelPrediction  # noqa: E402
from aaron_sound_sorter.neural_audio.gui_training import (  # noqa: E402
    DEFAULT_TRAINING_CONFIG,
    configured_clap_trainer,
)
from aaron_sound_sorter.neural_audio.ownership import assess_prototype_ownership  # noqa: E402
from aaron_sound_sorter.neural_audio.panns_mapping import (  # noqa: E402
    PannsEventScore,
    PannsMappedEvidence,
    PannsMappingRegistry,
)
from aaron_sound_sorter.neural_audio.panns_provider import (  # noqa: E402
    PannsProvider,
    configured_panns_provider,
)
from aaron_sound_sorter.neural_audio.prompt_brain import ClapPromptIndex  # noqa: E402
from aaron_sound_sorter.neural_audio.prototype_index import PrototypeIndex  # noqa: E402
from aaron_sound_sorter.neural_audio.semantic_panel import (  # noqa: E402
    flattened_semantic_prompts,
    predict_semantic_family,
)
from aaron_sound_sorter.taxonomy_registry import TaxonomyRegistry  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run neural inference and checkpoint every completed input row."""
    args = build_parser().parse_args(argv)
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _run_prediction_batch(args, output_path)
    except Exception as exc:
        existing = _read_prediction_payload(output_path)
        row_errors = dict(existing.get("row_errors", {}))
        row_errors["__runtime__"] = f"{type(exc).__name__}: {exc}"
        _write_prediction_payload(
            output_path,
            status="error",
            message=f"{type(exc).__name__}: {exc}",
            index_path=str(existing.get("index_path", "")),
            predictions=list(existing.get("predictions", [])),
            row_errors=row_errors,
        )
        print(f"Neural runtime failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


def _run_prediction_batch(args: argparse.Namespace, output_path: Path) -> int:
    """Load models once, process rows sequentially, and save partial results."""
    project_root = args.project_root.expanduser().resolve()
    request = json.loads(args.request_json.expanduser().resolve().read_text(encoding="utf-8"))
    items = request.get("items", [])
    if not isinstance(items, list):
        raise ValueError("request items must be a list")
    config = json.loads((project_root / DEFAULT_TRAINING_CONFIG).read_text(encoding="utf-8"))
    trainer = configured_clap_trainer(project_root)
    pointer_value = trainer.pointer_path.read_text(encoding="utf-8").strip()
    index_path = Path(pointer_value).expanduser()
    if not index_path.is_absolute():
        index_path = project_root / index_path
    index = PrototypeIndex.load(index_path)
    training_labels_by_hash = _training_labels_by_hash(index_path.parent / "training_manifest.csv")
    minimum_margin = float(config.get("minimum_ownership_margin", 0.10))
    minimum_label_examples = int(config.get("minimum_label_examples_for_ownership", 3))
    semantic_prompts, semantic_prompt_families = flattened_semantic_prompts()
    semantic_text_embeddings = trainer.provider.embed_texts(semantic_prompts)
    prompt_index, prompt_brain_status = load_optional_prompt_index(project_root, trainer.provider.model_id)
    panns_provider, panns_status = load_optional_panns_provider(project_root)
    panns_mapping = load_optional_panns_mapping(project_root)

    predictions: list[dict[str, Any]] = []
    row_errors: dict[str, str] = {}
    total_items = len(items)
    _write_prediction_payload(
        output_path,
        status="processing",
        message=f"Neural runtime initialized; 0/{total_items} audio waveform(s) complete.",
        index_path=str(index_path),
        predictions=predictions,
        row_errors=row_errors,
    )

    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError("each request item must be an object")
        row_id = str(item.get("row_id", ""))
        try:
            audio_path = Path(str(item["audio_path"])).expanduser().resolve()
            record = trainer.cache.get_or_compute(audio_path, trainer.provider)
            prediction, exact_training_match = select_neural_prediction(
                index,
                record,
                training_labels_by_hash,
            )
            ownership = assess_prototype_ownership(
                prediction,
                exact_training_match=exact_training_match,
                minimum_margin=minimum_margin,
                minimum_label_examples=minimum_label_examples,
            )
        except Exception as exc:
            row_errors[row_id] = f"{type(exc).__name__}: {exc}"
            _write_progress_checkpoint(
                output_path,
                position=position,
                total_items=total_items,
                index_path=index_path,
                predictions=predictions,
                row_errors=row_errors,
            )
            continue

        semantic_status = "advisory_only"
        semantic_family = ""
        semantic_second_family = ""
        semantic_top_score = 0.0
        semantic_second_score = 0.0
        semantic_margin = 0.0
        semantic_family_scores: dict[str, float] = {}
        try:
            semantic = predict_semantic_family(
                record,
                semantic_text_embeddings,
                semantic_prompt_families,
            )
            semantic_family = semantic.predicted_family
            semantic_second_family = semantic.second_family
            semantic_top_score = semantic.top_score
            semantic_second_score = semantic.second_score
            semantic_margin = semantic.margin
            semantic_family_scores = dict(semantic.family_scores)
        except Exception:
            semantic_status = "runtime_error"

        prompt_suggestions, row_prompt_status = predict_optional_prompt_suggestions(
            prompt_index,
            record,
            prompt_brain_status,
        )
        panns_events, row_panns_status, panns_model_id = predict_optional_panns(
            panns_provider,
            audio_path,
            panns_status,
        )
        panns_evidence = map_optional_panns_evidence(
            panns_mapping,
            panns_events,
            prediction.predicted_label,
        )
        panns_family_scores = map_optional_panns_family_scores(
            panns_mapping,
            panns_events,
        )
        predictions.append(
            {
                "row_id": row_id,
                "file_sha256": record.file_sha256,
                "predicted_label": prediction.predicted_label,
                "second_label": prediction.second_label,
                "top_similarity": prediction.top_similarity,
                "second_similarity": prediction.second_similarity,
                "margin": prediction.margin,
                "radius_ratio": prediction.radius_ratio,
                "known_distribution": prediction.known_distribution,
                "label_example_count": ownership.label_example_count,
                "exact_training_match": ownership.exact_training_match,
                "ownership_ready": ownership.ready,
                "ownership_block_reason": ownership.reason,
                "semantic_status": semantic_status,
                "semantic_family": semantic_family,
                "semantic_second_family": semantic_second_family,
                "semantic_top_score": semantic_top_score,
                "semantic_second_score": semantic_second_score,
                "semantic_margin": semantic_margin,
                "semantic_family_scores": semantic_family_scores,
                "prompt_brain_status": row_prompt_status,
                "prompt_suggestions": [asdict(score) for score in prompt_suggestions],
                "panns_status": row_panns_status,
                "panns_model_id": panns_model_id,
                "panns_events": [asdict(event) for event in panns_events],
                "panns_family_scores": panns_family_scores,
                "panns_support_score": panns_evidence.support_score,
                "panns_contradiction_score": panns_evidence.contradiction_score,
                "panns_supporting_events": [asdict(event) for event in panns_evidence.supporting_events],
                "panns_contradicting_events": [asdict(event) for event in panns_evidence.contradicting_events],
            }
        )

        _write_progress_checkpoint(
            output_path,
            position=position,
            total_items=total_items,
            index_path=index_path,
            predictions=predictions,
            row_errors=row_errors,
        )

    final_status = "predicted_with_errors" if row_errors else "predicted"
    _write_prediction_payload(
        output_path,
        status=final_status,
        message=(f"Predicted {len(predictions)} of {total_items} audio waveform(s); {len(row_errors)} row error(s)."),
        index_path=str(index_path),
        predictions=predictions,
        row_errors=row_errors,
    )
    return 0


def _write_prediction_payload(
    output_path: Path,
    *,
    status: str,
    message: str,
    index_path: str,
    predictions: list[dict[str, Any]],
    row_errors: dict[str, str],
) -> None:
    """Atomically checkpoint completed neural rows for timeout recovery."""
    payload = {
        "schema_version": 1,
        "status": status,
        "message": message,
        "index_path": index_path,
        "source_name_policy": "audio waveform only; path and row ID are operational metadata",
        "predictions": predictions,
        "row_errors": row_errors,
    }
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)


def _write_progress_checkpoint(
    output_path: Path,
    *,
    position: int,
    total_items: int,
    index_path: Path,
    predictions: list[dict[str, Any]],
    row_errors: dict[str, str],
) -> None:
    """Write one durable per-file progress checkpoint."""
    _write_prediction_payload(
        output_path,
        status="processing",
        message=(
            f"Processed {position}/{total_items} audio waveform(s); "
            f"{len(predictions)} prediction(s), {len(row_errors)} row error(s)."
        ),
        index_path=str(index_path),
        predictions=predictions,
        row_errors=row_errors,
    )


def _read_prediction_payload(output_path: Path) -> dict[str, Any]:
    """Read the last complete checkpoint without trusting partial temp files."""
    if not output_path.is_file():
        return {}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_optional_prompt_index(
    project_root: Path,
    model_id: str,
) -> tuple[ClapPromptIndex | None, str]:
    """Load the advisory prompt index without risking core inference.

    The prompt brain is optional and read-only. Missing or stale prompt
    artifacts must not make the production memory lane unavailable.
    """
    pointer_path = project_root / "config/runtime/neural_clap_prompt_index_path.txt"
    if not pointer_path.is_file():
        return None, "unavailable"
    try:
        raw_index_path = Path(pointer_path.read_text(encoding="utf-8").strip()).expanduser()
        index_path = raw_index_path if raw_index_path.is_absolute() else project_root / raw_index_path
        prompt_index = ClapPromptIndex.load(index_path)
        if prompt_index.metadata.model_id != model_id:
            return None, "model_mismatch"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None, "invalid_index"
    return prompt_index, "advisory_only"


def predict_optional_prompt_suggestions(
    prompt_index: ClapPromptIndex | None,
    record: EmbeddingRecord,
    configured_status: str,
) -> tuple[tuple[Any, ...], str]:
    """Return advisory CLAP prompt guesses without aborting the whole row."""
    if prompt_index is None:
        return (), configured_status
    try:
        return prompt_index.predict(record, top_k=3), "advisory_only"
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        return (), "runtime_error"


def load_optional_panns_provider(project_root: Path) -> tuple[PannsProvider | None, str]:
    """Configure the optional broad PANNs witness without loading its model."""
    try:
        return configured_panns_provider(project_root), "advisory_only"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None, "unavailable"


def load_optional_panns_mapping(project_root: Path) -> PannsMappingRegistry | None:
    """Load the versioned broad-event mapping without enabling ownership."""
    try:
        taxonomy = TaxonomyRegistry.load(
            project_root / "config/canonical_taxonomy.json",
            project_root / "config/taxonomy_aliases.json",
        )
        return PannsMappingRegistry.load(project_root / "config/panns_audioset_mapping.json", taxonomy)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def map_optional_panns_evidence(
    mapping: PannsMappingRegistry | None,
    events: tuple[Any, ...],
    candidate_label: str,
) -> PannsMappedEvidence:
    """Map PANNs events to broad support/contradiction for one candidate."""
    if mapping is None:
        return PannsMappedEvidence(candidate_label, 0.0, 0.0, (), (), ())
    mapped_events = tuple(PannsEventScore(str(event.label), float(event.score)) for event in events)
    return mapping.evaluate(mapped_events, candidate_label)


def map_optional_panns_family_scores(
    mapping: PannsMappingRegistry | None,
    events: tuple[Any, ...],
) -> dict[str, float]:
    """Return candidate-independent grouped PANNs evidence by broad family."""
    if mapping is None:
        return {}
    mapped_events = tuple(PannsEventScore(str(event.label), float(event.score)) for event in events)
    return {row.family: row.score for row in mapping.aggregate_family_scores(mapped_events)}


def predict_optional_panns(
    provider: PannsProvider | None,
    audio_path: Path,
    configured_status: str,
) -> tuple[tuple[Any, ...], str, str]:
    """Return optional PANNs events while preserving core CLAP inference."""
    if provider is None:
        return (), configured_status, ""
    try:
        prediction = provider.predict_file(audio_path, top_k=5)
    except (OSError, RuntimeError, ValueError, ImportError):
        return (), "unavailable", provider.model_id
    return prediction.top_events, "advisory_only", prediction.model_id


def select_neural_prediction(
    index: PrototypeIndex,
    record: EmbeddingRecord,
    training_labels_by_hash: dict[str, str],
) -> tuple[LabelPrediction, bool]:
    """Select exact human memory before prototype generalization."""
    exact_label = training_labels_by_hash.get(record.file_sha256)
    return index.predict(record, exact_label=exact_label), exact_label is not None


def _training_labels_by_hash(path: Path) -> dict[str, str]:
    """Load unambiguous exact content/label pairs from the active manifest."""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        labels_by_hash: dict[str, set[str]] = {}
        for row in csv.DictReader(handle):
            file_sha256 = str(row.get("file_sha256", "")).strip()
            label = str(row.get("label", "")).strip()
            if file_sha256 and label:
                labels_by_hash.setdefault(file_sha256, set()).add(label)
    return {file_sha256: next(iter(labels)) for file_sha256, labels in labels_by_hash.items() if len(labels) == 1}


if __name__ == "__main__":
    raise SystemExit(main())
