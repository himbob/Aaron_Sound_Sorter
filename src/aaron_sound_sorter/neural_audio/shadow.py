"""Legacy-versus-neural shadow comparison reports."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .contracts import EmbeddingRecord, ShadowDiffRow
from .prototype_index import PrototypeIndex


def build_shadow_rows(
    index: PrototypeIndex,
    records: Sequence[tuple[Path, EmbeddingRecord]],
    *,
    legacy_folders_by_hash: Mapping[str, str] | None = None,
    objective_structure_by_hash: Mapping[str, str] | None = None,
    trainer_warnings_by_hash: Mapping[str, str] | None = None,
) -> tuple[ShadowDiffRow, ...]:
    legacy = legacy_folders_by_hash or {}
    objective_structure = objective_structure_by_hash or {}
    trainer_warnings = trainer_warnings_by_hash or {}
    rows: list[ShadowDiffRow] = []
    for path, record in records:
        prediction = index.predict(record)
        legacy_folder = legacy.get(record.file_sha256, "")
        agreement = bool(legacy_folder and legacy_folder.strip("/") == prediction.predicted_label.strip("/"))
        rows.append(
            ShadowDiffRow(
                file_sha256=record.file_sha256,
                display_name=path.name,
                legacy_folder=legacy_folder,
                neural_label=prediction.predicted_label,
                neural_similarity=prediction.top_similarity,
                neural_second_label=prediction.second_label,
                neural_second_similarity=prediction.second_similarity,
                neural_margin=prediction.margin,
                neural_prototype_id=prediction.prototype_id,
                neural_prototype_support_count=int(prediction.evidence.get("prototype_support_count", 0)),
                neural_prototype_radius_p95=prediction.prototype_radius_p95,
                neural_category_spread_p95=float(prediction.evidence.get("category_spread_p95", 0.0)),
                neural_distance_to_prototype=prediction.distance_to_prototype,
                neural_radius_ratio=prediction.radius_ratio,
                neural_known_distribution=prediction.known_distribution,
                agreement=agreement,
                provider_id=prediction.provider_id,
                objective_structure_evidence=objective_structure.get(record.file_sha256, ""),
                trainer_warnings=trainer_warnings.get(record.file_sha256, ""),
                human_verdict="",
                notes="" if legacy_folder else "legacy result unavailable",
            )
        )
    return tuple(rows)


def write_shadow_csv(rows: Iterable[ShadowDiffRow], path: Path, *, disagreements_first: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(rows)
    if disagreements_first:
        ordered.sort(key=lambda row: (row.agreement, -row.neural_margin, row.display_name))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file_sha256",
                "display_name",
                "legacy_folder",
                "neural_label",
                "neural_similarity",
                "neural_second_label",
                "neural_second_similarity",
                "neural_margin",
                "neural_prototype_id",
                "neural_prototype_support_count",
                "neural_prototype_radius_p95",
                "neural_category_spread_p95",
                "neural_distance_to_prototype",
                "neural_radius_ratio",
                "neural_known_distribution",
                "agreement",
                "provider_id",
                "objective_structure_evidence",
                "trainer_warnings",
                "human_verdict",
                "notes",
            ]
        )
        for row in ordered:
            writer.writerow(
                [
                    row.file_sha256,
                    row.display_name,
                    row.legacy_folder,
                    row.neural_label,
                    f"{row.neural_similarity:.8f}",
                    row.neural_second_label,
                    f"{row.neural_second_similarity:.8f}",
                    f"{row.neural_margin:.8f}",
                    row.neural_prototype_id,
                    row.neural_prototype_support_count,
                    f"{row.neural_prototype_radius_p95:.8f}",
                    f"{row.neural_category_spread_p95:.8f}",
                    f"{row.neural_distance_to_prototype:.8f}",
                    f"{row.neural_radius_ratio:.8f}",
                    int(row.neural_known_distribution),
                    int(row.agreement),
                    row.provider_id,
                    row.objective_structure_evidence,
                    row.trainer_warnings,
                    row.human_verdict,
                    row.notes,
                ]
            )
