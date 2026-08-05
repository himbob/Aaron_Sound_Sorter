from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import soundfile as sf

from aaron_sound_sorter.neural_audio.cache import EmbeddingCache
from aaron_sound_sorter.neural_audio.contracts import EmbeddingRecord, l2_normalize
from aaron_sound_sorter.neural_audio.gui_training import (
    NeuralPrototypeTrainer,
    NeuralTrainingInbox,
)
from aaron_sound_sorter.neural_audio.hashing import (
    decoded_audio_sha256,
    normalized_audio_sha256,
    sha256_file,
)
from aaron_sound_sorter.neural_audio.prototype_index import PrototypeIndex

SYNTH_LABEL = "Instruments/Synths/Synth Pad/Loops"
VOICE_LABEL = "Instruments/Voice/Vocal Loops/Loops"


class ContentVectorProvider:
    """Small provider whose vectors are selected only by content hash."""

    provider_id = "content_test"
    model_id = "content-test-v1"

    def __init__(self, vectors_by_hash: Mapping[str, np.ndarray]) -> None:
        self.vectors_by_hash = dict(vectors_by_hash)
        self.embed_calls = 0

    @property
    def cache_identity(self) -> Mapping[str, str | int]:
        return {"provider_id": self.provider_id, "model_id": self.model_id, "schema_version": 1}

    def embed_file(self, path: Path, *, file_sha256: str | None = None) -> EmbeddingRecord:
        self.embed_calls += 1
        digest = file_sha256 or sha256_file(path)
        return EmbeddingRecord(
            provider_id=self.provider_id,
            model_id=self.model_id,
            file_sha256=digest,
            vector=l2_normalize(self.vectors_by_hash[digest]),
            segment_count=1,
            sample_rate=16000,
        )


def write_audio(path: Path, frequency: float) -> Path:
    """Write deterministic real audio for content-hash tests."""
    sample_rate = 16000
    time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.2 * np.sin(2.0 * np.pi * frequency * time_axis)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return path


def write_import_manifest(path: Path, staged_path: Path, label: str) -> Path:
    """Write the small GUI-import contract consumed by neural intake."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["approved_folder", "staged_path", "status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "approved_folder": label,
                "staged_path": staged_path,
                "status": "staged",
            }
        )
    return path


def audio_identity(path: Path) -> dict[str, str]:
    """Return the content identities stored by neural manifests."""
    return {
        "file_sha256": sha256_file(path),
        "decoded_audio_sha256": decoded_audio_sha256(path),
        "normalized_audio_sha256": normalized_audio_sha256(path),
    }


def write_base_split(
    path: Path,
    synth_paths: list[Path],
    voice_path: Path,
    correction_path: Path,
) -> None:
    """Write baseline training rows plus a held-out copy of the correction."""
    fieldnames = [
        "audio_path",
        "intended_label",
        "source_kind",
        "allowed_use",
        "duplicate_group_id",
        "file_sha256",
        "decoded_audio_sha256",
        "normalized_audio_sha256",
    ]
    rows: list[dict[str, str]] = []
    for index, audio_path in enumerate([*synth_paths, voice_path]):
        identity = audio_identity(audio_path)
        rows.append(
            {
                "audio_path": f"project://{audio_path.relative_to(path.parents[2]).as_posix()}",
                "intended_label": SYNTH_LABEL if audio_path in synth_paths else VOICE_LABEL,
                "source_kind": "locked_human_seed",
                "allowed_use": "prototype_training",
                "duplicate_group_id": f"base_{index}",
                **identity,
            }
        )
    correction_identity = audio_identity(correction_path)
    rows.append(
        {
            "audio_path": f"project://{correction_path.relative_to(path.parents[2]).as_posix()}",
            "intended_label": VOICE_LABEL,
            "source_kind": "locked_human_seed",
            "allowed_use": "final_heldout_evaluation",
            "duplicate_group_id": "heldout_correction",
            **correction_identity,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_neural_inbox_identity_and_target_ignore_filename(tmp_path: Path) -> None:
    staged = write_audio(tmp_path / "training" / "original_pad.wav", 220.0)
    first_manifest = write_import_manifest(tmp_path / "_reports/run_1/import.csv", staged, SYNTH_LABEL)
    inbox = NeuralTrainingInbox(tmp_path)

    first = inbox.queue_import_manifest(first_manifest)
    renamed = staged.with_name("misleading_voice_name.wav")
    renamed.write_bytes(staged.read_bytes())
    second_manifest = write_import_manifest(tmp_path / "_reports/run_2/import.csv", renamed, SYNTH_LABEL)
    second = inbox.queue_import_manifest(second_manifest)

    current_rows = list(csv.DictReader(first.current_manifest_path.open(encoding="utf-8")))
    assert first.queued_count == 1
    assert second.reaffirmed_count == 1
    assert len(current_rows) == 1
    assert current_rows[0]["approved_folder"] == SYNTH_LABEL
    assert current_rows[0]["confirmation_count"] == "2"
    assert current_rows[0]["candidate_id"].startswith("gui_audio_group_")
    assert "original_pad" not in current_rows[0]["candidate_id"]
    assert "voice_name" not in current_rows[0]["candidate_id"]
    assert current_rows[0]["allowed_use"] == "prototype_training"


def test_newer_gui_label_supersedes_same_audio_without_duplicate_training(tmp_path: Path) -> None:
    staged = write_audio(tmp_path / "training" / "sound.wav", 330.0)
    inbox = NeuralTrainingInbox(tmp_path)
    inbox.queue_import_manifest(write_import_manifest(tmp_path / "_reports/run_1/import.csv", staged, VOICE_LABEL))

    summary = inbox.queue_import_manifest(
        write_import_manifest(tmp_path / "_reports/run_2/import.csv", staged, SYNTH_LABEL)
    )

    current_rows = list(csv.DictReader(summary.current_manifest_path.open(encoding="utf-8")))
    assert summary.queued_count == 1
    assert summary.superseded_count == 1
    assert len(current_rows) == 1
    assert current_rows[0]["approved_folder"] == SYNTH_LABEL
    assert current_rows[0]["confirmation_count"] == "1"
    events = [json.loads(line) for line in summary.event_log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["action"] for event in events] == ["queued", "relabeled"]


def test_legacy_confirmation_totals_cannot_bypass_new_policy(tmp_path: Path) -> None:
    staged = write_audio(tmp_path / "training" / "sound.wav", 330.0)
    inbox = NeuralTrainingInbox(tmp_path)
    manifest = write_import_manifest(tmp_path / "_reports/run_1/import.csv", staged, SYNTH_LABEL)
    initial = inbox.queue_import_manifest(manifest)
    rows = list(csv.DictReader(initial.current_manifest_path.open(encoding="utf-8")))
    rows[0]["confirmation_count"] = "99"
    rows[0]["confirmation_policy"] = "latest_explicit_user_label_by_content_hash_v2"
    with initial.current_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = inbox.queue_import_manifest(manifest)
    current = list(csv.DictReader(summary.current_manifest_path.open(encoding="utf-8")))

    assert current[0]["confirmation_count"] == "1"
    assert current[0]["confirmation_policy"] == "two_step_locked_seed_override_by_content_hash_v3"


def test_inbox_collapses_same_audio_from_older_normalization_hash_schema(tmp_path: Path) -> None:
    staged = write_audio(tmp_path / "training" / "sound.wav", 330.0)
    inbox = NeuralTrainingInbox(tmp_path)
    manifest = write_import_manifest(tmp_path / "_reports/run_1/import.csv", staged, SYNTH_LABEL)
    initial = inbox.queue_import_manifest(manifest)
    rows = list(csv.DictReader(initial.current_manifest_path.open(encoding="utf-8")))
    stale = dict(rows[0])
    stale["candidate_id"] = "gui_stale_normalization_schema"
    stale["normalized_audio_sha256"] = "f" * 64
    stale["duplicate_group_id"] = "audio_group_stale"
    stale["last_approved_utc"] = "2026-01-01T00:00:00+00:00"
    with initial.current_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows([stale, rows[0]])

    summary = inbox.queue_import_manifest(manifest)
    current = list(csv.DictReader(summary.current_manifest_path.open(encoding="utf-8")))

    assert len(current) == 1
    assert current[0]["file_sha256"] == sha256_file(staged)


def test_stable_prototype_rebuild_teaches_correction_and_is_filename_invariant(
    tmp_path: Path,
) -> None:
    training_root = tmp_path / "training"
    synth_paths = [write_audio(training_root / f"synth_{index}.wav", 180.0 + index) for index in range(3)]
    voice_path = write_audio(training_root / "voice.wav", 440.0)
    correction_path = write_audio(training_root / "correction.wav", 330.0)
    base_split = tmp_path / "neural_artifacts/base/dataset_splits.csv"
    write_base_split(base_split, synth_paths, voice_path, correction_path)
    inbox = NeuralTrainingInbox(tmp_path)
    intake = inbox.queue_import_manifest(
        write_import_manifest(tmp_path / "_reports/gui/import.csv", correction_path, SYNTH_LABEL)
    )

    vectors_by_hash = {
        **{sha256_file(path): np.array([1.0, 0.0], dtype=np.float32) for path in synth_paths},
        sha256_file(voice_path): np.array([0.0, 1.0], dtype=np.float32),
        sha256_file(correction_path): np.array([0.35, 0.94], dtype=np.float32),
    }
    provider = ContentVectorProvider(vectors_by_hash)
    pointer_path = tmp_path / "config/runtime/neural_clap_index_path.txt"
    trainer = NeuralPrototypeTrainer(
        project_root=tmp_path,
        provider=provider,
        cache_root=tmp_path / "neural_artifacts/cache",
        index_root=tmp_path / "neural_artifacts/indexes",
        pointer_path=pointer_path,
        base_split_path=base_split,
        inbox_path=intake.current_manifest_path,
        training_root=training_root,
    )

    summary = trainer.rebuild()

    assert summary.status == "built"
    assert summary.index_path is not None
    assert summary.correction_predictions[0].predicted_before == VOICE_LABEL
    assert summary.correction_predictions[0].predicted_after == SYNTH_LABEL
    assert summary.correction_predictions[0].prototype_predicted_after == SYNTH_LABEL
    assert summary.correction_predictions[0].known_distribution_after
    assert summary.invalidated_evaluation_hashes == (sha256_file(correction_path),)
    assert pointer_path.read_text(encoding="utf-8") == "neural_artifacts/indexes/active/index\n"
    assert summary.index_path == tmp_path / "neural_artifacts/indexes/active/index"
    assert json.loads(summary.report_path.read_text(encoding="utf-8"))["production_ownership_enabled"] is False

    repeated_summary = trainer.rebuild()
    assert repeated_summary.status == "unchanged"
    assert repeated_summary.index_path == summary.index_path
    assert len(list((tmp_path / "neural_artifacts/indexes").glob("run_*"))) == 0
    assert (tmp_path / "neural_artifacts/indexes/active/build_summary.json").is_file()

    renamed = tmp_path / "completely_different_display_name.wav"
    renamed.write_bytes(correction_path.read_bytes())
    cache = EmbeddingCache(tmp_path / "neural_artifacts/cache")
    original_record = cache.get_or_compute(correction_path, provider)
    renamed_record = cache.get_or_compute(renamed, provider)
    index = PrototypeIndex.load(summary.index_path)
    assert np.array_equal(original_record.vector, renamed_record.vector)
    assert index.predict(original_record).predicted_label == SYNTH_LABEL
    assert index.predict(renamed_record).predicted_label == SYNTH_LABEL


def test_locked_seed_relabel_requires_two_identical_gui_approvals(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    trusted_audio = write_audio(training_root / "audio.wav", 440.0)
    base_split = tmp_path / "neural_artifacts/base/dataset_splits.csv"
    write_base_split(base_split, [], trusted_audio, trusted_audio)
    import_manifest = write_import_manifest(
        tmp_path / "_reports/gui/import.csv",
        trusted_audio,
        SYNTH_LABEL,
    )
    inbox = NeuralTrainingInbox(tmp_path)
    first_intake = inbox.queue_import_manifest(import_manifest)
    provider = ContentVectorProvider({sha256_file(trusted_audio): np.array([0.0, 1.0], dtype=np.float32)})
    trainer = NeuralPrototypeTrainer(
        project_root=tmp_path,
        provider=provider,
        cache_root=tmp_path / "neural_artifacts/cache",
        index_root=tmp_path / "neural_artifacts/indexes",
        pointer_path=tmp_path / "config/runtime/index.txt",
        base_split_path=base_split,
        inbox_path=first_intake.current_manifest_path,
        training_root=training_root,
    )

    first_build = trainer.rebuild()

    assert len(first_build.pending_training_conflicts) == 1
    assert first_build.pending_training_conflicts[0].locked_label == VOICE_LABEL
    assert first_build.pending_training_conflicts[0].proposed_label == SYNTH_LABEL
    assert first_build.pending_training_conflicts[0].confirmation_count == 1
    assert first_build.pending_training_conflicts[0].required_confirmation_count == 2
    first_manifest = list(csv.DictReader((first_build.index_path.parent / "training_manifest.csv").open()))
    assert {row["label"] for row in first_manifest} == {VOICE_LABEL}

    inbox.queue_import_manifest(import_manifest)
    confirmed_build = trainer.rebuild()

    assert confirmed_build.pending_training_conflicts == ()
    confirmed_manifest = list(csv.DictReader((confirmed_build.index_path.parent / "training_manifest.csv").open()))
    assert {row["label"] for row in confirmed_manifest} == {SYNTH_LABEL}


def test_rebuild_reuses_cached_embedding_when_approved_audio_is_missing(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    trusted_audio = write_audio(training_root / "audio.wav", 440.0)
    base_split = tmp_path / "neural_artifacts/base/dataset_splits.csv"
    write_base_split(base_split, [], trusted_audio, trusted_audio)
    identity = audio_identity(trusted_audio)
    digest = identity["file_sha256"]
    trainer = NeuralPrototypeTrainer(
        project_root=tmp_path,
        provider=ContentVectorProvider({digest: np.array([0.0, 1.0], dtype=np.float32)}),
        cache_root=tmp_path / "neural_artifacts/cache",
        index_root=tmp_path / "neural_artifacts/indexes",
        pointer_path=tmp_path / "config/runtime/index.txt",
        base_split_path=base_split,
        inbox_path=tmp_path / "neural_artifacts/gui_training_inbox/current.csv",
        training_root=training_root,
    )

    initial = trainer.rebuild()
    assert initial.training_example_count == 1
    trusted_audio.unlink()
    shutil.rmtree(initial.index_path.parent)

    rebuilt = trainer.rebuild()

    assert rebuilt.status == "built"
    assert rebuilt.training_example_count == 1
    assert rebuilt.missing_audio_hashes == ()
    assert list(csv.DictReader((rebuilt.index_path.parent / "training_manifest.csv").open())) == [
        {
            "file_sha256": digest,
            "decoded_audio_sha256": identity["decoded_audio_sha256"],
            "normalized_audio_sha256": identity["normalized_audio_sha256"],
            "duplicate_group_id": "base_0",
            "label": VOICE_LABEL,
            "source_kind": "locked_human_seed",
        }
    ]


def test_sample_library_uri_restores_explicit_hash_verified_training_audio(tmp_path: Path) -> None:
    sample_library_root = tmp_path / "external_samples"
    external_audio = write_audio(sample_library_root / "pack" / "approved.wav", 510.0)
    identity = audio_identity(external_audio)
    split_path = tmp_path / "neural_artifacts/base/dataset_splits.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "audio_path",
                "intended_label",
                "source_kind",
                "allowed_use",
                "duplicate_group_id",
                "file_sha256",
                "decoded_audio_sha256",
                "normalized_audio_sha256",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "audio_path": "sample-library://pack/approved.wav",
                "intended_label": VOICE_LABEL,
                "source_kind": "locked_human_seed",
                "allowed_use": "prototype_training",
                "duplicate_group_id": "external_approved",
                **identity,
            }
        )
    provider = ContentVectorProvider({identity["file_sha256"]: np.array([0.0, 1.0], dtype=np.float32)})
    inbox_path = tmp_path / "neural_artifacts/gui_training_inbox/current.csv"
    trainer = NeuralPrototypeTrainer(
        project_root=tmp_path,
        provider=provider,
        cache_root=tmp_path / "neural_artifacts/cache",
        index_root=tmp_path / "neural_artifacts/indexes",
        pointer_path=tmp_path / "config/runtime/index.txt",
        base_split_path=split_path,
        inbox_path=inbox_path,
        training_root=tmp_path / "training",
        sample_library_root=sample_library_root,
    )

    summary = trainer.rebuild()

    assert summary.status == "built"
    assert summary.training_example_count == 1
    assert summary.missing_audio_hashes == ()
    assert summary.index_path is not None
