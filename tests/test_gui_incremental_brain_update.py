from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from aaron_audio_intelligence.shape_memory_brain import (
    SHAPE_MEMORY_BRAIN_NAME,
    SHAPE_STARTER_MEMORY_BRAIN_NAME,
    build_empty_shape_memory_brain,
    merge_shape_memory_into_brain,
    shape_memory_match_for_facts,
    shape_target_for_label,
    update_shape_memory_with_row,
)
from aaron_audio_intelligence.user_memory_brain import (
    USER_MEMORY_BRAIN_NAME,
    build_empty_user_memory_brain,
    merge_user_memory_into_brain,
)
from aaron_sound_sorter.core import FEATURE_WEIGHTS, FP_SIZE, FeatureRow
from aaron_sound_sorter.domain.models import AudioPhysics, SharedAudioFacts
from aaron_sound_sorter.gui import incremental_brain_update as updater_module
from aaron_sound_sorter.gui.incremental_brain_update import (
    DEFAULT_HUMAN_OVERRIDE_EVIDENCE_WEIGHT,
    MAX_HUMAN_OVERRIDE_EVIDENCE_WEIGHT,
    IncrementalBrainUpdater,
    IncrementalCorrection,
    append_training_example,
    bounded_human_override_weight,
    corrections_from_import_manifest,
    dynamic_human_override_weight,
    update_brain_label_with_row,
)
from aaron_sound_sorter.voters.brain_voter import BrainVoter
from aaron_sound_sorter.voters.physics_voter import PhysicsVoter
from aaron_sound_sorter.voters.scoring_tools import PROFILE_FEATURES
from aaron_sound_sorter.voters.shape_voter import ShapeVoter


def _fingerprint(value: float = 0.25) -> list[float]:
    vector = np.zeros(FP_SIZE, dtype=np.float32)
    vector[0] = value
    vector[28] = 0.10
    vector[42] = 0.25
    vector[49] = 0.80
    return vector.astype(float).tolist()


def _prefix_fingerprint(value: float, *, width: int = 10) -> list[float]:
    vector = np.zeros(FP_SIZE, dtype=np.float32)
    vector[:width] = value
    return vector.astype(float).tolist()


def _base_brain(labels: list[str] | None = None) -> dict:
    labels = labels or []
    return {
        "brain_type": "phase3_pure_prototype_brain",
        "version": "unit-test",
        "feature_names": [f"f{i}" for i in range(FP_SIZE)],
        "feature_size": FP_SIZE,
        "feature_weights": FEATURE_WEIGHTS.astype(float).tolist(),
        "scaler_mean": [0.0] * FP_SIZE,
        "scaler_std": [1.0] * FP_SIZE,
        "scalers_by_structure": {
            "one_shot": {"mean": [0.0] * FP_SIZE, "std": [1.0] * FP_SIZE, "count": 1},
            "loop": {"mean": [0.0] * FP_SIZE, "std": [1.0] * FP_SIZE, "count": 1},
            "long_fx": {"mean": [0.0] * FP_SIZE, "std": [1.0] * FP_SIZE, "count": 1},
        },
        "labels": labels,
        "counts": {label: 1 for label in labels},
        "effective_counts": {label: 1 for label in labels},
        "raw_counts": {label: 1 for label in labels},
        "top_by_label": {label: label.split("/", 1)[0] for label in labels},
        "structure_by_label": {label: "one_shot" for label in labels},
        "centroid_counts": {label: 1 for label in labels},
        "label_models_by_label": {},
        "label_reliability_by_label": {},
        "centroids": {label: [_fingerprint(0.1)] for label in labels},
        "exemplars_by_label": {label: [_fingerprint(0.1)] for label in labels},
        "anchors_by_label": {label: [_fingerprint(0.1)] for label in labels},
        "examples_by_label": {},
        "training_examples_detailed_by_label": {},
        "category_fact_profiles": {},
        "category_rival_contrast_facts": {},
        "max_anchors_per_label": 12,
    }


def _row(label: str, source: Path, value: float = 0.25) -> FeatureRow:
    structure = "loop" if label.endswith("/Loops") else "one_shot"
    return FeatureRow(
        path=str(source),
        group_key=label,
        label=label,
        top=label.split("/", 1)[0],
        structure=structure,
        duration_sec=1.0,
        fingerprint=_fingerprint(value),
        read_status="ok",
        source_pack="unit",
    )


def _profile_for_vector(vector: list[float]) -> dict:
    stats = {}
    for index, name in enumerate(PROFILE_FEATURES):
        value = float(vector[index])
        stats[name] = {
            "median": value,
            "p10": value,
            "p90": value,
            "iqr": 1.0,
            "mad": 1.0,
        }
    return {
        "feature_stats": stats,
        "raw_count": 10,
        "effective_count": 10,
        "eligible_count": 10,
        "fact_profile_strength": "ok",
    }


def _facts_for_vector(vector: list[float]) -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={},
        feature_count=FP_SIZE,
        feature_vector=tuple(float(value) for value in vector),
    )


def _audio_physics_for_vector(tmp_path: Path, vector: list[float]) -> AudioPhysics:
    return AudioPhysics(
        source_path=tmp_path / "unit_audio.wav",
        fingerprint=np.asarray(vector, dtype=np.float32),
        duration_sec=4.0,
        read_status="ok",
    )


def test_incremental_update_adds_new_label_without_full_rebuild(tmp_path: Path) -> None:
    label = "Instruments/Plucked Strings/Koto/One Shots"
    brain = _base_brain(["Instruments/Synths/Synth Chord/One Shots"])

    update_brain_label_with_row(brain, _row(label, tmp_path / "koto.wav"))

    assert label in brain["labels"]
    assert brain["counts"][label] == DEFAULT_HUMAN_OVERRIDE_EVIDENCE_WEIGHT
    assert brain["raw_counts"][label] == 1
    assert brain["top_by_label"][label] == "Instruments"
    assert brain["structure_by_label"][label] == "one_shot"
    assert brain["centroids"][label]
    assert brain["anchors_by_label"][label]
    assert brain["category_fact_profiles"][label]["raw_count"] == DEFAULT_HUMAN_OVERRIDE_EVIDENCE_WEIGHT
    assert brain["label_reliability_by_label"][label]["unique_training_count"] == 1
    assert (
        brain["label_reliability_by_label"][label]["human_override_effective_weight"]
        == DEFAULT_HUMAN_OVERRIDE_EVIDENCE_WEIGHT
    )


def test_human_override_weight_is_bounded_and_reinforced(tmp_path: Path) -> None:
    label = "Instruments/Plucked Strings/Koto/Loops"
    brain = _base_brain()
    first_row = _row(label, tmp_path / "koto.wav")

    examples = append_training_example(brain, first_row, evidence_weight=999_999)
    examples = append_training_example(brain, first_row, evidence_weight=999_999)

    assert bounded_human_override_weight(999_999) == MAX_HUMAN_OVERRIDE_EVIDENCE_WEIGHT
    assert len(examples) == 1
    assert examples[0]["human_override_confirmation_count"] == 2
    assert examples[0]["human_override_evidence_weight"] == MAX_HUMAN_OVERRIDE_EVIDENCE_WEIGHT


def test_dynamic_human_override_weight_beats_competing_label_support() -> None:
    label = "Instruments/Plucked Strings/Koto/One Shots"
    competitor = "Instruments/Instrument Loops/Loops"
    brain = _base_brain([competitor])
    brain["counts"][competitor] = 240
    brain["effective_counts"][competitor] = 240
    brain["raw_counts"][competitor] = 240

    weight = dynamic_human_override_weight(brain, label)

    assert weight > 240


def test_corrections_from_import_manifest_uses_duplicate_existing_as_trainable(tmp_path: Path) -> None:
    audio_path = tmp_path / "already_here.wav"
    audio_path.write_bytes(b"fake")
    manifest = tmp_path / "Aaron_GUI_Training_Import.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "source_path",
                "approved_folder",
                "staged_path",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "row_id": "00001",
                "source_path": str(tmp_path / "source.wav"),
                "approved_folder": "Instruments/Plucked Strings/Koto/One Shots",
                "staged_path": str(audio_path),
                "status": "duplicate_existing",
            }
        )

    corrections = corrections_from_import_manifest(manifest)

    assert len(corrections) == 1
    assert corrections[0].audio_path == audio_path
    assert corrections[0].label == "Instruments/Plucked Strings/Koto/One Shots"


def test_incremental_updater_backs_up_and_updates_active_brain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    brain_path = tmp_path / "stage4_folder_brain.json"
    brain_path.write_text(json.dumps(_base_brain()), encoding="utf-8")
    audio_path = tmp_path / "koto.wav"
    audio_path.write_bytes(b"not real audio")
    monkeypatch.setattr(
        updater_module,
        "make_fingerprint_safe",
        lambda path: (np.asarray(_fingerprint(0.55), dtype=np.float32), 1.0, "ok"),
    )
    correction = IncrementalCorrection(
        label="Instruments/Plucked Strings/Koto/One Shots",
        audio_path=audio_path,
        source_path=audio_path,
        row_id="00001",
        status="staged",
    )

    summary = IncrementalBrainUpdater(tmp_path, ("stage4_folder_brain.json",)).apply(
        [correction],
        report_dir=tmp_path / "_reports" / "gui_training_imports" / "run",
        backup_dir=tmp_path / "_reports" / "gui_training_imports" / "run" / "brain_backups",
    )

    updated = json.loads(brain_path.read_text(encoding="utf-8"))
    assert summary.applied_correction_count == 1
    assert summary.updated_brains[0].label_count_after == 1
    assert (summary.backup_dir / "stage4_folder_brain.json").exists()
    assert summary.manifest_path.exists()
    assert "Instruments/Plucked Strings/Koto/One Shots" in updated["labels"]
    assert updated["counts"]["Instruments/Plucked Strings/Koto/One Shots"] == DEFAULT_HUMAN_OVERRIDE_EVIDENCE_WEIGHT
    assert updated["incremental_global_heads_need_full_rebuild"] is True
    assert updated["incremental_gui_update_policy"] == "weighted_prototype_fact_profile_delta_only"


def test_incremental_updater_creates_dedicated_user_memory_brain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    brain_path = tmp_path / "stage4_folder_brain.json"
    brain_path.write_text(json.dumps(_base_brain()), encoding="utf-8")
    audio_path = tmp_path / "koto.wav"
    audio_path.write_bytes(b"not real audio")
    monkeypatch.setattr(
        updater_module,
        "make_fingerprint_safe",
        lambda path: (np.asarray(_fingerprint(0.55), dtype=np.float32), 1.0, "ok"),
    )
    correction = IncrementalCorrection(
        label="Instruments/Plucked Strings/Koto/One Shots",
        audio_path=audio_path,
        source_path=audio_path,
        row_id="00001",
        status="staged",
    )

    summary = IncrementalBrainUpdater(tmp_path, ("stage4_folder_brain.json", USER_MEMORY_BRAIN_NAME)).apply(
        [correction],
        report_dir=tmp_path / "_reports" / "gui_training_imports" / "run",
        backup_dir=tmp_path / "_reports" / "gui_training_imports" / "run" / "brain_backups",
    )

    memory_path = tmp_path / USER_MEMORY_BRAIN_NAME
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    updated_names = {result.brain_path.name for result in summary.updated_brains}
    assert memory["brain_type"] == "user_correction_memory_brain"
    assert "Instruments/Plucked Strings/Koto/One Shots" in memory["labels"]
    assert USER_MEMORY_BRAIN_NAME in updated_names


def test_incremental_updater_creates_dedicated_shape_memory_brain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    brain_path = tmp_path / "stage4_folder_brain.json"
    brain_path.write_text(json.dumps(_base_brain()), encoding="utf-8")
    audio_path = tmp_path / "loop.wav"
    audio_path.write_bytes(b"not real audio")
    monkeypatch.setattr(
        updater_module,
        "make_fingerprint_safe",
        lambda path: (np.asarray(_fingerprint(0.55), dtype=np.float32), 1.0, "ok"),
    )
    correction = IncrementalCorrection(
        label="Instruments/Mixed Musical Loops/Multi Instrument/Loops",
        audio_path=audio_path,
        source_path=audio_path,
        row_id="00001",
        status="staged",
    )

    summary = IncrementalBrainUpdater(tmp_path, ("stage4_folder_brain.json", SHAPE_MEMORY_BRAIN_NAME)).apply(
        [correction],
        report_dir=tmp_path / "_reports" / "gui_training_imports" / "run",
        backup_dir=tmp_path / "_reports" / "gui_training_imports" / "run" / "brain_backups",
    )

    memory_path = tmp_path / SHAPE_MEMORY_BRAIN_NAME
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    updated_names = {result.brain_path.name for result in summary.updated_brains}
    payload = memory["shape_memory"]
    assert memory["brain_type"] == "shape_memory_brain"
    assert "mixed_instrument_loop" in memory["labels"]
    assert payload["shape_counts_by_shape"]["mixed_instrument_loop"] == 1
    assert SHAPE_MEMORY_BRAIN_NAME in updated_names


def test_default_incremental_updater_does_not_mutate_shape_starter_memory() -> None:
    assert SHAPE_MEMORY_BRAIN_NAME in updater_module.DEFAULT_INCREMENTAL_BRAIN_NAMES
    assert SHAPE_STARTER_MEMORY_BRAIN_NAME not in updater_module.DEFAULT_INCREMENTAL_BRAIN_NAMES


def test_shape_memory_target_uses_broad_taxonomy_structure() -> None:
    assert shape_target_for_label("Drums/Drum Loops/Loops", "loop") == "beat_loop"
    assert (
        shape_target_for_label("Instruments/Mixed Musical Loops/Multi Instrument/Loops", "loop")
        == "mixed_instrument_loop"
    )
    assert shape_target_for_label("Instruments/Plucked Strings/Koto/One Shots", "one_shot") == "solo_phrase"
    assert (
        shape_target_for_label(
            "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX",
            "long_fx",
        )
        == "transition_riser"
    )
    assert shape_target_for_label("FX/Designed Noise FX/Siren/Long FX", "long_fx") == "hybrid_fx_motion"
    assert shape_target_for_label("FX/Textures/Noise and Static/Hiss/Long FX", "long_fx") == "texture_bed"


def test_shape_memory_teaches_shape_voter_with_exact_correction(tmp_path: Path) -> None:
    label = "Instruments/Mixed Musical Loops/Multi Instrument/Loops"
    query = _fingerprint(0.55)
    brain = _base_brain()
    memory = build_empty_shape_memory_brain(brain)
    update_shape_memory_with_row(memory, _row(label, tmp_path / "loop.wav", value=0.55), evidence_weight=240)
    merge_shape_memory_into_brain(brain, memory)
    physics = _audio_physics_for_vector(tmp_path, query)
    facts = _facts_for_vector(query)

    result = ShapeVoter().vote(physics, facts, brain)
    shape_vote = result.diagnostics["shape_vote"]

    assert shape_vote["primary_shape"] == "mixed_instrument_loop"
    assert shape_vote["learned_shape_memory_matched"] is True
    assert shape_vote["learned_shape_memory_match_kind"] == "fingerprint"


def test_shape_memory_does_not_fire_for_far_audio(tmp_path: Path) -> None:
    label = "Instruments/Mixed Musical Loops/Multi Instrument/Loops"
    brain = _base_brain()
    memory = build_empty_shape_memory_brain(brain)
    update_shape_memory_with_row(memory, _row(label, tmp_path / "loop.wav", value=0.0), evidence_weight=32)
    merge_shape_memory_into_brain(brain, memory)
    match = shape_memory_match_for_facts(brain, _prefix_fingerprint(0.95, width=FP_SIZE))

    assert match.example_count == 1
    assert match.matched is False


def test_incremental_updater_uses_dynamic_competitor_aware_weight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    competitor = "Instruments/Instrument Loops/Loops"
    label = "Instruments/Plucked Strings/Koto/One Shots"
    brain_path = tmp_path / "stage4_folder_brain.json"
    brain = _base_brain([competitor])
    brain["counts"][competitor] = 180
    brain["effective_counts"][competitor] = 180
    brain["raw_counts"][competitor] = 180
    brain_path.write_text(json.dumps(brain), encoding="utf-8")
    audio_path = tmp_path / "koto.wav"
    audio_path.write_bytes(b"not real audio")
    monkeypatch.setattr(
        updater_module,
        "make_fingerprint_safe",
        lambda path: (np.asarray(_fingerprint(0.55), dtype=np.float32), 1.0, "ok"),
    )
    correction = IncrementalCorrection(
        label=label,
        audio_path=audio_path,
        source_path=audio_path,
        row_id="00001",
        status="staged",
    )

    summary = IncrementalBrainUpdater(tmp_path, ("stage4_folder_brain.json",)).apply(
        [correction],
        report_dir=tmp_path / "_reports" / "gui_training_imports" / "run",
        backup_dir=tmp_path / "_reports" / "gui_training_imports" / "run" / "brain_backups",
    )

    updated = json.loads(brain_path.read_text(encoding="utf-8"))
    assert updated["counts"][label] > updated["counts"][competitor]
    assert updated["label_reliability_by_label"][label]["human_override_effective_weight"] > 180
    assert summary.updated_brains[0].effective_evidence_weight == updated["counts"][label]


def test_user_memory_merge_makes_correction_visible_to_brain_and_physics(tmp_path: Path) -> None:
    label = "Instruments/Plucked Strings/Koto/One Shots"
    competitor = "Instruments/Synths/Synth Chord/One Shots"
    query = _fingerprint(0.55)
    brain = _base_brain([competitor])
    brain["category_fact_profiles"] = {competitor: _profile_for_vector(query)}
    memory = build_empty_user_memory_brain(brain)
    update_brain_label_with_row(memory, _row(label, tmp_path / "koto.wav", value=0.55), evidence_weight=240)

    merge_user_memory_into_brain(brain, memory)

    assert brain["_user_memory_brain_loaded"] is True
    assert label in brain["labels"]
    brain_rows = BrainVoter().score_labels(
        _audio_physics_for_vector(tmp_path, query),
        _facts_for_vector(query),
        brain,
        [label, competitor],
    )
    physics_rows = PhysicsVoter().score_labels(
        _audio_physics_for_vector(tmp_path, query),
        _facts_for_vector(query),
        brain,
        [label, competitor],
        brain["category_fact_profiles"],
    )

    assert sorted(brain_rows, key=lambda row: float(row["score"]))[0]["label"] == label
    assert sorted(physics_rows, key=lambda row: float(row["score"]))[0]["label"] == label


def test_brain_voter_honors_exact_human_override_fingerprint(tmp_path: Path) -> None:
    label = "Instruments/Plucked Strings/Koto/Loops"
    competitor = "Instruments/Instrument Loops/Loops"
    query = _fingerprint(0.55)
    brain = _base_brain([label, competitor])
    brain["centroids"][label] = [_fingerprint(0.05)]
    brain["centroids"][competitor] = [query]
    brain["training_examples_detailed_by_label"][label] = [
        {
            "incremental_gui_correction": True,
            "human_override_evidence_weight": 240,
            "human_override_confirmation_count": 1,
            "fingerprint": query,
        }
    ]
    brain["label_reliability_by_label"][label] = {"human_override_effective_weight": 240}

    rows = BrainVoter().score_labels(
        _audio_physics_for_vector(tmp_path, query),
        _facts_for_vector(query),
        brain,
        [label, competitor],
    )
    ordered = sorted(rows, key=lambda row: float(row["score"]))

    assert ordered[0]["label"] == label
    assert ordered[0]["evidence"]["human_override_exact_audio_match"] is True
    assert ordered[0]["score"] < 0.0


def test_brain_voter_generalizes_multiple_human_corrections_to_nearby_audio(tmp_path: Path) -> None:
    label = "Instruments/Voice/Phrase/One Shots"
    competitor = "Instruments/Instrument Loops/Loops"
    query = _prefix_fingerprint(0.52)
    brain = _base_brain([label, competitor])
    brain["feature_weights"] = [1.0] * FP_SIZE
    brain["centroids"][label] = [_prefix_fingerprint(0.0)]
    brain["centroids"][competitor] = [query]
    brain["training_examples_detailed_by_label"][label] = [
        {
            "incremental_gui_correction": True,
            "human_override_evidence_weight": 240,
            "human_override_confirmation_count": 1,
            "fingerprint": _prefix_fingerprint(0.00),
        },
        {
            "incremental_gui_correction": True,
            "human_override_evidence_weight": 240,
            "human_override_confirmation_count": 1,
            "fingerprint": _prefix_fingerprint(0.04),
        },
    ]
    brain["label_reliability_by_label"][label] = {"human_override_effective_weight": 240}

    rows = BrainVoter().score_labels(
        _audio_physics_for_vector(tmp_path, query),
        _facts_for_vector(query),
        brain,
        [label, competitor],
    )
    ordered = sorted(rows, key=lambda row: float(row["score"]))

    assert ordered[0]["label"] == label
    assert ordered[0]["evidence"]["human_override_exact_audio_match"] is False
    assert ordered[0]["evidence"]["human_override_generalized_audio_match"] is True
    assert ordered[0]["evidence"]["human_override_match_kind"] == "teacher_cloud"
    assert ordered[0]["score"] < 0.0


def test_brain_voter_does_not_generalize_one_human_correction_to_far_audio(tmp_path: Path) -> None:
    label = "Instruments/Voice/Phrase/One Shots"
    competitor = "Instruments/Instrument Loops/Loops"
    query = _prefix_fingerprint(0.85)
    brain = _base_brain([label, competitor])
    brain["feature_weights"] = [1.0] * FP_SIZE
    brain["centroids"][label] = [_prefix_fingerprint(0.0)]
    brain["centroids"][competitor] = [query]
    brain["training_examples_detailed_by_label"][label] = [
        {
            "incremental_gui_correction": True,
            "human_override_evidence_weight": 240,
            "human_override_confirmation_count": 1,
            "fingerprint": _prefix_fingerprint(0.0),
        }
    ]
    brain["label_reliability_by_label"][label] = {"human_override_effective_weight": 240}

    rows = BrainVoter().score_labels(
        _audio_physics_for_vector(tmp_path, query),
        _facts_for_vector(query),
        brain,
        [label, competitor],
    )
    label_row = next(row for row in rows if row["label"] == label)

    assert label_row["evidence"]["human_override_matched"] is False
    assert label_row["evidence"]["human_override_match_kind"] == "none"


def test_physics_voter_honors_exact_human_override_fingerprint(tmp_path: Path) -> None:
    label = "Instruments/Plucked Strings/Koto/Loops"
    competitor = "Instruments/Instrument Loops/Loops"
    query = _fingerprint(0.55)
    brain = _base_brain([label, competitor])
    brain["category_fact_profiles"] = {
        label: _profile_for_vector(_fingerprint(0.05)),
        competitor: _profile_for_vector(query),
    }
    brain["training_examples_detailed_by_label"][label] = [
        {
            "incremental_gui_correction": True,
            "human_override_evidence_weight": 240,
            "human_override_confirmation_count": 1,
            "fingerprint": query,
        }
    ]
    brain["label_reliability_by_label"][label] = {"human_override_effective_weight": 240}

    rows = PhysicsVoter().score_labels(
        _audio_physics_for_vector(tmp_path, query),
        _facts_for_vector(query),
        brain,
        [label, competitor],
        brain["category_fact_profiles"],
    )
    ordered = sorted(rows, key=lambda row: float(row["score"]))

    assert ordered[0]["label"] == label
    assert ordered[0]["evidence"]["human_override_exact_audio_match"] is True
    assert ordered[0]["score"] < 0.0
