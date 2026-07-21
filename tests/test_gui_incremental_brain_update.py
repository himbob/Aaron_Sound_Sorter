from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from aaron_audio_intelligence.physics_memory_brain import (
    PHYSICS_MEMORY_BRAIN_NAME,
    PHYSICS_MEMORY_KEY,
    attach_physics_memory_to_facts,
    build_empty_physics_memory_brain,
    merge_physics_memory_into_brain,
    physics_memory_match_for_facts,
    physics_memory_target_for_label,
    update_physics_memory_with_row,
)
from aaron_audio_intelligence.shape_memory_brain import (
    SHAPE_MEMORY_BRAIN_NAME,
    SHAPE_MEMORY_KEY,
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
from aaron_audio_intelligence.voter_memory_brain import (
    VOTER_MEMORY_BRAIN_NAME,
    VOTER_MEMORY_KEY,
    build_empty_voter_memory_brain,
    merge_voter_memory_into_brain,
    update_voter_memory_with_row,
    voter_memory_match_for_facts,
    voter_memory_role_for_label,
)
from aaron_sound_sorter.core import FEATURE_NAMES, FEATURE_WEIGHTS, FP_SIZE, FeatureRow
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
    supersede_conflicting_label_examples,
    supersede_conflicting_physics_examples,
    supersede_conflicting_shape_examples,
    supersede_conflicting_voter_examples,
    update_brain_label_with_row,
)
from aaron_sound_sorter.voters.brain_voter import BrainVoter
from aaron_sound_sorter.voters.layered_physics_scorer import LayeredPhysicsScorer
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


def _row_with_fingerprint(label: str, source: Path, fingerprint: list[float]) -> FeatureRow:
    structure = "loop" if label.endswith("/Loops") else "one_shot"
    return FeatureRow(
        path=str(source),
        group_key=label,
        label=label,
        top=label.split("/", 1)[0],
        structure=structure,
        duration_sec=3.0 if structure == "loop" else 1.0,
        fingerprint=fingerprint,
        read_status="ok",
        source_pack="unit",
    )


def _base_brain_with_real_feature_names(labels: list[str] | None = None) -> dict:
    brain = _base_brain(labels)
    brain["feature_names"] = list(FEATURE_NAMES)
    brain["feature_weights"] = [1.0] * FP_SIZE
    return brain


def _named_fingerprint(values_by_feature_name: dict[str, float]) -> list[float]:
    vector = np.zeros(FP_SIZE, dtype=np.float32)
    for feature_name, value in values_by_feature_name.items():
        vector[FEATURE_NAMES.index(feature_name)] = float(value)
    return vector.astype(float).tolist()


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


def test_new_gui_correction_supersedes_same_audio_in_wrong_user_memory_label(tmp_path: Path) -> None:
    wrong_label = "Drums/Percussion/Generic Percussion/One Shots"
    right_label = "FX/Impacts and Hits/Generic Impact/One Shots"
    brain = _base_brain()
    update_brain_label_with_row(brain, _row(wrong_label, tmp_path / "old_slot.wav", value=0.42), evidence_weight=80)
    right_row = _row(right_label, tmp_path / "new_slot.wav", value=0.42)

    superseded = supersede_conflicting_label_examples(brain, right_row)
    update_brain_label_with_row(brain, right_row, evidence_weight=240)

    assert superseded == [wrong_label]
    assert wrong_label not in brain["labels"]
    assert wrong_label not in brain["training_examples_detailed_by_label"]
    assert right_label in brain["labels"]
    assert brain["incremental_gui_supersession_history"][-1]["old_target"] == wrong_label


def test_new_gui_correction_supersedes_same_audio_in_wrong_shape_memory(tmp_path: Path) -> None:
    wrong_label = "Drums/Kick Drums/Generic Kick/One Shots"
    right_label = "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX"
    memory = build_empty_shape_memory_brain(_base_brain())
    update_shape_memory_with_row(memory, _row(wrong_label, tmp_path / "kick_slot.wav", value=0.47), evidence_weight=80)
    right_row = _row(right_label, tmp_path / "riser_slot.wav", value=0.47)
    target_shape = shape_target_for_label(right_label, right_row.structure)

    superseded = supersede_conflicting_shape_examples(memory, right_row, target_shape)
    update_shape_memory_with_row(memory, right_row, evidence_weight=240)
    examples_by_shape = memory[SHAPE_MEMORY_KEY]["shape_examples_by_shape"]

    assert superseded
    assert target_shape in examples_by_shape
    assert all(
        example["approved_label"] == right_label for examples in examples_by_shape.values() for example in examples
    )


def test_new_gui_correction_supersedes_same_audio_in_wrong_voter_and_physics_memory(tmp_path: Path) -> None:
    wrong_label = "Drums/Kick Drums/Generic Kick/One Shots"
    right_label = "FX/Impacts and Hits/Generic Impact/One Shots"
    voter_memory = build_empty_voter_memory_brain(_base_brain())
    physics_memory = build_empty_physics_memory_brain(_base_brain())
    wrong_row = _row(wrong_label, tmp_path / "wrong_slot.wav", value=0.52)
    right_row = _row(right_label, tmp_path / "right_slot.wav", value=0.52)
    update_voter_memory_with_row(voter_memory, wrong_row, evidence_weight=80)
    update_physics_memory_with_row(physics_memory, wrong_row, evidence_weight=80)

    supersede_conflicting_voter_examples(
        voter_memory,
        right_row,
        voter_memory_role_for_label(right_label, right_row.structure),
    )
    supersede_conflicting_physics_examples(
        physics_memory,
        right_row,
        physics_memory_target_for_label(right_label, right_row.structure).key,
    )
    update_voter_memory_with_row(voter_memory, right_row, evidence_weight=240)
    update_physics_memory_with_row(physics_memory, right_row, evidence_weight=240)

    assert all(
        example["approved_label"] == right_label
        for examples in voter_memory[VOTER_MEMORY_KEY]["examples_by_role"].values()
        for example in examples
    )
    assert all(
        example["approved_label"] == right_label
        for examples in physics_memory[PHYSICS_MEMORY_KEY]["examples_by_target"].values()
        for example in examples
    )


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
    assert SHAPE_MEMORY_KEY not in updated
    assert VOTER_MEMORY_KEY not in updated
    assert PHYSICS_MEMORY_KEY not in updated
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
    assert SHAPE_MEMORY_KEY not in memory
    assert VOTER_MEMORY_KEY not in memory
    assert PHYSICS_MEMORY_KEY not in memory
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
    assert VOTER_MEMORY_KEY not in memory
    assert PHYSICS_MEMORY_KEY not in memory
    assert SHAPE_MEMORY_BRAIN_NAME in updated_names


def test_incremental_updater_creates_dedicated_voter_memory_brain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    brain_path = tmp_path / "stage4_folder_brain.json"
    brain_path.write_text(json.dumps(_base_brain()), encoding="utf-8")
    audio_path = tmp_path / "impact.wav"
    audio_path.write_bytes(b"not real audio")
    monkeypatch.setattr(
        updater_module,
        "make_fingerprint_safe",
        lambda path: (np.asarray(_fingerprint(0.57), dtype=np.float32), 1.0, "ok"),
    )
    correction = IncrementalCorrection(
        label="FX/Impacts and Hits/Generic Impact/Long FX",
        audio_path=audio_path,
        source_path=audio_path,
        row_id="00001",
        status="staged",
    )

    summary = IncrementalBrainUpdater(tmp_path, ("stage4_folder_brain.json", VOTER_MEMORY_BRAIN_NAME)).apply(
        [correction],
        report_dir=tmp_path / "_reports" / "gui_training_imports" / "run",
        backup_dir=tmp_path / "_reports" / "gui_training_imports" / "run" / "brain_backups",
    )

    memory_path = tmp_path / VOTER_MEMORY_BRAIN_NAME
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    updated_names = {result.brain_path.name for result in summary.updated_brains}
    assert memory["brain_type"] == "voter_memory_brain"
    assert "fx_impact" in memory["labels"]
    assert memory["voter_memory"]["counts_by_role"]["fx_impact"] == 1
    assert SHAPE_MEMORY_KEY not in memory
    assert PHYSICS_MEMORY_KEY not in memory
    assert VOTER_MEMORY_BRAIN_NAME in updated_names


def test_incremental_updater_creates_dedicated_physics_memory_brain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    brain_path = tmp_path / "stage4_folder_brain.json"
    brain_path.write_text(json.dumps(_base_brain()), encoding="utf-8")
    audio_path = tmp_path / "spoken_voice.wav"
    audio_path.write_bytes(b"not real audio")
    monkeypatch.setattr(
        updater_module,
        "make_fingerprint_safe",
        lambda path: (np.asarray(_fingerprint(0.61), dtype=np.float32), 1.0, "ok"),
    )
    correction = IncrementalCorrection(
        label="FX/Human and Voice FX/Spoken Voice/Long FX",
        audio_path=audio_path,
        source_path=audio_path,
        row_id="00001",
        status="staged",
    )

    summary = IncrementalBrainUpdater(tmp_path, ("stage4_folder_brain.json", PHYSICS_MEMORY_BRAIN_NAME)).apply(
        [correction],
        report_dir=tmp_path / "_reports" / "gui_training_imports" / "run",
        backup_dir=tmp_path / "_reports" / "gui_training_imports" / "run" / "brain_backups",
    )

    memory_path = tmp_path / PHYSICS_MEMORY_BRAIN_NAME
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    updated_names = {result.brain_path.name for result in summary.updated_brains}
    assert memory["brain_type"] == "physics_memory_brain"
    assert "FX/HumanCreatureFX" in memory["labels"]
    assert memory["physics_memory"]["counts_by_target"]["FX/HumanCreatureFX"] == 1
    assert SHAPE_MEMORY_KEY not in memory
    assert VOTER_MEMORY_KEY not in memory
    assert PHYSICS_MEMORY_BRAIN_NAME in updated_names


def test_default_incremental_updater_does_not_mutate_shape_starter_memory() -> None:
    assert "stage4_folder_brain.json" not in updater_module.DEFAULT_INCREMENTAL_BRAIN_NAMES
    assert "stage4_folder_brain_core_baby.json" not in updater_module.DEFAULT_INCREMENTAL_BRAIN_NAMES
    assert "stage4_folder_brain_spread_baby.json" not in updater_module.DEFAULT_INCREMENTAL_BRAIN_NAMES
    assert "stage4_folder_brain_outlier_baby.json" not in updater_module.DEFAULT_INCREMENTAL_BRAIN_NAMES
    assert USER_MEMORY_BRAIN_NAME in updater_module.DEFAULT_INCREMENTAL_BRAIN_NAMES
    assert SHAPE_MEMORY_BRAIN_NAME in updater_module.DEFAULT_INCREMENTAL_BRAIN_NAMES
    assert VOTER_MEMORY_BRAIN_NAME in updater_module.DEFAULT_INCREMENTAL_BRAIN_NAMES
    assert PHYSICS_MEMORY_BRAIN_NAME in updater_module.DEFAULT_INCREMENTAL_BRAIN_NAMES
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


def test_voter_memory_role_uses_supervised_label_metadata() -> None:
    assert voter_memory_role_for_label("Instruments/Plucked Strings/Koto/One Shots") == "instrument_plucked_one_shot"
    assert voter_memory_role_for_label("Instruments/Keys/Piano/One Shots") == "instrument_keys_one_shot"
    assert voter_memory_role_for_label("FX/Impacts and Hits/Generic Impact/Long FX") == "fx_impact"
    assert voter_memory_role_for_label("FX/Human and Voice FX/Spoken Voice/Long FX") == "fx_human_voice_long_fx"


def test_physics_memory_target_uses_supervised_label_metadata() -> None:
    assert physics_memory_target_for_label("Instruments/Voice/Vocal Loops/Loops").key == "Instruments/Voice"
    assert physics_memory_target_for_label("FX/Human and Voice FX/Spoken Voice/Long FX").key == "FX/HumanCreatureFX"
    assert physics_memory_target_for_label("Instruments/Plucked Strings/Koto/One Shots").key == (
        "Instruments/PluckedString"
    )
    assert physics_memory_target_for_label("Drums/Drum Loops/Loops").key == "Drums/DrumLoop"


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


def test_shape_memory_generalizes_role_across_pitch_register(tmp_path: Path) -> None:
    label = "Instruments/Woodwinds/Saxophone/Loops"
    teacher = _named_fingerprint(
        {
            "f0_median_hz": 220.0,
            "pitch_confidence": 0.88,
            "f0_voiced_ratio": 0.82,
            "harmonic_to_noise_ratio": 0.72,
            "loop_pitched_event_ratio": 0.84,
            "loop_percussive_event_ratio": 0.08,
            "loop_sustained_tonal_frame_ratio": 0.68,
            "event_rate_hz": 2.4,
            "spectral_entropy_mean": 0.34,
            "attack_noise_ratio": 0.22,
            "body_noise_ratio": 0.18,
        }
    )
    query = list(teacher)
    query[FEATURE_NAMES.index("f0_median_hz")] = 440.0
    brain = _base_brain_with_real_feature_names()
    memory = build_empty_shape_memory_brain(brain)
    update_shape_memory_with_row(
        memory,
        _row_with_fingerprint(label, tmp_path / "sax_teacher.wav", teacher),
        evidence_weight=1200,
    )
    merge_shape_memory_into_brain(brain, memory)

    match = shape_memory_match_for_facts(brain, query)

    assert match.matched is True
    assert match.shape == "pitched_repetition_phrase"
    assert match.match_kind == "teacher_shape_signature_cloud"


def test_voter_memory_teaches_low_level_role_with_exact_correction(tmp_path: Path) -> None:
    label = "FX/Impacts and Hits/Generic Impact/Long FX"
    brain = _base_brain([label])
    memory = build_empty_voter_memory_brain(brain)
    update_voter_memory_with_row(memory, _row(label, tmp_path / "impact.wav", value=0.60), evidence_weight=240)
    merge_voter_memory_into_brain(brain, memory)

    match = voter_memory_match_for_facts(brain, _fingerprint(0.60))

    assert match.matched is True
    assert match.role == "fx_impact"
    assert match.top_family == "FX"


def test_voter_memory_generalizes_source_owner_across_pitch_register(tmp_path: Path) -> None:
    label = "Instruments/Woodwinds/Saxophone/Loops"
    teacher = _named_fingerprint(
        {
            "mfcc_mu_1": 0.40,
            "mfcc_mu_2": -0.18,
            "mfcc_std_1": 0.08,
            "f0_median_hz": 196.0,
            "pitch_confidence": 0.90,
            "f0_voiced_ratio": 0.86,
            "f0_stability_cents": 0.64,
            "harmonic_to_noise_ratio": 0.74,
            "harmonic_energy_ratio": 0.78,
            "formant_like_peak_spacing": 0.52,
            "spectral_envelope_slope": -0.18,
            "attack_noise_ratio": 0.24,
            "body_noise_ratio": 0.16,
            "loop_pitched_event_ratio": 0.82,
            "loop_percussive_event_ratio": 0.07,
            "loop_sustained_tonal_frame_ratio": 0.71,
        }
    )
    query = list(teacher)
    query[FEATURE_NAMES.index("f0_median_hz")] = 392.0
    query[FEATURE_NAMES.index("top1_peak_frequency_hz")] = 784.0
    brain = _base_brain_with_real_feature_names([label])
    memory = build_empty_voter_memory_brain(brain)
    update_voter_memory_with_row(
        memory,
        _row_with_fingerprint(label, tmp_path / "sax_teacher.wav", teacher),
        evidence_weight=1200,
    )
    merge_voter_memory_into_brain(brain, memory)

    match = voter_memory_match_for_facts(brain, query)

    assert match.matched is True
    assert match.role == "instrument_wind_loop"
    assert match.top_family == "Instruments"
    assert match.match_kind == "teacher_role_signature_cloud"


def test_physics_memory_teaches_physics_branch_with_exact_correction(tmp_path: Path) -> None:
    label = "Instruments/Voice/Phrase/One Shots"
    query = _fingerprint(0.60)
    brain = _base_brain()
    memory = build_empty_physics_memory_brain(brain)
    update_physics_memory_with_row(memory, _row(label, tmp_path / "voice.wav", value=0.60), evidence_weight=240)
    merge_physics_memory_into_brain(brain, memory)

    match = physics_memory_match_for_facts(brain, query)

    assert match.matched is True
    assert match.target_key == "Instruments/Voice"
    assert match.top_family == "Instruments"
    assert match.branch == "Voice"
    assert match.match_kind == "fingerprint"


def test_physics_memory_generalizes_branch_across_pitch_register(tmp_path: Path) -> None:
    label = "Instruments/Woodwinds/Saxophone/Loops"
    teacher = _named_fingerprint(
        {
            "f0_median_hz": 185.0,
            "low_peak_frequency_hz": 185.0,
            "pitch_confidence": 0.91,
            "f0_voiced_ratio": 0.88,
            "f0_stability_cents": 0.62,
            "harmonic_to_noise_ratio": 0.73,
            "harmonic_peak_count": 0.58,
            "harmonic_energy_ratio": 0.79,
            "formant_like_peak_spacing": 0.54,
            "spectral_envelope_slope": -0.16,
            "attack_noise_ratio": 0.26,
            "body_noise_ratio": 0.18,
            "tail_noise_ratio": 0.20,
            "loop_pitched_event_ratio": 0.80,
            "loop_percussive_event_ratio": 0.06,
            "loop_sustained_tonal_frame_ratio": 0.72,
            "loop_non_event_tonal_ratio": 0.61,
        }
    )
    query = list(teacher)
    query[FEATURE_NAMES.index("f0_median_hz")] = 370.0
    query[FEATURE_NAMES.index("low_peak_frequency_hz")] = 370.0
    query[FEATURE_NAMES.index("top1_peak_frequency_hz")] = 740.0
    brain = _base_brain_with_real_feature_names()
    memory = build_empty_physics_memory_brain(brain)
    update_physics_memory_with_row(
        memory,
        _row_with_fingerprint(label, tmp_path / "sax_teacher.wav", teacher),
        evidence_weight=1200,
    )
    merge_physics_memory_into_brain(brain, memory)

    match = physics_memory_match_for_facts(brain, query)

    assert match.matched is True
    assert match.target_key == "Instruments/Woodwinds"
    assert match.top_family == "Instruments"
    assert match.branch == "Woodwinds"
    assert match.match_kind == "teacher_physics_register_invariant_cloud"


def test_physics_memory_does_not_seed_from_voter_memory_when_dedicated_lane_exists(tmp_path: Path) -> None:
    voter_label = "FX/Impacts and Hits/Generic Impact/Long FX"
    physics_label = "Instruments/Voice/Phrase/One Shots"
    brain = _base_brain()
    voter_memory = build_empty_voter_memory_brain(brain)
    physics_memory = build_empty_physics_memory_brain(brain)
    update_voter_memory_with_row(
        voter_memory, _row(voter_label, tmp_path / "impact.wav", value=0.62), evidence_weight=240
    )
    update_physics_memory_with_row(
        physics_memory,
        _row(physics_label, tmp_path / "voice.wav", value=0.40),
        evidence_weight=240,
    )

    merge_voter_memory_into_brain(brain, voter_memory)
    merge_physics_memory_into_brain(brain, physics_memory)

    payload = brain[PHYSICS_MEMORY_KEY]
    assert physics_memory_target_for_label(physics_label).key in payload["examples_by_target"]
    assert physics_memory_target_for_label(voter_label).key not in payload["examples_by_target"]
    assert brain["_physics_memory_brain_loaded"] is True


def test_physics_memory_can_migrate_legacy_voter_memory_when_no_dedicated_lane_exists(tmp_path: Path) -> None:
    voter_label = "FX/Impacts and Hits/Generic Impact/Long FX"
    brain = _base_brain()
    voter_memory = build_empty_voter_memory_brain(brain)
    update_voter_memory_with_row(
        voter_memory, _row(voter_label, tmp_path / "impact.wav", value=0.62), evidence_weight=240
    )

    merge_voter_memory_into_brain(brain, voter_memory)
    merge_physics_memory_into_brain(brain, None)

    payload = brain[PHYSICS_MEMORY_KEY]
    assert physics_memory_target_for_label(voter_label).key in payload["examples_by_target"]
    assert brain["_physics_memory_brain_loaded"] is False


def test_physics_memory_calibrates_layered_physics_scoring(tmp_path: Path) -> None:
    label = "Instruments/Voice/Phrase/One Shots"
    competitor = "Drums/Drum Loops/Loops"
    query = _fingerprint(0.62)
    brain = _base_brain()
    memory = build_empty_physics_memory_brain(brain)
    update_physics_memory_with_row(memory, _row(label, tmp_path / "voice.wav", value=0.62), evidence_weight=240)
    merge_physics_memory_into_brain(brain, memory)
    facts = _facts_for_vector(query)
    attach_physics_memory_to_facts(brain, facts)
    scorer = LayeredPhysicsScorer()
    decision = scorer.analyze(facts)

    voice_score, voice_evidence = scorer.apply(label, 0.90, decision)
    drum_score, drum_evidence = scorer.apply(competitor, 0.90, decision)

    assert decision.top_family == "Instruments"
    assert decision.branch == "Voice"
    assert voice_score < drum_score
    assert voice_evidence["physics_memory_score_calibration"] == "applied"
    assert drum_evidence["physics_memory_score_calibration"] == "applied"


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
