from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aaron_sound_sorter.domain.models import (
    AudioPhysics,
    CategoryGuess,
    ConsensusDecision,
    SharedAudioFacts,
    SortFileResult,
    VoterResult,
)
from aaron_sound_sorter.gui.models import PreviewRow, SortPreviewSession
from aaron_sound_sorter.gui.preview_service import (
    SortPlanExporter,
    SortPreviewService,
    TrainingCorrectionImporter,
    apply_neural_runtime_authority,
    correction_evidence,
    detected_candidate_folders,
    gui_worker_count,
    load_available_labels,
    load_simple_yaml_mapping,
    training_slot_path,
    write_corrections_csv,
)
from aaron_sound_sorter.neural_audio.runtime import (
    NeuralRuntimeBatch,
    NeuralRuntimePrediction,
)


def _audio_file(tmp_path: Path, name: str = "sample.wav") -> Path:
    path = tmp_path / name
    path.write_bytes(b"fake audio for export tests")
    return path


def _row(source: Path, proposed: str = "FX/Hybrid Designed FX", approved: str | None = None) -> PreviewRow:
    return PreviewRow(
        row_id="00001",
        source_path=source,
        display_name=source.name,
        proposed_folder=proposed,
        approved_folder=approved or proposed,
        final_top=proposed.split("/", 1)[0],
        consensus_status="unit_test",
        confidence=0.75,
        duration_sec=1.25,
        read_status="ok",
        decision_reason="unit test row",
        diagnostic_summary="shape=test; brain=test; physics=test",
    )


def _session(tmp_path: Path, rows: list[PreviewRow]) -> SortPreviewSession:
    brain = tmp_path / "brain.json"
    brain.write_text(json.dumps({"labels": ["Drums/Kicks/One Shots"]}), encoding="utf-8")
    return SortPreviewSession(
        run_dir=tmp_path / "run",
        input_path=tmp_path,
        brain_path=brain,
        available_labels=["Drums/Kicks/One Shots"],
        rows=rows,
    )


def _neural_prediction(
    label: str,
    *,
    known: bool,
    row_id: str = "00001",
    margin: float = 0.26,
    label_example_count: int = 4,
    exact_training_match: bool = False,
    ownership_ready: bool | None = None,
    ownership_reason: str = "supported_separated_neighborhood",
) -> NeuralRuntimePrediction:
    ready = known if ownership_ready is None else ownership_ready
    return NeuralRuntimePrediction(
        row_id=row_id,
        file_sha256="a" * 64,
        predicted_label=label,
        second_label="Instruments/Mixed Musical Loops/Multi Instrument/Loops",
        top_similarity=0.81,
        second_similarity=0.55,
        margin=margin,
        radius_ratio=0.72 if known else 2.4,
        known_distribution=known,
        label_example_count=label_example_count,
        exact_training_match=exact_training_match,
        ownership_ready=ready,
        ownership_block_reason=ownership_reason,
    )


def _neural_batch(prediction: NeuralRuntimePrediction) -> NeuralRuntimeBatch:
    return NeuralRuntimeBatch(
        status="predicted",
        message="Predicted one audio waveform.",
        predictions=(prediction,),
    )


def _guess(folder_path: str, rank: int) -> CategoryGuess:
    return CategoryGuess(
        label=folder_path,
        folder_path=folder_path,
        top_family=folder_path.split("/", 1)[0],
        score=float(rank),
        confidence=0.90,
        rank=rank,
        reason="unit test candidate",
    )


def _candidate_result(*, is_loop_like: bool) -> SortFileResult:
    proposed = "Instruments/Bass/Bass Loops" if is_loop_like else "FX/Designed Noise FX/Alarm/One Shots"
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=is_loop_like,
        is_single_event_like=not is_loop_like,
        is_short_hit_like=not is_loop_like,
        is_long=True,
        evidence={
            "duration_sec": 8.0,
            "structure_facts": {
                "event_count_estimate": 9.0 if is_loop_like else 1.0,
                "onset_span_ratio": 0.72 if is_loop_like else 0.10,
            },
            "shape_vote": {
                "primary_shape": "solo_phrase" if is_loop_like else "hit_with_tail",
                "confidence": 0.91,
                "onset_count": 9.0 if is_loop_like else 1.0,
                "onset_span_ratio": 0.72 if is_loop_like else 0.10,
                "true_repetition_score": 0.64 if is_loop_like else 0.0,
                "pitched_event_ratio": 1.0 if is_loop_like else 0.0,
                "percussive_event_ratio": 0.0 if is_loop_like else 0.82,
                "drumlike_frame_ratio": 0.0,
            },
            "measured_roles": {
                "bass_loop": 1.0 if is_loop_like else 0.0,
            },
        },
    )
    brain_votes = VoterResult(
        voter_name="brain",
        guesses=[
            _guess("Instruments/Bass/Electric Bass/One Shots", 1),
            _guess("Instruments/Bass/Electric Bass/Loops", 2),
        ],
    )
    physics_votes = VoterResult(
        voter_name="physics",
        guesses=[
            _guess("FX/Designed Noise FX/Alarm/One Shots", 1),
            _guess("Instruments/Bass/Generic Bass/Loops", 2),
        ],
    )
    decision = ConsensusDecision(
        final_label=proposed,
        final_top=proposed.split("/", 1)[0],
        folder_path=proposed,
        consensus_status="unit_test",
        reason="unit test decision",
        shared_candidates=[
            {"folder_path": "Drums/Kick Drums/Generic Kick/One Shots"},
            {"folder_path": "Instruments/Bass/Synth Bass/Loops"},
        ],
    )
    return SortFileResult(
        source_path=Path("candidate.wav"),
        placed_path=None,
        physics=AudioPhysics(
            source_path=Path("candidate.wav"),
            fingerprint=np.zeros(1, dtype=np.float32),
            duration_sec=8.0,
            read_status="ok",
        ),
        facts=facts,
        brain_votes=brain_votes,
        physics_votes=physics_votes,
        decision=decision,
    )


def test_possible_matches_hide_one_shots_for_decisive_loop_structure() -> None:
    candidates = detected_candidate_folders(_candidate_result(is_loop_like=True))

    assert "Instruments/Bass/Electric Bass/Loops" in candidates
    assert all(not candidate.endswith("/One Shots") for candidate in candidates)


def test_possible_matches_keep_long_one_shot_when_audio_is_not_a_loop() -> None:
    candidates = detected_candidate_folders(_candidate_result(is_loop_like=False))

    assert "FX/Designed Noise FX/Alarm/One Shots" in candidates


def test_known_neural_neighborhood_takes_gui_ownership_for_any_category(tmp_path: Path) -> None:
    row = _row(
        _audio_file(tmp_path, "words_that_must_not_matter.wav"),
        proposed="FX/Hybrid Designed FX",
    )

    apply_neural_runtime_authority(
        [row],
        _neural_batch(_neural_prediction("Drums/Drum Loops/Loops", known=True)),
    )

    assert row.proposed_folder == "Drums/Drum Loops/Loops"
    assert row.approved_folder == "Drums/Drum Loops/Loops"
    assert row.consensus_status == "neural_known_distribution_owner"
    assert row.neural_known_distribution is True
    assert row.neural_folder == "Drums/Drum Loops/Loops"


def test_known_but_ambiguous_voice_sax_prediction_cannot_steal_ownership(tmp_path: Path) -> None:
    row = _row(
        _audio_file(tmp_path, "display_text_is_not_evidence.wav"),
        proposed="Instruments/Winds/Saxophone/Sax Loops",
    )

    apply_neural_runtime_authority(
        [row],
        _neural_batch(
            _neural_prediction(
                "Instruments/Voice/Vocal Loops/Loops",
                known=True,
                margin=0.04,
                ownership_ready=False,
                ownership_reason="ambiguous_nearest_labels",
            )
        ),
    )

    assert row.proposed_folder == "_TO_REVIEW/Measured Role Conflict"
    assert row.consensus_status == "neural_legacy_owner_conflict_review"
    assert row.neural_known_distribution is True
    assert row.neural_ownership_ready is False
    assert row.neural_ownership_reason == "ambiguous_nearest_labels"


def test_sparse_pad_prediction_cannot_steal_keys_ownership(tmp_path: Path) -> None:
    legacy = "Instruments/Keys/Electric Piano/Loops"
    row = _row(_audio_file(tmp_path), proposed=legacy)

    apply_neural_runtime_authority(
        [row],
        _neural_batch(
            _neural_prediction(
                "Instruments/Synths/Synth Pad/Loops",
                known=True,
                label_example_count=1,
                ownership_ready=False,
                ownership_reason="insufficient_label_examples",
            )
        ),
    )

    assert row.proposed_folder == legacy
    assert row.neural_label_example_count == 1
    assert row.neural_ownership_ready is False


def test_unready_drum_prediction_cannot_steal_mixed_instrument_ownership(
    tmp_path: Path,
) -> None:
    row = _row(
        _audio_file(tmp_path),
        proposed="Instruments/Mixed Musical Loops/Multi Instrument/Loops",
    )

    apply_neural_runtime_authority(
        [row],
        _neural_batch(
            _neural_prediction(
                "Drums/Drum Loops/Loops",
                known=False,
                ownership_ready=False,
                ownership_reason="outside_learned_radius",
            )
        ),
    )

    assert row.proposed_folder == "_TO_REVIEW/Measured Role Conflict"
    assert row.consensus_status == "neural_legacy_owner_conflict_review"


def test_unready_altered_voice_prediction_cannot_steal_drum_ownership(
    tmp_path: Path,
) -> None:
    row = _row(
        _audio_file(tmp_path),
        proposed="Drums/Drum Loops/Full Drum Loops/Loops",
    )

    apply_neural_runtime_authority(
        [row],
        _neural_batch(
            _neural_prediction(
                "FX/Human and Voice FX/Altered Voice/One Shots",
                known=False,
                ownership_ready=False,
                ownership_reason="outside_learned_radius",
            )
        ),
    )

    assert row.proposed_folder == "_TO_REVIEW/Measured Role Conflict"
    assert row.consensus_status == "neural_legacy_owner_conflict_review"


def test_exact_human_training_match_owns_even_for_sparse_label(tmp_path: Path) -> None:
    label = "Instruments/Voice/Voice Phrase One Shots/One Shots"
    row = _row(_audio_file(tmp_path), proposed="FX/Human and Voice FX/Altered Voice/Long")

    apply_neural_runtime_authority(
        [row],
        _neural_batch(
            _neural_prediction(
                label,
                known=True,
                margin=0.01,
                label_example_count=1,
                exact_training_match=True,
                ownership_ready=True,
                ownership_reason="exact_human_training_match",
            )
        ),
    )

    assert row.proposed_folder == label
    assert row.consensus_status == "neural_known_distribution_owner"
    assert row.neural_exact_training_match is True


def test_unknown_neural_nonvoice_vs_legacy_voice_forces_review(tmp_path: Path) -> None:
    row = _row(
        _audio_file(tmp_path, "also_not_evidence.wav"),
        proposed="Instruments/Voice/Vocal Loops/Loops",
    )

    apply_neural_runtime_authority(
        [row],
        _neural_batch(
            _neural_prediction(
                "Instruments/Synths/Synth Lead/Loops",
                known=False,
            )
        ),
    )

    assert row.proposed_folder == "_TO_REVIEW/Measured Role Conflict"
    assert row.approved_folder == "_TO_REVIEW/Measured Role Conflict"
    assert row.consensus_status == "neural_legacy_owner_conflict_review"
    assert "instruments_nonvoice vs voice" in row.decision_reason


def test_unknown_neural_cross_owner_review_is_not_voice_specific(tmp_path: Path) -> None:
    row = _row(
        _audio_file(tmp_path),
        proposed="Drums/Snares/Acoustic Snare/One Shots",
    )

    apply_neural_runtime_authority(
        [row],
        _neural_batch(
            _neural_prediction(
                "FX/Impacts and Hits/Generic Impact/One Shots",
                known=False,
            )
        ),
    )

    assert row.proposed_folder == "_TO_REVIEW/Measured Role Conflict"
    assert row.consensus_status == "neural_legacy_owner_conflict_review"


def test_unknown_neural_same_owner_keeps_legacy_proposal(tmp_path: Path) -> None:
    legacy = "Instruments/Synths/Synth Pad/Loops"
    row = _row(_audio_file(tmp_path), proposed=legacy)

    apply_neural_runtime_authority(
        [row],
        _neural_batch(
            _neural_prediction(
                "Instruments/Synths/Synth Lead/Loops",
                known=False,
            )
        ),
    )

    assert row.proposed_folder == legacy
    assert row.approved_folder == legacy
    assert row.neural_folder == "Instruments/Synths/Synth Lead/Loops"


def test_exporter_copies_approved_folder_tree(tmp_path: Path) -> None:
    source = _audio_file(tmp_path)
    row = _row(source, approved="Instruments/Synths/Synth Loops")
    session = _session(tmp_path, [row])

    summary = SortPlanExporter(project_root=tmp_path).export(
        session,
        tmp_path / "destination",
        mode="copy",
    )

    exported = summary.sorted_root / "Instruments" / "Synths" / "Synth Loops" / source.name
    assert exported.exists()
    assert source.exists()
    assert summary.exported_count == 1
    assert summary.corrected_count == 1
    assert summary.approved_plan_path.exists()
    assert summary.corrections_path.exists()


def test_exporter_symlinks_when_requested(tmp_path: Path) -> None:
    source = _audio_file(tmp_path)
    row = _row(source)
    session = _session(tmp_path, [row])

    summary = SortPlanExporter(project_root=tmp_path).export(
        session,
        tmp_path / "destination",
        mode="symlink",
    )

    exported = summary.sorted_root / "FX" / "Hybrid Designed FX" / source.name
    assert exported.is_symlink()
    assert exported.resolve() == source.resolve()


def test_correction_csv_contains_only_changed_rows(tmp_path: Path) -> None:
    changed = _row(_audio_file(tmp_path, "changed.wav"), approved="Drums/Drum Loops/Loops")
    unchanged = _row(_audio_file(tmp_path, "unchanged.wav"))
    session = _session(tmp_path, [changed, unchanged])
    path = tmp_path / "corrections.csv"

    write_corrections_csv(path, session)

    text = path.read_text(encoding="utf-8")
    assert "changed.wav" in text
    assert "unchanged.wav" not in text


def test_correction_evidence_uses_manual_label_without_requiring_result(tmp_path: Path) -> None:
    row = _row(_audio_file(tmp_path), approved="Instruments/Keys/Piano/Loops")

    evidence = correction_evidence(row)

    assert evidence["proposed_folder"] == "FX/Hybrid Designed FX"
    assert evidence["approved_folder"] == "Instruments/Keys/Piano/Loops"
    assert "feature_values_by_name" not in evidence


def test_single_file_correction_evidence_pack_allows_existing_folder(tmp_path: Path) -> None:
    row = _row(_audio_file(tmp_path), approved="Drums/Kicks/One Shots")
    session = _session(tmp_path, [row])

    evidence_path = SortPlanExporter(project_root=tmp_path).write_correction_evidence_package(session)

    text = evidence_path.read_text(encoding="utf-8")
    assert '"approved_folder": "Drums/Kicks/One Shots"' in text
    assert '"proposed_folder": "FX/Hybrid Designed FX"' in text


def test_single_file_correction_evidence_pack_allows_new_folder(tmp_path: Path) -> None:
    row = _row(_audio_file(tmp_path), approved="Instruments/Synths/New Custom Synth Folder/Loops")
    session = _session(tmp_path, [row])

    evidence_path = SortPlanExporter(project_root=tmp_path).write_correction_evidence_package(session)

    text = evidence_path.read_text(encoding="utf-8")
    assert '"approved_folder": "Instruments/Synths/New Custom Synth Folder/Loops"' in text
    assert row.source_path.exists()


def test_training_slot_path_maps_gui_label_to_training_structure(tmp_path: Path) -> None:
    slot = training_slot_path(tmp_path / "training", "Instruments/Plucked Strings/Koto/Loops")

    assert slot == tmp_path / "training" / "Instruments" / "Plucked Strings" / "Koto" / "_LOOPS"


def test_training_correction_importer_copies_corrected_audio_to_training_slot(tmp_path: Path) -> None:
    source = _audio_file(tmp_path, "koto.wav")
    row = _row(
        source,
        proposed="FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX",
        approved="Instruments/Plucked Strings/Koto/Loops",
    )
    session = _session(tmp_path, [row])

    summary = TrainingCorrectionImporter(project_root=tmp_path).import_session(session)

    expected_slot = tmp_path / "training" / "locked_curated_v1" / "Instruments" / "Plucked Strings" / "Koto" / "_LOOPS"
    staged = expected_slot / "koto.wav"
    assert summary.staged_count == 1
    assert summary.skipped_count == 0
    assert staged.exists()
    assert staged.read_bytes() == source.read_bytes()
    assert summary.manifest_path.exists()
    assert summary.correction_evidence_path.exists()
    assert summary.neural_queued_count == 1
    assert summary.neural_intake_path.exists()
    assert summary.neural_event_log_path.exists()


def test_training_correction_importer_reuses_duplicate_audio_in_same_slot(tmp_path: Path) -> None:
    source = _audio_file(tmp_path, "koto.wav")
    existing_slot = tmp_path / "training" / "locked_curated_v1" / "Instruments" / "Plucked Strings" / "Koto" / "_LOOPS"
    existing_slot.mkdir(parents=True)
    existing_copy = existing_slot / "already_here.wav"
    existing_copy.write_bytes(source.read_bytes())
    row = _row(source, approved="Instruments/Plucked Strings/Koto/Loops")
    session = _session(tmp_path, [row])

    summary = TrainingCorrectionImporter(project_root=tmp_path).import_session(session)

    assert summary.staged_count == 0
    assert summary.reused_existing_count == 1
    assert summary.skipped_count == 0
    assert summary.errors == []
    assert not (existing_slot / "koto.wav").exists()


def test_training_correction_importer_skips_untrainable_review_folder(tmp_path: Path) -> None:
    source = _audio_file(tmp_path, "review.wav")
    row = _row(source, approved="_TO_REVIEW/Needs Human Review")
    session = _session(tmp_path, [row])

    summary = TrainingCorrectionImporter(project_root=tmp_path).import_session(session)

    assert summary.staged_count == 0
    assert summary.skipped_count == 1
    assert "not a valid training label" in summary.errors[0] or "review folders are not trainable" in summary.errors[0]


def test_training_correction_importer_archives_same_audio_from_wrong_slot(tmp_path: Path) -> None:
    source = _audio_file(tmp_path, "corrected.wav")
    wrong_slot = tmp_path / "training" / "locked_curated_v1" / "Instruments" / "Bass" / "Synth Bass" / "_LOOPS"
    wrong_slot.mkdir(parents=True)
    wrong_teacher = wrong_slot / "wrong_teacher.wav"
    wrong_teacher.write_bytes(source.read_bytes())
    row = _row(
        source,
        proposed="Instruments/Bass/Synth Bass/Loops",
        approved="FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX",
    )
    session = _session(tmp_path, [row])

    summary = TrainingCorrectionImporter(project_root=tmp_path).import_session(session)

    correct_slot = (
        tmp_path
        / "training"
        / "locked_curated_v1"
        / "FX"
        / "Structural and Transitional FX"
        / "Risers and Builds"
        / "Generic Riser"
        / "_LONG_FX"
    )
    assert summary.staged_count == 1
    assert not wrong_teacher.exists()
    assert (correct_slot / "corrected.wav").exists()
    archived = list((summary.report_dir / "superseded_training_conflicts").rglob("wrong_teacher.wav"))
    assert len(archived) == 1


def test_load_available_labels_adds_review_options_and_sorts_brain_labels(tmp_path: Path) -> None:
    brain = tmp_path / "brain.json"
    brain.write_text(
        json.dumps({"labels": ["Instruments/Synths/Synth Loops", "Drums/Kick Drums/Generic Kick/One Shots"]}),
        encoding="utf-8",
    )

    labels = load_available_labels(brain)

    assert labels[0] == "_TO_REVIEW/Needs Human Review"
    assert "Drums/Kick Drums/Generic Kick/One Shots" in labels
    assert "Instruments/Synths/Synth Loops" in labels


def test_load_available_labels_canonicalizes_contradictory_structure_labels(tmp_path: Path) -> None:
    brain = tmp_path / "brain.json"
    brain.write_text(
        json.dumps(
            {
                "labels": [
                    "Instruments/Guitar/Guitar Loops/One Shots",
                    "Instruments/Voice/Vocal One Shots/Loops",
                ]
            }
        ),
        encoding="utf-8",
    )

    labels = load_available_labels(brain)

    assert "Instruments/Guitar/Guitar Loops/Loops" in labels
    assert "Instruments/Guitar/Guitar Loops/One Shots" not in labels
    assert "Instruments/Voice/Vocal One Shots/One Shots" in labels
    assert "Instruments/Voice/Vocal One Shots/Loops" not in labels


def test_load_available_labels_merges_future_catalog_and_training_slots(tmp_path: Path) -> None:
    brain = tmp_path / "brain.json"
    brain.write_text(json.dumps({"labels": ["Instruments/Synths/Synth Loops"]}), encoding="utf-8")
    catalog = tmp_path / "config" / "gui_taxonomy_catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        json.dumps({"labels": ["Instruments/Plucked Strings/Koto/One Shots"]}),
        encoding="utf-8",
    )
    training_slot = tmp_path / "training" / "locked_curated_v1" / "Instruments" / "Plucked Strings" / "Koto" / "_LOOPS"
    training_slot.mkdir(parents=True)

    labels = load_available_labels(
        brain,
        project_root=tmp_path,
        taxonomy_catalog_path=Path("config/gui_taxonomy_catalog.json"),
    )

    assert labels[0] == "_TO_REVIEW/Needs Human Review"
    assert "Instruments/Synths/Synth Loops" in labels
    assert "Instruments/Plucked Strings/Koto/One Shots" in labels
    assert "Instruments/Plucked Strings/Koto/Loops" in labels


def test_preview_empty_audio_folder_writes_clear_error_report(tmp_path: Path) -> None:
    brain = tmp_path / "brain.json"
    brain.write_text(json.dumps({"labels": []}), encoding="utf-8")
    empty_input = tmp_path / "empty_sorted_shell"
    empty_input.mkdir()
    service = SortPreviewService(project_root=tmp_path, brain_config_path=tmp_path / "missing_gui_brains.yaml")

    try:
        service.classify_input(empty_input, brain_path=brain)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("empty audio folder should fail preview")

    run_dirs = sorted((tmp_path / "_reports" / "gui_preview").glob("run_*"))
    assert run_dirs
    error_report = run_dirs[-1] / "Aaron_GUI_Preview_Error.txt"
    assert "No audio files found under folder input" in message
    assert "supported audio" in message
    assert error_report.exists()
    assert "Traceback:" in error_report.read_text(encoding="utf-8")


def test_gui_brain_config_loads_full_and_baby_brain_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "gui_brains.yaml"
    config_path.write_text(
        "\n".join(
            [
                "brains:",
                "  full: brains/full.json",
                "  baby:",
                "    legacy: brains/legacy.json",
                "    core: brains/core.json",
                "    spread: brains/spread.json",
                "    outlier: brains/outlier.json",
                "  harmonic_baby:",
                "    core: brains/harmonic_core.json",
                "    spread: brains/harmonic_spread.json",
                "    outlier: brains/harmonic_outlier.json",
                "options:",
                "  use_baby_brains_in_sort: true",
                "  use_harmonic_brains_in_sort: false",
            ]
        ),
        encoding="utf-8",
    )
    service = SortPreviewService(project_root=tmp_path, brain_config_path=config_path)

    brain_config = service.load_brain_family_config()

    assert brain_config.full_brain_path == tmp_path / "brains" / "full.json"
    assert brain_config.core_baby_brain_path == tmp_path / "brains" / "core.json"
    assert brain_config.user_memory_brain_path == tmp_path / "stage4_folder_brain_user_memory.json"
    assert brain_config.voter_memory_brain_path == tmp_path / "stage4_voter_memory_brain.json"
    assert brain_config.physics_memory_brain_path == tmp_path / "stage4_physics_memory_brain.json"
    assert brain_config.shape_starter_memory_brain_path == tmp_path / "stage4_shape_memory_starter_brain.json"
    assert brain_config.shape_memory_brain_path == tmp_path / "stage4_shape_memory_brain.json"
    assert brain_config.harmonic_outlier_baby_brain_path == tmp_path / "brains" / "harmonic_outlier.json"
    assert brain_config.use_baby_brains_in_sort is True
    assert brain_config.use_harmonic_brains_in_sort is False


def test_simple_yaml_loader_reads_nested_boolean_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "brains:\n  full: full.json\noptions:\n  use_baby_brains_in_sort: false\n",
        encoding="utf-8",
    )

    loaded = load_simple_yaml_mapping(config_path)

    assert loaded["brains"]["full"] == "full.json"
    assert loaded["options"]["use_baby_brains_in_sort"] is False


def test_gui_worker_count_uses_faster_desktop_default(monkeypatch) -> None:
    monkeypatch.delenv("AARON_GUI_SORT_WORKERS", raising=False)
    monkeypatch.setattr("aaron_sound_sorter.gui.preview_service.os.cpu_count", lambda: 12)

    assert gui_worker_count() == 10


def test_gui_worker_count_allows_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("AARON_GUI_SORT_WORKERS", "3")

    assert gui_worker_count() == 3
