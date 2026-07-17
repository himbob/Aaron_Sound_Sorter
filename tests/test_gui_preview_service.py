from __future__ import annotations

import json
from pathlib import Path

from aaron_sound_sorter.gui.models import PreviewRow, SortPreviewSession
from aaron_sound_sorter.gui.preview_service import (
    SortPlanExporter,
    SortPreviewService,
    TrainingCorrectionImporter,
    correction_evidence,
    load_available_labels,
    load_simple_yaml_mapping,
    training_slot_path,
    write_corrections_csv,
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


def test_training_correction_importer_skips_duplicate_audio_in_same_slot(tmp_path: Path) -> None:
    source = _audio_file(tmp_path, "koto.wav")
    existing_slot = tmp_path / "training" / "locked_curated_v1" / "Instruments" / "Plucked Strings" / "Koto" / "_LOOPS"
    existing_slot.mkdir(parents=True)
    existing_copy = existing_slot / "already_here.wav"
    existing_copy.write_bytes(source.read_bytes())
    row = _row(source, approved="Instruments/Plucked Strings/Koto/Loops")
    session = _session(tmp_path, [row])

    summary = TrainingCorrectionImporter(project_root=tmp_path).import_session(session)

    assert summary.staged_count == 0
    assert summary.skipped_count == 1
    assert summary.errors == ["00001: koto.wav: matching audio is already staged in this training slot"]
    assert not (existing_slot / "koto.wav").exists()


def test_training_correction_importer_skips_untrainable_review_folder(tmp_path: Path) -> None:
    source = _audio_file(tmp_path, "review.wav")
    row = _row(source, approved="_TO_REVIEW/Needs Human Review")
    session = _session(tmp_path, [row])

    summary = TrainingCorrectionImporter(project_root=tmp_path).import_session(session)

    assert summary.staged_count == 0
    assert summary.skipped_count == 1
    assert "not a valid training label" in summary.errors[0] or "review folders are not trainable" in summary.errors[0]


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
