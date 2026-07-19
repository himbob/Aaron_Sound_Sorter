from __future__ import annotations

from pathlib import Path

from aaron_audio_intelligence.physics_memory_brain import PHYSICS_MEMORY_BRAIN_NAME
from aaron_audio_intelligence.shape_memory_brain import SHAPE_MEMORY_BRAIN_NAME, SHAPE_STARTER_MEMORY_BRAIN_NAME
from aaron_audio_intelligence.user_memory_brain import USER_MEMORY_BRAIN_NAME
from aaron_audio_intelligence.voter_memory_brain import VOTER_MEMORY_BRAIN_NAME
from aaron_sound_sorter.gui.models import PreviewRow, SortPreviewSession
from aaron_sound_sorter.gui.web_app import (
    BrainTrainingJob,
    ExportJob,
    PreviewJob,
    add_preview_job_row,
    apply_overrides,
    audio_content_type,
    backup_active_brain_family,
    browser_preview_audio_path,
    build_brain_family_training_command,
    chooser_default_location,
    create_export_job,
    export_job_to_payload,
    parse_range_header,
    preview_job_to_payload,
    render_index_html,
    session_to_payload,
    training_job_to_payload,
    update_preview_job_progress,
    write_audio_range_to_stream,
)


class DisconnectingStream:
    """Binary stream test double that simulates a browser closing playback."""

    def write(self, _chunk: bytes) -> int:
        """Raise the same error macOS reports for cancelled browser audio."""
        raise BrokenPipeError("browser cancelled range request")


def test_web_gui_shell_contains_core_controls() -> None:
    html = render_index_html(
        brain_config_path="/tmp/config/gui_brains.yaml",
        brain_summary="full=stage4_folder_brain.json; baby=3 lanes (on)",
    )

    assert "Aaron Sound Sorter" in html
    assert "Preview Sort" in html
    assert "Export Approved Sort" in html
    assert "audioPlayer" in html
    assert "Listen" in html
    assert 'preload="metadata"' in html
    assert "waitForAudioReady" in html
    assert "categoryHints" in html
    assert "Detected category options" in html
    assert "categoryModal" in html
    assert "Choose Approved Folder" in html
    assert "Apply Approved Folder" in html
    assert "progressBar" in html
    assert "queueTopScroll" in html
    assert "queueTopScrollSpacer" in html
    assert "queue-scrollbar-top" in html
    assert "Scroll from the top or bottom" in html
    assert "syncQueueScrollbars" in html
    assert "queueScrollSlider" not in html
    assert "Use the local slider" not in html
    assert "detailsResizeHandle" in html
    assert "detailsRevealButton" in html
    assert "pane-resizer" in html
    assert "initDetailsResizer" in html
    assert "--details-pane-width" in html
    assert "stickyActionDock" in html
    assert "quickExportButton" in html
    assert "quickTrainCorrectionsButton" in html
    assert "quickOpenTrainingReportButton" in html
    assert "already in training" in html
    assert "setAudioSourceForRow" in html
    assert "audioIsActivelyPlaying" in html
    assert "currentRowKey === nextRowKey" in html
    assert "replaceChildren(fragment)" in html
    assert "rowsEl.addEventListener" in html
    assert "/api/job-audio/" in html
    assert "/api/export-job/" in html
    assert "MAX_JOB_POLL_FAILURES" in html
    assert "current=" in html
    assert "Selected exact folder:" in html
    assert 'aria-label="Collapse all panels"' in html
    assert 'aria-label="Expand all panels"' in html
    assert "panel-toggle" in html
    assert "collapse-icon" in html
    assert "data-collapsible-header" in html
    assert 'data-collapse-target="details"' in html
    assert "correctionNotice" in html
    assert "Open Sorted Folder" in html
    assert "Train Brains From Corrections" in html
    assert "Open Training Report" in html
    assert "Commit Training Evidence" not in html
    assert "/tmp/config/gui_brains.yaml" in html
    assert "Brain JSON" not in html


def test_chooser_default_location_uses_existing_path_context(tmp_path: Path) -> None:
    folder = tmp_path / "sample pack"
    folder.mkdir()
    audio_file = folder / "sound.wav"
    audio_file.write_bytes(b"fake")

    assert chooser_default_location("folder", str(folder)) == str(folder.resolve())
    assert chooser_default_location("folder", str(audio_file)) == str(folder.resolve())
    assert chooser_default_location("file", str(audio_file)) == str(folder.resolve())
    assert chooser_default_location("folder", str(tmp_path / "missing")) == ""


def test_session_payload_preserves_preview_rows() -> None:
    session = SortPreviewSession(
        run_dir=Path("/tmp/run"),
        input_path=Path("/tmp/source"),
        brain_path=Path("/tmp/brain.json"),
        available_labels=["Drums/Kick Drums/Generic Kick/One Shots"],
        rows=[
            PreviewRow(
                row_id="row-1",
                source_path=Path("/tmp/source/kick.wav"),
                display_name="kick.wav",
                proposed_folder="Drums/Kick Drums/Generic Kick/One Shots",
                approved_folder="Drums/Kick Drums/Generic Kick/One Shots",
                final_top="Drums",
                consensus_status="strong_consensus",
                confidence=0.9,
                duration_sec=0.5,
                read_status="ok",
                decision_reason="test",
                diagnostic_summary="shape=drum_hit",
                candidate_folders=[
                    "Drums/Kick Drums/Generic Kick/One Shots",
                    "FX/Impacts and Hits/Boom/One Shots",
                ],
            )
        ],
    )

    payload = session_to_payload(session)

    assert payload["run_dir"] == "/tmp/run"
    assert payload["available_labels"] == ["Drums/Kick Drums/Generic Kick/One Shots"]
    assert payload["rows"][0]["display_name"] == "kick.wav"
    assert payload["rows"][0]["approved_folder"] == "Drums/Kick Drums/Generic Kick/One Shots"
    assert payload["rows"][0]["index"] == 0
    assert payload["rows"][0]["candidate_folders"] == [
        "Drums/Kick Drums/Generic Kick/One Shots",
        "FX/Impacts and Hits/Boom/One Shots",
    ]


def test_apply_overrides_updates_approved_folder_only() -> None:
    row = PreviewRow(
        row_id="row-1",
        source_path=Path("/tmp/source/kick.wav"),
        display_name="kick.wav",
        proposed_folder="Drums/Kick Drums/Generic Kick/One Shots",
        approved_folder="Drums/Kick Drums/Generic Kick/One Shots",
        final_top="Drums",
        consensus_status="strong_consensus",
        confidence=0.9,
        duration_sec=0.5,
        read_status="ok",
        decision_reason="test",
        diagnostic_summary="shape=drum_hit",
    )
    session = SortPreviewSession(
        run_dir=Path("/tmp/run"),
        input_path=Path("/tmp/source"),
        brain_path=Path("/tmp/brain.json"),
        available_labels=[],
        rows=[row],
    )

    apply_overrides(session, [{"index": 0, "approved_folder": "Drums/Kick Drums/808 Kick/One Shots"}])

    assert row.proposed_folder == "Drums/Kick Drums/Generic Kick/One Shots"
    assert row.approved_folder == "Drums/Kick Drums/808 Kick/One Shots"
    assert row.is_corrected


def test_brain_family_training_command_rebuilds_all_gui_brains(tmp_path: Path) -> None:
    command = build_brain_family_training_command(tmp_path, tmp_path / "training" / "locked_curated_v1")

    assert command[1] == str(tmp_path / "Aaron_Sound_Sorter.py")
    assert "train-brain-family" in command
    assert str(tmp_path / "training" / "locked_curated_v1") in command
    assert str(tmp_path / "stage4_folder_brain.json") in command
    assert str(tmp_path / "stage4_folder_brain_core_baby.json") in command
    assert str(tmp_path / "stage4_folder_brain_spread_baby.json") in command
    assert str(tmp_path / "stage4_folder_brain_outlier_baby.json") in command


def test_backup_active_brain_family_copies_existing_gui_brains(tmp_path: Path) -> None:
    full_brain = tmp_path / "stage4_folder_brain.json"
    core_brain = tmp_path / "stage4_folder_brain_core_baby.json"
    memory_brain = tmp_path / USER_MEMORY_BRAIN_NAME
    voter_memory_brain = tmp_path / VOTER_MEMORY_BRAIN_NAME
    physics_memory_brain = tmp_path / PHYSICS_MEMORY_BRAIN_NAME
    shape_memory_brain = tmp_path / SHAPE_MEMORY_BRAIN_NAME
    shape_starter_memory_brain = tmp_path / SHAPE_STARTER_MEMORY_BRAIN_NAME
    full_brain.write_text('{"labels": ["full"]}', encoding="utf-8")
    core_brain.write_text('{"labels": ["core"]}', encoding="utf-8")
    memory_brain.write_text('{"labels": ["memory"]}', encoding="utf-8")
    voter_memory_brain.write_text('{"labels": ["voter"]}', encoding="utf-8")
    physics_memory_brain.write_text('{"labels": ["physics"]}', encoding="utf-8")
    shape_memory_brain.write_text('{"labels": ["shape"]}', encoding="utf-8")
    shape_starter_memory_brain.write_text('{"labels": ["starter"]}', encoding="utf-8")

    backup_dir = backup_active_brain_family(tmp_path, tmp_path / "_reports" / "run" / "brain_backups")

    assert (backup_dir / "stage4_folder_brain.json").read_text(encoding="utf-8") == '{"labels": ["full"]}'
    assert (backup_dir / "stage4_folder_brain_core_baby.json").read_text(encoding="utf-8") == '{"labels": ["core"]}'
    assert (backup_dir / USER_MEMORY_BRAIN_NAME).read_text(encoding="utf-8") == '{"labels": ["memory"]}'
    assert (backup_dir / VOTER_MEMORY_BRAIN_NAME).read_text(encoding="utf-8") == '{"labels": ["voter"]}'
    assert (backup_dir / PHYSICS_MEMORY_BRAIN_NAME).read_text(encoding="utf-8") == '{"labels": ["physics"]}'
    assert (backup_dir / SHAPE_STARTER_MEMORY_BRAIN_NAME).read_text(encoding="utf-8") == '{"labels": ["starter"]}'
    assert (backup_dir / SHAPE_MEMORY_BRAIN_NAME).read_text(encoding="utf-8") == '{"labels": ["shape"]}'
    assert not (backup_dir / "stage4_folder_brain_spread_baby.json").exists()


def test_audio_content_type_prefers_browser_friendly_audio_types() -> None:
    assert audio_content_type(Path("/tmp/sample.wav")) == "audio/wav"
    assert audio_content_type(Path("/tmp/sample.aiff")) == "audio/aiff"
    assert audio_content_type(Path("/tmp/sample.flac")) == "audio/flac"


def test_browser_preview_audio_path_converts_aiff_to_cached_wav(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "Dry Low Body Hit.aif"
    source.write_bytes(b"fake aiff bytes")
    run_dir = tmp_path / "run"
    written: list[tuple[Path, Path]] = []

    def fake_write_wav_preview(source_path: Path, target_path: Path) -> None:
        written.append((source_path, target_path))
        target_path.write_bytes(b"RIFF fake WAVE")

    monkeypatch.setattr("aaron_sound_sorter.gui.web_app.write_wav_preview_from_audio", fake_write_wav_preview)

    preview_path = browser_preview_audio_path(source, run_dir)

    assert preview_path.suffix == ".wav"
    assert preview_path.parent == run_dir / "audio_preview_cache"
    assert preview_path.exists()
    assert written == [(source.resolve(), preview_path)]
    assert browser_preview_audio_path(source, run_dir) == preview_path
    assert len(written) == 1


def test_browser_preview_audio_path_keeps_browser_native_audio(tmp_path: Path) -> None:
    source = tmp_path / "sample.wav"
    source.write_bytes(b"fake wav bytes")

    assert browser_preview_audio_path(source, tmp_path / "run") == source.resolve()


def test_parse_range_header_supports_browser_audio_ranges() -> None:
    assert parse_range_header(None, 1000) == (0, 999, False)
    assert parse_range_header("bytes=100-199", 1000) == (100, 199, True)
    assert parse_range_header("bytes=900-", 1000) == (900, 999, True)
    assert parse_range_header("bytes=-200", 1000) == (800, 999, True)
    assert parse_range_header("bytes=1000-1001", 1000) is None


def test_audio_stream_treats_browser_disconnect_as_normal(tmp_path: Path) -> None:
    source = tmp_path / "sample.wav"
    source.write_bytes(b"0123456789")

    completed = write_audio_range_to_stream(
        source,
        DisconnectingStream(),
        start_byte=0,
        end_byte=9,
    )

    assert not completed


def test_preview_job_progress_tracks_completed_and_total_files() -> None:
    job = PreviewJob(job_id="job-1")

    update_preview_job_progress(job, 3, 10, "snare.wav")

    assert job.completed_files == 3
    assert job.total_files == 10
    assert job.latest_file == "snare.wav"
    assert job.message == "Classified 3 of 10: snare.wav"
    assert job.updated_at >= job.started_at


def test_preview_job_payload_streams_completed_rows_in_final_order() -> None:
    job = PreviewJob(job_id="job-1")
    first_row = PreviewRow(
        row_id="00001",
        source_path=Path("/tmp/source/kick.wav"),
        display_name="kick.wav",
        proposed_folder="Drums/Kick Drums/Generic Kick/One Shots",
        approved_folder="Drums/Kick Drums/Generic Kick/One Shots",
        final_top="Drums",
        consensus_status="strong_consensus",
        confidence=0.91,
        duration_sec=0.4,
        read_status="ok",
        decision_reason="test",
        diagnostic_summary="shape=drum_hit",
    )
    second_row = PreviewRow(
        row_id="00002",
        source_path=Path("/tmp/source/snare.wav"),
        display_name="snare.wav",
        proposed_folder="Drums/Snares/Generic Snare/One Shots",
        approved_folder="Drums/Snares/Generic Snare/One Shots",
        final_top="Drums",
        consensus_status="strong_consensus",
        confidence=0.88,
        duration_sec=0.5,
        read_status="ok",
        decision_reason="test",
        diagnostic_summary="shape=drum_hit",
    )

    add_preview_job_row(job, second_row)
    add_preview_job_row(job, first_row)
    payload = preview_job_to_payload(job)

    assert [row["display_name"] for row in payload["partial_rows"]] == ["kick.wav", "snare.wav"]
    assert [row["index"] for row in payload["partial_rows"]] == [0, 1]
    assert payload["updated_at"] >= payload["started_at"]


def test_create_export_job_starts_as_pollable_background_job() -> None:
    job = create_export_job()

    assert job.job_id
    assert job.status == "queued"
    assert job.message == "Export queued..."
    assert job.updated_at >= job.started_at


def test_export_job_payload_preserves_long_running_export_status() -> None:
    job = ExportJob(
        job_id="export-1",
        status="done",
        message="Exported 2 files.",
        exported_count=2,
        corrected_count=1,
        sorted_root="/tmp/sorted",
        approved_plan_path="/tmp/plan.csv",
        corrections_path="/tmp/corrections.csv",
        errors=["warning"],
    )

    payload = export_job_to_payload(job)

    assert payload["job_id"] == "export-1"
    assert payload["status"] == "done"
    assert payload["exported_count"] == 2
    assert payload["corrected_count"] == 1
    assert payload["sorted_root"] == "/tmp/sorted"
    assert payload["errors"] == ["warning"]
    assert payload["updated_at"] >= payload["started_at"]


def test_training_job_payload_reports_reused_existing_corrections() -> None:
    job = BrainTrainingJob(
        job_id="train-1",
        status="done",
        message="Updated brains.",
        corrected_count=12,
        staged_count=6,
        reused_existing_count=6,
        trainable_count=12,
        skipped_count=0,
    )

    payload = training_job_to_payload(job)

    assert payload["corrected_count"] == 12
    assert payload["staged_count"] == 6
    assert payload["reused_existing_count"] == 6
    assert payload["trainable_count"] == 12
    assert payload["skipped_count"] == 0
