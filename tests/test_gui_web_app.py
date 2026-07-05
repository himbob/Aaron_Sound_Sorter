from __future__ import annotations

from pathlib import Path

from aaron_sound_sorter.gui.models import PreviewRow, SortPreviewSession
from aaron_sound_sorter.gui.web_app import (
    PreviewJob,
    apply_overrides,
    audio_content_type,
    backup_active_brain_family,
    build_brain_family_training_command,
    parse_range_header,
    render_index_html,
    session_to_payload,
    update_preview_job_progress,
)


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
    assert "categoryModal" in html
    assert "Choose Approved Folder" in html
    assert "Apply Approved Folder" in html
    assert "progressBar" in html
    assert "correctionNotice" in html
    assert "Open Sorted Folder" in html
    assert "Train Brains From Corrections" in html
    assert "Open Training Report" in html
    assert "Commit Training Evidence" not in html
    assert "/tmp/config/gui_brains.yaml" in html
    assert "Brain JSON" not in html


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
            )
        ],
    )

    payload = session_to_payload(session)

    assert payload["run_dir"] == "/tmp/run"
    assert payload["available_labels"] == ["Drums/Kick Drums/Generic Kick/One Shots"]
    assert payload["rows"][0]["display_name"] == "kick.wav"
    assert payload["rows"][0]["approved_folder"] == "Drums/Kick Drums/Generic Kick/One Shots"
    assert payload["rows"][0]["index"] == 0


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
    full_brain.write_text('{"labels": ["full"]}', encoding="utf-8")
    core_brain.write_text('{"labels": ["core"]}', encoding="utf-8")

    backup_dir = backup_active_brain_family(tmp_path, tmp_path / "_reports" / "run" / "brain_backups")

    assert (backup_dir / "stage4_folder_brain.json").read_text(encoding="utf-8") == '{"labels": ["full"]}'
    assert (backup_dir / "stage4_folder_brain_core_baby.json").read_text(encoding="utf-8") == '{"labels": ["core"]}'
    assert not (backup_dir / "stage4_folder_brain_spread_baby.json").exists()


def test_audio_content_type_prefers_browser_friendly_audio_types() -> None:
    assert audio_content_type(Path("/tmp/sample.wav")) == "audio/wav"
    assert audio_content_type(Path("/tmp/sample.aiff")) == "audio/aiff"
    assert audio_content_type(Path("/tmp/sample.flac")) == "audio/flac"


def test_parse_range_header_supports_browser_audio_ranges() -> None:
    assert parse_range_header(None, 1000) == (0, 999, False)
    assert parse_range_header("bytes=100-199", 1000) == (100, 199, True)
    assert parse_range_header("bytes=900-", 1000) == (900, 999, True)
    assert parse_range_header("bytes=-200", 1000) == (800, 999, True)
    assert parse_range_header("bytes=1000-1001", 1000) is None


def test_preview_job_progress_tracks_completed_and_total_files() -> None:
    job = PreviewJob(job_id="job-1")

    update_preview_job_progress(job, 3, 10, "snare.wav")

    assert job.completed_files == 3
    assert job.total_files == 10
    assert job.latest_file == "snare.wav"
    assert job.message == "Classified 3 of 10: snare.wav"
