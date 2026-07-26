from __future__ import annotations

from pathlib import Path

from aaron_sound_sorter.gui.models import PreviewRow
from aaron_sound_sorter.neural_audio.hashing import sha256_file
from tools.audit_trusted_audio_runtime import comparison_rows, materialize_hash_named_audio


def test_trusted_audit_materializes_hash_only_runtime_names(tmp_path: Path) -> None:
    source = tmp_path / "a_name_that_must_be_locator_only.wav"
    source.write_bytes(b"audio fixture bytes")
    input_dir = tmp_path / "audit_input"
    input_dir.mkdir()

    cases_by_hash = materialize_hash_named_audio(
        tmp_path,
        input_dir,
        [
            {
                "id": "explicit_claim",
                "audio_path": str(source),
                "training_label": "FX/Impacts and Hits/Generic Impact/One Shots",
            }
        ],
    )

    copied = list(input_dir.iterdir())
    assert len(copied) == 1
    assert copied[0].name.startswith("audio_")
    assert "name_that_must" not in copied[0].name
    assert next(iter(cases_by_hash.values()))["training_label"].startswith("FX/")


def test_trusted_audit_records_prompt_accuracy_separately_from_final_sort(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio fixture bytes")
    file_sha256 = sha256_file(audio)
    claimed = "Instruments/Woodwinds/Saxophone/Alto/One Shots"
    row = PreviewRow(
        row_id="00001",
        source_path=audio,
        display_name="audio.wav",
        proposed_folder=claimed,
        approved_folder=claimed,
        final_top="Instruments",
        consensus_status="neural_known_distribution_owner",
        confidence=0.8,
        duration_sec=1.0,
        read_status="ok",
        decision_reason="test",
        diagnostic_summary="test",
        neural_prompt_status="advisory_only",
        neural_prompt_suggestions=[
            {"path": "Instruments/Brass/Trumpet/One Shots"},
            {"path": claimed},
        ],
    )

    report_rows = comparison_rows(
        [row],
        {
            file_sha256: {
                "id": "sax",
                "audio_path": str(audio),
                "source_locator": str(audio),
                "training_label": claimed,
                "file_sha256": file_sha256,
            }
        },
    )

    assert report_rows[0]["prompt_top1_matches_claim"] == "0"
    assert report_rows[0]["prompt_top3_matches_claim"] == "1"
    assert report_rows[0]["matches_claimed_label"] == "1"
