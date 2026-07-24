import json

from aaron_sound_sorter.neural_audio.legacy_brain_audit import audit_legacy_brain_trainers


def test_legacy_brain_audit_finds_same_audio_under_multiple_labels(tmp_path) -> None:
    audio_path = tmp_path / "source.wav"
    audio_path.write_bytes(b"same audio bytes")
    brain_path = tmp_path / "brain.json"
    brain_path.write_text(
        json.dumps(
            {
                "brain_type": "folder",
                "training_examples_detailed_by_label": {
                    "Instruments/Voice/Vocal Loops/Loops": [
                        {"source_path": str(audio_path), "fingerprint": [1.0, 2.0]}
                    ],
                    "FX/Human and Voice FX/Altered Voice/Long FX": [
                        {"source_path": str(audio_path), "fingerprint": [1.0, 2.0]}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    rows = audit_legacy_brain_trainers([brain_path])

    assert len(rows) == 2
    assert {row.status for row in rows} == {"approved_label_conflict"}
    assert {row.assigned_label_count for row in rows} == {2}


def test_legacy_brain_audit_uses_stored_fingerprint_when_source_is_missing(tmp_path) -> None:
    brain_path = tmp_path / "memory.json"
    brain_path.write_text(
        json.dumps(
            {
                "brain_type": "voter_memory",
                "voter_memory": {
                    "examples_by_role": {
                        "instrument_voice_loop": [
                            {
                                "approved_label": "Instruments/Voice/Vocal Loops/Loops",
                                "fingerprint": [3.0, 4.0],
                                "source_path": str(tmp_path / "missing.wav"),
                                "target_role": "instrument_voice_loop",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    (row,) = audit_legacy_brain_trainers([brain_path])

    assert row.identity_kind == "fingerprint_sha256"
    assert row.status == "missing_source"
