from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aaron_sound_sorter.neural_audio.cluster_approval import (
    import_cluster_approval,
    rollback_cluster_import,
)
from aaron_sound_sorter.neural_audio.hashing import sha256_file

SYNTH_LABEL = "Instruments/Synths/Synth Pad/Loops"
VOICE_LABEL = "Instruments/Voice/Vocal Loops/Loops"


def _approval_files(tmp_path: Path, approved_category: str) -> tuple[Path, str]:
    pack_root = tmp_path / "pack"
    member_dir = pack_root / "Cluster_0001/members"
    member_dir.mkdir(parents=True, exist_ok=True)
    audio_path = member_dir / "hash-only.wav"
    sf.write(audio_path, np.sin(np.linspace(0, 40, 16000)).astype(np.float32), 16000)
    file_sha256 = sha256_file(audio_path)
    manifest_path = pack_root / "cluster_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clusters": [
                    {
                        "cluster_id": "Cluster_0001",
                        "safe_core_hashes": [file_sha256],
                        "member_paths_by_hash": {
                            file_sha256: str(audio_path.relative_to(pack_root)),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    approval_path = tmp_path / f"approval_{approved_category.split('/')[1]}.json"
    approval_path.write_text(
        json.dumps(
            {
                "cluster_id": "Cluster_0001",
                "cluster_manifest_path": str(manifest_path),
                "approved_category": approved_category,
                "approved_safe_core_hashes": [file_sha256],
                "rejected_hashes": [],
                "boundary_review_required": [],
                "reviewed_by": "test-reviewer",
                "review_version": 1,
            }
        ),
        encoding="utf-8",
    )
    return approval_path, file_sha256


def test_cluster_safe_core_import_is_explicit_and_reversible(tmp_path: Path) -> None:
    approval_path, file_sha256 = _approval_files(tmp_path, SYNTH_LABEL)
    inbox_root = tmp_path / "inbox"

    summary = import_cluster_approval(
        tmp_path,
        approval_path,
        tmp_path / "report",
        inbox_root=inbox_root,
    )

    rows = list(csv.DictReader(summary.intake_summary.current_manifest_path.open(encoding="utf-8")))
    assert summary.imported_count == 1
    assert rows[0]["file_sha256"] == file_sha256
    assert rows[0]["approved_folder"] == SYNTH_LABEL
    assert summary.copied_audio_paths[0].name.startswith(f"audio_{file_sha256}")
    assert json.loads((tmp_path / "report/import_summary.json").read_text())["prototype_rebuild_performed"] is False

    rollback_cluster_import(summary.rollback_record_path)

    assert not summary.intake_summary.current_manifest_path.exists()
    assert not summary.intake_summary.event_log_path.exists()
    assert not summary.copied_audio_paths[0].exists()


def test_cluster_import_rejects_existing_hash_under_another_label(tmp_path: Path) -> None:
    approval_path, _file_sha256 = _approval_files(tmp_path, SYNTH_LABEL)
    inbox_root = tmp_path / "inbox"
    import_cluster_approval(tmp_path, approval_path, tmp_path / "first_report", inbox_root=inbox_root)
    conflicting_approval, _same_hash = _approval_files(tmp_path, VOICE_LABEL)

    with pytest.raises(ValueError, match="another label"):
        import_cluster_approval(
            tmp_path,
            conflicting_approval,
            tmp_path / "second_report",
            inbox_root=inbox_root,
        )
