from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from aaron_sound_sorter.gui.calibration_feedback import record_approved_session_feedback
from aaron_sound_sorter.gui.learning_center import LearningCenterService, category_readiness
from aaron_sound_sorter.gui.learning_center_page import render_learning_center_html
from aaron_sound_sorter.gui.models import PreviewRow, SortPreviewSession
from aaron_sound_sorter.neural_audio.calibration import ReviewFeedbackStore
from aaron_sound_sorter.neural_audio.hashing import sha256_file

SYNTH_LABEL = "Instruments/Synths/Synth Pad/Loops"


def _cluster_pack(tmp_path: Path, member_count: int = 2) -> tuple[Path, list[str]]:
    pack_root = tmp_path / "review_pack"
    member_root = pack_root / "Cluster_0001" / "members"
    member_root.mkdir(parents=True)
    member_paths: dict[str, str] = {}
    hashes: list[str] = []
    for index in range(member_count):
        audio_path = member_root / f"audio_{index}.wav"
        waveform = np.sin(np.linspace(0, 30 + index, 8000)).astype(np.float32)
        sf.write(audio_path, waveform, 16_000, subtype="FLOAT")
        file_hash = sha256_file(audio_path)
        hashes.append(file_hash)
        member_paths[file_hash] = str(audio_path.relative_to(pack_root))
    cluster = {
        "cluster_id": "Cluster_0001",
        "member_count": member_count,
        "safe_core_count": 1,
        "boundary_count": member_count - 1,
        "outlier_count": 0,
        "structure_bucket": "Loops",
        "broad_measured_family": "Tonal Instrument",
        "center_hash": hashes[0],
        "typical_hashes": [],
        "boundary_hash": hashes[1] if member_count > 1 else "",
        "outlier_hash": "",
        "member_hashes": hashes,
        "safe_core_hashes": hashes[:1],
        "boundary_hashes": hashes[1:],
        "outlier_hashes": [],
        "member_paths_by_hash": member_paths,
        "suggested_parent_category": "Instruments/Synths",
        "suggested_detailed_categories": [SYNTH_LABEL],
        "mean_similarity": 0.91,
        "radius_p95": 0.07,
        "novelty_score": 0.12,
        "conflict_score": 0.0,
    }
    (pack_root / "cluster_manifest.json").write_text(
        json.dumps({"schema_version": 1, "clusters": [cluster]}),
        encoding="utf-8",
    )
    return pack_root, hashes


def test_learning_center_page_keeps_primary_workflow_simple() -> None:
    html = render_learning_center_html()

    assert "Listen, choose, approve" in html
    assert "Approve Safe Core" in html
    assert "Add Approved Audio to Training Inbox" in html
    assert "Approval and training import are deliberately separate" in html
    assert "More review choices and technical evidence" in html
    assert "Training Coverage" in html
    assert "Confidence Learning" in html
    assert "Build Updated Neural Brain" in html
    assert "/api/learning/rebuild-prototypes" in html


def test_prototype_rebuild_is_explicit_and_reports_human_status(tmp_path: Path, monkeypatch) -> None:
    def fake_rebuild(_project_root: Path, report_dir: Path) -> dict[str, object]:
        report_dir.mkdir(parents=True, exist_ok=True)
        return {
            "status": "built",
            "index_path": "neural/index",
            "training_example_count": 12,
            "label_count": 4,
        }

    monkeypatch.setattr("aaron_sound_sorter.gui.learning_center.run_configured_neural_rebuild", fake_rebuild)
    service = LearningCenterService(tmp_path)

    started = service.start_prototype_rebuild()
    deadline = time.monotonic() + 2.0
    current = started
    while current["status"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.01)
        current = service.prototype_rebuild_status(str(started["job_id"]))

    assert current["status"] == "done"
    assert current["training_example_count"] == 12
    assert "12 approved sounds" in str(current["message"])


def test_learning_center_saves_then_separately_imports_safe_core(tmp_path: Path) -> None:
    pack_root, hashes = _cluster_pack(tmp_path)
    service = LearningCenterService(tmp_path)
    loaded = service.open_cluster_pack(str(pack_root))

    decision = service.save_cluster_decision(
        pack_id=str(loaded["pack_id"]),
        cluster_id="Cluster_0001",
        action="approve_safe_core",
        approved_category=SYNTH_LABEL,
        reviewed_by="Aaron",
    )

    decision_payload = json.loads(Path(str(decision["decision_path"])).read_text(encoding="utf-8"))
    assert decision["training_ready"] is True
    assert decision_payload["approved_safe_core_hashes"] == hashes[:1]
    assert decision_payload["training_imported"] is False

    imported = service.import_saved_decision(str(decision["decision_path"]))

    assert imported["imported_count"] == 1
    updated = json.loads(Path(str(decision["decision_path"])).read_text(encoding="utf-8"))
    assert updated["training_imported"] is True
    assert Path(str(updated["rollback_record"])).is_file()


def test_explicit_approve_all_can_import_noncore_members(tmp_path: Path) -> None:
    pack_root, hashes = _cluster_pack(tmp_path)
    service = LearningCenterService(tmp_path)
    loaded = service.open_cluster_pack(str(pack_root))

    decision = service.save_cluster_decision(
        pack_id=str(loaded["pack_id"]),
        cluster_id="Cluster_0001",
        action="approve_all",
        approved_category=SYNTH_LABEL,
        reviewed_by="Aaron",
    )
    imported = service.import_saved_decision(str(decision["decision_path"]))

    assert imported["imported_count"] == len(hashes)


def test_category_readiness_never_hides_empty_categories() -> None:
    assert category_readiness(example_count=0, prototype_count=0, heldout_count=0, heldout_top1=None)[0] == "D"
    assert category_readiness(example_count=6, prototype_count=2, heldout_count=0, heldout_top1=None)[0] == "B"
    assert category_readiness(example_count=10, prototype_count=2, heldout_count=4, heldout_top1=0.9)[0] == "A"


def test_export_approval_feedback_includes_accepted_and_corrected_rows(tmp_path: Path) -> None:
    rows: list[PreviewRow] = []
    for index, approved_label in enumerate((SYNTH_LABEL, "Instruments/Keys/Piano/Loops")):
        audio_path = tmp_path / f"audio_{index}.wav"
        sf.write(audio_path, np.sin(np.linspace(0, 20 + index, 4000)), 16_000)
        rows.append(
            PreviewRow(
                row_id=f"row-{index}",
                source_path=audio_path,
                display_name=audio_path.name,
                proposed_folder=SYNTH_LABEL,
                approved_folder=approved_label,
                final_top="Instruments",
                consensus_status="neural_production_owner",
                confidence=0.8,
                duration_sec=0.25,
                read_status="ok",
                decision_reason="test",
                diagnostic_summary="test",
                neural_folder=SYNTH_LABEL,
                neural_known_distribution=True,
                neural_similarity=0.82,
                neural_margin=0.11,
                neural_radius_ratio=0.7,
                neural_label_example_count=8,
                neural_semantic_family="synth",
            )
        )
    session = SortPreviewSession(tmp_path / "run", tmp_path, tmp_path / "brain.json", [], rows)

    assert record_approved_session_feedback(tmp_path, session) == 2

    stored = ReviewFeedbackStore(tmp_path / "neural_artifacts/calibration_feedback/review_feedback.jsonl").read_all()
    assert [row.accepted for row in stored] == [True, False]
    assert stored[1].parent_family_accepted is True
