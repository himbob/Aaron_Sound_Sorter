from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_neural_curation_system import (  # noqa: E402
    contamination_action,
    is_legacy_brain_warning,
    public_label_from_training_path,
    sanitize_persistent_paths,
    write_csv,
)


def test_public_label_from_training_slot_uses_only_explicit_curated_structure(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    audio_path = (
        training_root
        / "FX"
        / "Structural and Transitional FX"
        / "Risers and Builds"
        / "Generic Riser"
        / "_LONG_FX"
        / "display.wav"
    )

    label = public_label_from_training_path(audio_path, training_root)

    assert label == "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX"


def test_contamination_actions_never_delete_training_automatically() -> None:
    assert contamination_action("cross_label_conflict") == "propose_quarantine_pending_human_verdict"
    assert contamination_action("supported") == "candidate_only_until_provenance_review"
    assert contamination_action("not_audited") == "exclude_not_audited"


def test_clean_legacy_brain_rows_are_not_written_as_quarantine_warnings() -> None:
    assert not is_legacy_brain_warning("clean")
    assert not is_legacy_brain_warning("ok")
    assert is_legacy_brain_warning("missing_source")
    assert is_legacy_brain_warning("approved_label_conflict")


def test_persistent_evidence_csv_uses_repository_safe_lf_endings(tmp_path: Path) -> None:
    output_path = tmp_path / "evidence.csv"

    write_csv(output_path, [{"label": "Voice/Musical Vocal", "count": 1}])

    payload = output_path.read_bytes()
    assert b"\r\n" not in payload
    assert payload == b"label,count\nVoice/Musical Vocal,1\n"


def test_persistent_evidence_replaces_private_roots_with_portable_references(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    samples_root = tmp_path / "private-samples"
    output_dir = project_root / "neural_artifacts" / "run"
    output_dir.mkdir(parents=True)
    artifact_path = output_dir / "provenance.csv"
    artifact_path.write_text(
        f"source\n{project_root}/training/example.wav\n{samples_root}/Pack/example.wav\n",
        encoding="utf-8",
    )

    sanitize_persistent_paths(output_dir, project_root, samples_root)

    assert artifact_path.read_text(encoding="utf-8") == (
        "source\nproject://training/example.wav\nsample-library://Pack/example.wav\n"
    )
