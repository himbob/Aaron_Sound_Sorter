from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tools.build_product_bundle import ProductBundleRequest, build_product_bundle, select_product_files

REQUIRED_BRAINS = (
    "stage4_folder_brain.json",
    "stage4_folder_brain_user_memory.json",
    "stage4_physics_memory_brain.json",
    "stage4_voter_memory_brain.json",
    "stage4_shape_memory_brain.json",
)


def make_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    for directory in ("src", "commands", "config", "docs", "tools", "tests"):
        (project_root / directory).mkdir(parents=True, exist_ok=True)

    (project_root / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project_root / "tests" / "test_app.py").write_text("def test_value():\n    assert True\n", encoding="utf-8")
    (project_root / "tests" / "large_fixture.wav").write_bytes(b"audio")
    (project_root / "_reports" / "gui_preview").mkdir(parents=True)
    (project_root / "_reports" / "gui_preview" / "report.txt").write_text("generated", encoding="utf-8")

    manifest_lines = [*(f"required-file {name}" for name in REQUIRED_BRAINS)]
    manifest_lines.extend(
        (
            "optional-file stage4_folder_brain_core_baby.json",
            "model-dir _models/laion_larger_clap_music_and_speech",
            "model-dir _models/mert_v1_95m_7d1bb4c",
            "model-dir _models/panns_cnn14",
            "artifact-dir neural_artifacts/prototype_indexes",
        )
    )
    (project_root / "config" / "product_bundle_assets.txt").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )

    for brain_name in REQUIRED_BRAINS:
        (project_root / brain_name).write_text('{"live": true}\n', encoding="utf-8")
    (project_root / "stage4_folder_brain_core_baby.json").write_text('{"live": true}\n', encoding="utf-8")
    (project_root / "stage4_folder_brain_backup_2026.json").write_text('{"backup": true}\n', encoding="utf-8")

    clap_dir = project_root / "_models" / "laion_larger_clap_music_and_speech"
    mert_dir = project_root / "_models" / "mert_v1_95m_7d1bb4c"
    panns_dir = project_root / "_models" / "panns_cnn14"
    prototype_dir = project_root / "neural_artifacts" / "prototype_indexes"
    embedding_cache = project_root / "neural_artifacts" / "embedding_cache"
    for directory in (clap_dir, mert_dir, panns_dir, prototype_dir, embedding_cache):
        directory.mkdir(parents=True, exist_ok=True)
    (clap_dir / "model.safetensors").write_bytes(b"clap")
    (mert_dir / "pytorch_model.bin").write_bytes(b"mert")
    (panns_dir / "Cnn14_mAP=0.431.pth").write_bytes(b"panns")
    (panns_dir / "class_labels_indices.csv").write_text("index,mid,display_name\n", encoding="utf-8")
    (prototype_dir / "runtime_index.npz").write_bytes(b"index")
    (embedding_cache / "development_cache.npy").write_bytes(b"cache")
    return project_root


def make_request(
    project_root: Path, *, include_neural_models: bool = True, dry_run: bool = True
) -> ProductBundleRequest:
    return ProductBundleRequest(
        project_root=project_root,
        bundle_name="Aaron_Product_Test",
        output_dir=project_root / "_reports" / "bundles",
        dry_run=dry_run,
        include_neural_models=include_neural_models,
    )


def test_product_selection_includes_live_runtime_assets_and_test_code(tmp_path: Path) -> None:
    project_root = make_project(tmp_path)

    selected = select_product_files(make_request(project_root))
    selected_paths = {item.relative_path.as_posix() for item in selected}

    assert "tests/test_app.py" in selected_paths
    assert "tests/large_fixture.wav" not in selected_paths
    assert "stage4_folder_brain.json" in selected_paths
    assert "stage4_folder_brain_backup_2026.json" not in selected_paths
    assert "_models/laion_larger_clap_music_and_speech/model.safetensors" in selected_paths
    assert "_models/mert_v1_95m_7d1bb4c/pytorch_model.bin" in selected_paths
    assert "_models/panns_cnn14/Cnn14_mAP=0.431.pth" in selected_paths
    assert "_models/panns_cnn14/class_labels_indices.csv" in selected_paths
    assert "neural_artifacts/prototype_indexes/runtime_index.npz" in selected_paths
    assert "neural_artifacts/embedding_cache/development_cache.npy" not in selected_paths
    assert "_reports/gui_preview/report.txt" not in selected_paths


def test_brain_only_mode_keeps_brains_and_indexes_but_omits_model_snapshots(tmp_path: Path) -> None:
    project_root = make_project(tmp_path)

    selected = select_product_files(make_request(project_root, include_neural_models=False))
    selected_paths = {item.relative_path.as_posix() for item in selected}

    assert "stage4_folder_brain.json" in selected_paths
    assert "neural_artifacts/prototype_indexes/runtime_index.npz" in selected_paths
    assert not any(path.startswith("_models/") for path in selected_paths)


def test_missing_required_brain_blocks_broken_product_bundle(tmp_path: Path) -> None:
    project_root = make_project(tmp_path)
    (project_root / "stage4_shape_memory_brain.json").unlink()

    with pytest.raises(SystemExit):
        select_product_files(make_request(project_root))


def test_written_zip_contains_no_generated_or_backup_payload(tmp_path: Path) -> None:
    project_root = make_project(tmp_path)
    result = build_product_bundle(make_request(project_root, dry_run=False))

    with zipfile.ZipFile(result.zip_path) as archive:
        names = set(archive.namelist())

    assert "Aaron_Product_Test/stage4_folder_brain.json" in names
    assert "Aaron_Product_Test/tests/test_app.py" in names
    assert "Aaron_Product_Test/_models/laion_larger_clap_music_and_speech/model.safetensors" in names
    assert "Aaron_Product_Test/_models/panns_cnn14/Cnn14_mAP=0.431.pth" in names
    assert "Aaron_Product_Test/_models/panns_cnn14/class_labels_indices.csv" in names
    assert not any("_reports/gui_preview" in name for name in names)
    assert not any("embedding_cache" in name for name in names)
    assert not any(name.endswith("large_fixture.wav") for name in names)
    assert not any("brain_backup" in name for name in names)
