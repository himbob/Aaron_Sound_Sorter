from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_panns_runtime_is_declared_for_end_user_bundles() -> None:
    manifest = (PROJECT_ROOT / "config/product_bundle_assets.txt").read_text(encoding="utf-8")

    assert "model-dir _models/panns_cnn14" in manifest.splitlines()


def test_panns_runtime_is_not_hidden_from_git() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "!_models/panns_cnn14/" in gitignore.splitlines()
    assert "!_models/panns_cnn14/**" in gitignore.splitlines()


def test_panns_checkpoint_uses_git_lfs() -> None:
    attributes = (PROJECT_ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "_models/panns_cnn14/** filter=lfs diff=lfs merge=lfs -text" in attributes.splitlines()
