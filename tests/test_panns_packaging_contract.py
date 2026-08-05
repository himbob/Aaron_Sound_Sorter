from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_panns_runtime_is_declared_for_end_user_bundles() -> None:
    manifest = (PROJECT_ROOT / "config/product_bundle_assets.txt").read_text(encoding="utf-8")

    assert "model-dir _models/panns_cnn14" in manifest.splitlines()


def test_panns_runtime_has_a_local_verified_download_command() -> None:
    command = PROJECT_ROOT / "commands/neural/PREFETCH_PANNS_MODEL.command"

    assert command.is_file()
    assert command.stat().st_mode & 0o111


def test_panns_runtime_is_local_only_in_git() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "_models/" in gitignore.splitlines()


def test_panns_checkpoint_is_not_configured_for_git_lfs() -> None:
    attributes = (PROJECT_ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "_models/panns_cnn14" not in attributes
