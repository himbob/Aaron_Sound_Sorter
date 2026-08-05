from __future__ import annotations

from tools.audit_public_runtime_assets import (
    forbidden_tracked_paths,
    private_value_locations,
    sanitize_value,
    tracked_private_path_locations,
)


def test_public_brain_sanitizer_removes_paths_filenames_and_pack_names() -> None:
    payload = {
        "source_path": "/Volumes/Private Pack/Vocal One.wav",
        "new_source_path": "/Users/person/training/Sax.aif",
        "source_pack": "Private Pack Name",
        "source_pack_counts_top10": {"Private Pack Name": 2},
        "examples_by_label": {"Voice": ["hidden_voice.wav"]},
        "fingerprint": [0.1, 0.2],
    }

    sanitized = sanitize_value(payload)

    assert "source_path" not in sanitized
    assert "new_source_path" not in sanitized
    assert "source_pack" not in sanitized
    assert "source_pack_counts_top10" not in sanitized
    assert sanitized["source_id"].startswith("private-sha256:")
    assert sanitized["new_source_id"].startswith("private-sha256:")
    assert sanitized["source_group_id"].startswith("private-sha256:")
    assert list(sanitized["source_group_counts_top10"])[0].startswith("private-sha256:")
    assert sanitized["fingerprint"] == [0.1, 0.2]
    assert private_value_locations(sanitized) == []


def test_public_audit_rejects_local_runtime_assets() -> None:
    paths = [
        "src/aaron_sound_sorter/core.py",
        "_models/clap/model.safetensors",
        "neural_artifacts/active/index.json",
        "stage4_folder_brain.json",
        "tests/audio/private.wav",
    ]

    assert forbidden_tracked_paths(paths) == sorted(paths[1:])


def test_public_audit_reports_private_machine_paths_without_values(tmp_path) -> None:
    source = tmp_path / "config.py"
    source.write_text('ROOT = "' + "/Volumes" + '/T9/private"\n', encoding="utf-8")

    assert tracked_private_path_locations(tmp_path, ["config.py"]) == ["config.py:1"]
