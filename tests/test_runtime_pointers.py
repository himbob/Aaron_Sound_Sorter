from pathlib import Path

from aaron_sound_sorter.commands import (
    portable_project_path,
    resolve_brain_path,
    resolve_project_pointer,
    runtime_pointer_path,
)
from aaron_sound_sorter.preview import write_runtime_pointer


def test_runtime_pointer_round_trip_uses_project_relative_path(tmp_path: Path) -> None:
    brain_path = tmp_path / "brains" / "voice.json"
    brain_path.parent.mkdir()
    brain_path.write_text("{}", encoding="utf-8")

    pointer_path = write_runtime_pointer(
        tmp_path,
        "phase3_latest_pure_brain_path.txt",
        brain_path,
    )

    assert pointer_path == tmp_path / "config/runtime/phase3_latest_pure_brain_path.txt"
    assert pointer_path.read_text(encoding="utf-8") == "brains/voice.json\n"
    assert resolve_project_pointer(tmp_path, "brains/voice.json") == brain_path


def test_resolve_brain_path_reads_runtime_pointer(tmp_path: Path) -> None:
    brain_path = tmp_path / "brains" / "trained.json"
    brain_path.parent.mkdir()
    brain_path.write_text("{}", encoding="utf-8")
    pointer_path = runtime_pointer_path(
        tmp_path,
        "phase4_latest_folder_brain_path.txt",
    )
    pointer_path.write_text("brains/trained.json\n", encoding="utf-8")

    resolved = resolve_brain_path(tmp_path, "missing-default.json")

    assert resolved == brain_path
    assert portable_project_path(tmp_path, brain_path) == "brains/trained.json"
