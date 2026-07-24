from __future__ import annotations

import csv

from aaron_sound_sorter.neural_audio.hashing import sha256_file
from aaron_sound_sorter.neural_audio.legacy_manifest import read_legacy_folder_map_by_hash


def test_legacy_shadow_reader_can_compute_missing_hash_from_existing_source(tmp_path) -> None:
    audio = tmp_path / "arbitrary_name.wav"
    audio.write_bytes(b"not-real-audio-but-stable-bytes")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["original_path", "folder_path"])
        writer.writeheader()
        writer.writerow({"original_path": str(audio), "folder_path": "FX/Legacy Label"})
    result = read_legacy_folder_map_by_hash(manifest)
    assert result == {sha256_file(audio): "FX/Legacy Label"}
