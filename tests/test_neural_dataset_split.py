from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aaron_sound_sorter.neural_audio.dataset import (
    build_evaluation_split,
    build_explicit_evaluation_split,
    discover_curated_examples,
)
from aaron_sound_sorter.neural_audio.hashing import decoded_audio_sha256, normalized_audio_sha256


def rows(tmp_path: Path, label: str, count: int):
    return [(tmp_path / label / f"sample_{index}.wav", f"{index + 1:064x}") for index in range(count)]


def test_split_is_disjoint_and_keeps_training_examples(tmp_path: Path) -> None:
    split = build_evaluation_split(
        {
            "Instruments/Voice/Vocal Loops": rows(tmp_path, "vocal", 10),
            "Drums/Snares": [(tmp_path / "snare" / f"sample_{index}.wav", f"{100 + index:064x}") for index in range(5)],
        },
        holdout_fraction=0.20,
        seed="test",
    )
    preview_hashes = {row.file_sha256 for row in split.review_preview}
    heldout_hashes = {row.file_sha256 for row in split.heldout_eval}
    assert preview_hashes.isdisjoint(heldout_hashes)
    assert len([row for row in split.heldout_eval if row.label.endswith("Vocal Loops")]) == 2
    assert len([row for row in split.review_preview if row.label.endswith("Vocal Loops")]) == 8


def test_small_label_is_prototype_only(tmp_path: Path) -> None:
    split = build_evaluation_split({"FX/Impacts": rows(tmp_path, "impact", 2)}, seed="test")
    assert len(split.review_preview) == 2
    assert not split.heldout_eval
    assert any("prototype-only" in warning for warning in split.warnings)


def test_split_rejects_cross_partition_hash_overlap() -> None:
    from aaron_sound_sorter.neural_audio.contracts import EvaluationSplit, LabeledAudioExample

    digest = "a" * 64
    example_a = LabeledAudioExample("A", Path("a.wav"), digest, "review_preview")
    example_b = LabeledAudioExample("A", Path("b.wav"), digest, "heldout_eval")
    with pytest.raises(ValueError, match="leakage"):
        EvaluationSplit((example_a,), (example_b,))


def test_explicit_split_preserves_human_selected_partitions(tmp_path: Path) -> None:
    preview = {"Voice": rows(tmp_path, "preview", 3)}
    heldout = {"Voice": [(tmp_path / "heldout" / "sample.wav", "f" * 64)]}

    split = build_explicit_evaluation_split(preview, heldout)

    assert len(split.review_preview) == 3
    assert len(split.heldout_eval) == 1
    assert split.heldout_eval[0].file_sha256 == "f" * 64


def test_explicit_split_rejects_lossless_container_copy(tmp_path: Path) -> None:
    samples = np.asarray([0, 1000, -1000, 2000, -2000], dtype=np.int16)
    preview_path = tmp_path / "preview.wav"
    heldout_path = tmp_path / "heldout.aiff"
    sf.write(preview_path, samples, 48000, subtype="PCM_16")
    sf.write(heldout_path, samples, 48000, subtype="PCM_16")
    preview = discover_curated_examples(_single_label_root(tmp_path, "preview_root", preview_path))
    heldout = discover_curated_examples(_single_label_root(tmp_path, "heldout_root", heldout_path))

    with pytest.raises(ValueError, match="decoded-audio leakage"):
        build_explicit_evaluation_split(preview, heldout)


def test_normalized_audio_hash_groups_gain_and_boundary_silence_copies(tmp_path: Path) -> None:
    sample_rate = 16_000
    time_axis = np.arange(sample_rate // 4, dtype=np.float32) / sample_rate
    tone = np.sin(2.0 * np.pi * 440.0 * time_axis).astype(np.float32)
    original = tmp_path / "original.wav"
    derived = tmp_path / "derived.wav"
    sf.write(original, tone, sample_rate, subtype="FLOAT")
    sf.write(
        derived,
        np.concatenate([np.zeros(32, dtype=np.float32), tone * 0.5, np.zeros(48, dtype=np.float32)]),
        sample_rate,
        subtype="FLOAT",
    )

    assert decoded_audio_sha256(original) != decoded_audio_sha256(derived)
    assert normalized_audio_sha256(original) == normalized_audio_sha256(derived)


def test_explicit_split_rejects_gain_and_boundary_silence_copy(tmp_path: Path) -> None:
    sample_rate = 16_000
    time_axis = np.arange(sample_rate // 4, dtype=np.float32) / sample_rate
    tone = np.sin(2.0 * np.pi * 330.0 * time_axis).astype(np.float32)
    preview_path = tmp_path / "preview_gain.wav"
    heldout_path = tmp_path / "heldout_gain.wav"
    sf.write(preview_path, tone, sample_rate, subtype="FLOAT")
    sf.write(
        heldout_path,
        np.concatenate([np.zeros(40, dtype=np.float32), tone * 0.4, np.zeros(50, dtype=np.float32)]),
        sample_rate,
        subtype="FLOAT",
    )
    preview = discover_curated_examples(_single_label_root(tmp_path, "preview_gain_root", preview_path))
    heldout = discover_curated_examples(_single_label_root(tmp_path, "heldout_gain_root", heldout_path))

    with pytest.raises(ValueError, match="normalized-audio leakage"):
        build_explicit_evaluation_split(preview, heldout)


def _single_label_root(tmp_path: Path, root_name: str, source: Path) -> Path:
    root = tmp_path / root_name
    label_dir = root / "Voice"
    label_dir.mkdir(parents=True)
    destination = label_dir / source.name
    destination.write_bytes(source.read_bytes())
    return root
