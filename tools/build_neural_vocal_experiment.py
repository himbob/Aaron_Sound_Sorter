#!/usr/bin/env python3
"""Build the leakage-safe broad-vocal CLAP shadow experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.dataset import AUDIO_SUFFIXES
from aaron_sound_sorter.neural_audio.hashing import decoded_audio_sha256, sha256_file

VOICE_LABEL = "Voice/Musical Vocal"

PROTECTED_NEGATIVES = {
    "NonVoice/Drums and Percussion": (
        "tests/regression_audio/2.Drum Loop_1_100bpm.wav",
        "tests/regression_audio/A1_Kick_Clap_Loop_99bpm.wav",
        "tests/regression_audio/ABOUTME_94_DRUMLOOP.wav",
        "tests/regression_audio/DRUMS_LOOP_125BPM.wav",
        "tests/regression_audio/Drumloop_hats_Dark_Rap_140BPM.wav",
        "tests/regression_audio/Full Drum Loop 03 - 78BPM.wav",
        "tests/regression_audio/JL_TDBL_Drum Full_act up_138bpm.wav",
        "tests/regression_audio/QUp_DzU_DrumLp_03_100bpm.wav",
    ),
    "NonVoice/Other Instruments": (
        "tests/regression_audio/Chord 1_G.wav",
        "tests/regression_audio/Compton_Fmin_100bpm.wav",
        "tests/regression_audio/GangstaFunk_Cmaj_96bpm.wav",
        "tests/regression_audio/Piano_G.wav",
    ),
    "NonVoice/Sax and Woodwind": (
        "tests/regression_audio/1.Saxophone_1_110bpm_Am.wav",
        "tests/regression_audio/AA_JBL_74bpm_Am_Sax_Loop_18.wav",
        "tests/regression_audio/AA_JBL_78bpm_Cm_Sax_Loop_1.wav",
        "tests/regression_audio/AA_JBL_86bpm_Cm_Sax_Loop_12.wav",
        "tests/regression_audio/AA_JBL_90bpm_Em_Sax_Loop_3.wav",
        "tests/regression_audio/AMV_VRNB1_102_brass_saxophone_loop_cranesinthesky_Am.wav",
        "tests/regression_audio/GrimyHipHop_Saxophone_26_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav",
        "tests/regression_audio/HipHopTapes_28_Saxophone_D#m_90bpm.wav",
        "tests/regression_audio/HipHopTapes_29_Saxophone_D#m_90bpm.wav",
        "tests/regression_audio/HipHopTapes_31_Saxophone_D#m_90bpm.wav",
        "tests/regression_audio/SCY093_02_Sax_Loop_KeyAbm_89bpm_01.wav",
        "tests/regression_audio/SCY097_03_Sax_Loop_KeyEm_90bpm_01.wav",
    ),
    "NonVoice/Synth Pad and Keys": (
        "tests/regression_audio/CS_NJ2_135bpm_Pad_Aster_Am.wav",
        "tests/regression_audio/GS_Synth_Gangsta_Lead_G#min_97bpm.wav",
        "tests/regression_audio/MS_TLV1_03_The Way It Is_Synth Lead 1_Eminor_98bpm_Wet.wav",
        "tests/regression_audio/Stab 3.wav",
    ),
}


def broad_training_label(training_label: str) -> str:
    """Map an explicit trusted taxonomy label into the broad experiment."""
    if training_label.startswith("Drums/"):
        return "NonVoice/Drums and Percussion"
    if training_label.startswith("FX/"):
        return "NonVoice/Designed FX"
    if training_label.startswith("Instruments/Voice/"):
        return VOICE_LABEL
    if training_label.startswith("Instruments/Woodwinds/"):
        return "NonVoice/Sax and Woodwind"
    if training_label.startswith(("Instruments/Keys/", "Instruments/Synths/", "Instruments/Mallets and Bells/")):
        return "NonVoice/Synth Pad and Keys"
    if training_label.startswith("Instruments/"):
        return "NonVoice/Other Instruments"
    raise ValueError(f"trusted seed label is outside the supported taxonomy: {training_label}")


def safe_extract_audio_zip(zip_path: Path, output_dir: Path) -> list[tuple[Path, str]]:
    """Extract audio members without allowing path traversal."""
    extracted: list[tuple[Path, str]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename.replace("\\", "/"))
            if info.is_dir() or member.is_absolute() or ".." in member.parts:
                continue
            if Path(member.name).suffix.lower() not in AUDIO_SUFFIXES:
                continue
            destination = output_dir.joinpath(*member.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, destination.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
            subgroup = "One_Shot" if "One_Shot" in member.parts else "Loop"
            extracted.append((destination, subgroup))
    return sorted(extracted, key=lambda pair: sha256_file(pair[0]))


def _materialize(
    *,
    source_path: Path,
    destination_root: Path,
    label: str,
    prefix: str,
    position: int,
) -> Path:
    label_dir = destination_root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    destination = label_dir / f"{prefix}_{position:03d}{source_path.suffix.lower()}"
    os.symlink(source_path.resolve(), destination)
    return destination


def build_experiment(
    *,
    pack_zip: Path,
    seed_manifest: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Create separate preview and held-out roots with two leakage guards."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"experiment output must be absent or empty: {output_dir}")
    review_root = output_dir / "data" / "review_preview"
    heldout_root = output_dir / "data" / "heldout_eval"
    source_root = output_dir / "sources" / "premium_deep_house_vocals"
    review_root.mkdir(parents=True, exist_ok=True)
    heldout_root.mkdir(parents=True, exist_ok=True)

    seed_payload = json.loads(Path(seed_manifest).read_text(encoding="utf-8"))
    manifest_rows: list[dict[str, str]] = []
    byte_hashes_by_split: dict[str, set[str]] = {"review_preview": set(), "heldout_eval": set()}
    decoded_hashes_by_split: dict[str, set[str]] = {"review_preview": set(), "heldout_eval": set()}

    for position, case in enumerate(seed_payload["cases"], start=1):
        raw_path = Path(case["audio_path"])
        source_path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        label = broad_training_label(case["training_label"])
        destination = _materialize(
            source_path=source_path,
            destination_root=review_root,
            label=label,
            prefix="seed",
            position=position,
        )
        byte_digest = sha256_file(source_path)
        decoded_digest = decoded_audio_sha256(source_path)
        byte_hashes_by_split["review_preview"].add(byte_digest)
        decoded_hashes_by_split["review_preview"].add(decoded_digest)
        manifest_rows.append(
            {
                "split": "review_preview",
                "label": label,
                "subgroup": case["training_label"],
                "provenance": "trusted_training_seed_v1",
                "source_path": str(source_path.resolve()),
                "materialized_path": str(destination.absolute()),
                "file_sha256": byte_digest,
                "decoded_audio_sha256": decoded_digest,
            }
        )

    heldout_position = 0
    for source_path, subgroup in safe_extract_audio_zip(pack_zip, source_root):
        heldout_position += 1
        destination = _materialize(
            source_path=source_path,
            destination_root=heldout_root,
            label=VOICE_LABEL,
            prefix="pack",
            position=heldout_position,
        )
        byte_digest = sha256_file(source_path)
        decoded_digest = decoded_audio_sha256(source_path)
        byte_hashes_by_split["heldout_eval"].add(byte_digest)
        decoded_hashes_by_split["heldout_eval"].add(decoded_digest)
        manifest_rows.append(
            {
                "split": "heldout_eval",
                "label": VOICE_LABEL,
                "subgroup": subgroup,
                "provenance": "Premium Deep House Vocals.zip supplied failure set",
                "source_path": str(source_path.resolve()),
                "materialized_path": str(destination.absolute()),
                "file_sha256": byte_digest,
                "decoded_audio_sha256": decoded_digest,
            }
        )

    for label, relative_paths in sorted(PROTECTED_NEGATIVES.items()):
        subgroup = label.removeprefix("NonVoice/")
        for relative_path in relative_paths:
            heldout_position += 1
            source_path = PROJECT_ROOT / relative_path
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            destination = _materialize(
                source_path=source_path,
                destination_root=heldout_root,
                label=label,
                prefix="negative",
                position=heldout_position,
            )
            byte_digest = sha256_file(source_path)
            decoded_digest = decoded_audio_sha256(source_path)
            byte_hashes_by_split["heldout_eval"].add(byte_digest)
            decoded_hashes_by_split["heldout_eval"].add(decoded_digest)
            manifest_rows.append(
                {
                    "split": "heldout_eval",
                    "label": label,
                    "subgroup": subgroup,
                    "provenance": "maintained real-audio regression fixture",
                    "source_path": str(source_path.resolve()),
                    "materialized_path": str(destination.absolute()),
                    "file_sha256": byte_digest,
                    "decoded_audio_sha256": decoded_digest,
                }
            )

    byte_overlap = byte_hashes_by_split["review_preview"] & byte_hashes_by_split["heldout_eval"]
    decoded_overlap = decoded_hashes_by_split["review_preview"] & decoded_hashes_by_split["heldout_eval"]
    if byte_overlap or decoded_overlap:
        raise ValueError(f"experiment leakage: byte={len(byte_overlap)} decoded={len(decoded_overlap)}")

    manifest_path = output_dir / "experiment_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    split_counts = Counter(row["split"] for row in manifest_rows)
    summary = {
        "schema_version": 1,
        "training_count": split_counts["review_preview"],
        "heldout_count": split_counts["heldout_eval"],
        "heldout_vocal_count": sum(
            row["split"] == "heldout_eval" and row["label"] == VOICE_LABEL for row in manifest_rows
        ),
        "heldout_negative_count": sum(
            row["split"] == "heldout_eval" and row["label"] != VOICE_LABEL for row in manifest_rows
        ),
        "byte_hash_overlap_count": 0,
        "decoded_audio_overlap_count": 0,
        "training_label_counts": dict(
            sorted(Counter(row["label"] for row in manifest_rows if row["split"] == "review_preview").items())
        ),
        "heldout_label_counts": dict(
            sorted(Counter(row["label"] for row in manifest_rows if row["split"] == "heldout_eval").items())
        ),
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-zip", type=Path, required=True)
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        default=PROJECT_ROOT / "tests" / "acceptance" / "locked_smoke_v1" / "trusted_training_seed_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the experiment and print its split summary."""
    args = parse_args(argv)
    summary = build_experiment(
        pack_zip=args.pack_zip,
        seed_manifest=args.seed_manifest,
        output_dir=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
