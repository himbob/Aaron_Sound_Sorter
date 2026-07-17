"""Tests for the read-only audio intelligence foundation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aaron_audio_intelligence import (
    AAI_SIDECAR_SCHEMA_VERSION,
    PrototypeOwnerBrain,
    TrainableBrainExample,
    build_sidecar_from_sort_result,
)
from aaron_audio_intelligence.sidecar_writer import write_audio_intelligence_sidecars
from aaron_sound_sorter.domain.models import (
    AudioPhysics,
    CategoryGuess,
    ConsensusDecision,
    SharedAudioFacts,
    SortFileResult,
    VoterResult,
)


def test_prototype_owner_brain_uses_positive_and_negative_examples() -> None:
    """Positive and negative prototypes should both affect ownership evidence."""
    brain = PrototypeOwnerBrain(
        owner_name="DesignedMotionFXOwnerBrain",
        examples=(
            TrainableBrainExample(
                example_id="riser-positive",
                label="transition_fx",
                vector=(1.0, 0.0, 0.0),
                polarity="positive",
                weight=1.0,
            ),
            TrainableBrainExample(
                example_id="synth-arp-negative",
                label="synth_arp",
                vector=(0.0, 1.0, 0.0),
                polarity="negative",
                weight=1.0,
            ),
        ),
        allowed_final_tops=("FX",),
        blocked_final_tops=("Drums", "Instruments"),
    )

    fx_result = brain.score_vector((0.9, 0.1, 0.0))
    synth_result = brain.score_vector((0.1, 0.9, 0.0))

    assert fx_result.score > 0
    assert fx_result.allowed_final_tops == ("FX",)
    assert "riser-positive" in fx_result.training_match_ids
    assert synth_result.score < 0
    assert "synth-arp-negative" in synth_result.training_match_ids


def test_sidecar_separates_final_folder_from_source_like_and_usable_as(tmp_path: Path) -> None:
    """The sidecar should preserve final folder and reusable evidence separately."""
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not real audio but stable bytes")
    result = fake_sort_result(audio_path)

    sidecar = build_sidecar_from_sort_result(result)
    payload = sidecar.as_dict()

    assert payload["schema_version"] == AAI_SIDECAR_SCHEMA_VERSION
    assert payload["final_sort"]["folder"] == "Instruments/Keys/Piano/One Shots"
    assert payload["source_like"]["piano_keys"] == 0.91
    assert payload["usable_as"]["one_shot"] == 0.82
    assert payload["music"]["policy"] == "read_only_placeholder_until_music_property_phase"
    assert payload["components"]["policy"] == "read_only_placeholder_until_component_phase"
    assert payload["training"]["safe_to_auto_train"] is False


def test_audio_intelligence_sidecar_writer_outputs_jsonl(tmp_path: Path) -> None:
    """Sort outputs should be able to write a read-only sidecar JSONL file."""
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"stable bytes")
    output_path = tmp_path / "sidecars.jsonl"

    write_audio_intelligence_sidecars(output_path, [fake_sort_result(audio_path)])
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert rows[0]["schema_version"] == AAI_SIDECAR_SCHEMA_VERSION
    assert rows[0]["debug_refs"]["debug_packet"] == "Aaron_Sorted_Sounds_debug_packets.jsonl:1"


def fake_sort_result(audio_path: Path) -> SortFileResult:
    """Return a minimal sorter result with useful evidence values."""
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "shape_vote": {"shape": "one_shot_hit_shape", "confidence": 0.82},
            "physics_subpanels": {
                "flat": {
                    "struck_keys_score": 0.91,
                    "role_one_shot_score": 0.82,
                    "role_loop_score": 0.12,
                    "drum_hit_score": 0.18,
                }
            },
        },
    )
    guess = CategoryGuess(
        label="Instruments/Keys/Piano/One Shots",
        folder_path="Instruments/Keys/Piano/One Shots",
        top_family="Instruments",
        score=0.91,
        confidence=0.86,
        rank=1,
        reason="test",
    )
    return SortFileResult(
        source_path=audio_path,
        placed_path=None,
        physics=AudioPhysics(
            source_path=audio_path,
            fingerprint=np.array([0.1, 0.2, 0.3]),
            duration_sec=0.4,
            read_status="ok",
        ),
        facts=facts,
        brain_votes=VoterResult("brain_full", [guess]),
        physics_votes=VoterResult("physics", [guess]),
        decision=ConsensusDecision(
            final_label="Instruments/Keys/Piano/One Shots",
            final_top="Instruments",
            folder_path="Instruments/Keys/Piano/One Shots",
            consensus_status="test_status",
            reason="test reason",
        ),
    )
