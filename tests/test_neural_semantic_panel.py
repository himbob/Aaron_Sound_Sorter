from __future__ import annotations

import numpy as np
import pytest

from aaron_sound_sorter.neural_audio.contracts import EmbeddingRecord
from aaron_sound_sorter.neural_audio.semantic_panel import (
    assess_semantic_label_compatibility,
    compatible_semantic_families,
    flattened_semantic_prompts,
    predict_semantic_family,
)


def test_semantic_panel_ranks_family_from_audio_text_similarity() -> None:
    prompts, families = flattened_semantic_prompts({"voice": ("voice one", "voice two"), "drums": ("drum",)})
    record = EmbeddingRecord(
        provider_id="hf_clap",
        model_id="model",
        file_sha256="a" * 64,
        vector=np.asarray([1.0, 0.0], dtype=np.float32),
        segment_count=1,
        sample_rate=48_000,
    )
    text_embeddings = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)

    prediction = predict_semantic_family(record, text_embeddings, families)

    assert prompts == ("voice one", "voice two", "drum")
    assert prediction.predicted_family == "voice"
    assert prediction.second_family == "drums"
    assert prediction.margin > 0.8


def test_semantic_panel_rejects_dimension_mismatch() -> None:
    record = EmbeddingRecord(
        provider_id="hf_clap",
        model_id="model",
        file_sha256="b" * 64,
        vector=np.asarray([1.0, 0.0], dtype=np.float32),
        segment_count=1,
        sample_rate=48_000,
    )

    with pytest.raises(ValueError, match="dimensions differ"):
        predict_semantic_family(record, np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), ("voice",))


def test_decisive_impact_evidence_rejects_a_voice_label() -> None:
    assessment = assess_semantic_label_compatibility(
        "Instruments/Voice/Vocal Loops/Loops",
        predicted_family="fx_impact",
        top_score=0.24,
        family_scores={"fx_impact": 0.24, "human_voice": 0.11},
    )

    assert assessment.contradictory is True
    assert assessment.reason == "semantic_family_contradiction"


def test_nonvoice_label_requires_a_larger_contradiction_gap() -> None:
    assessment = assess_semantic_label_compatibility(
        "FX/Impacts and Hits/Generic Impact/One Shots",
        predicted_family="bass",
        top_score=0.22,
        family_scores={"bass": 0.22, "drums": 0.15, "fx_impact": 0.07},
    )

    assert assessment.contradictory is False
    assert assessment.reason == "semantic_family_ambiguous"


def test_ambiguous_semantic_panel_does_not_veto_training() -> None:
    assessment = assess_semantic_label_compatibility(
        "Instruments/Woodwinds/Saxophone/Loops",
        predicted_family="human_voice",
        top_score=0.13,
        family_scores={"human_voice": 0.13, "woodwind_reed": 0.10, "brass": 0.09},
    )

    assert assessment.contradictory is False
    assert assessment.reason == "semantic_family_ambiguous"


def test_taxonomy_compatibility_keeps_rich_fx_and_instrument_families() -> None:
    assert compatible_semantic_families("FX/Everyday Foley/Glass/One Shots") == (
        "fx_foley",
        "fx_impact",
    )
    riser_families = compatible_semantic_families("FX/Structural and Transitional FX/Risers and Builds/Long FX")
    assert riser_families[0] == "fx_transition"
    assert {"synth", "drum_cymbal", "fx_texture"}.issubset(riser_families)
    assert compatible_semantic_families("Instruments/Woodwinds/Saxophone/Loops") == (
        "woodwind_reed",
        "brass",
    )
    assert compatible_semantic_families("FX/Designed Noise FX/Siren/Long FX") == (
        "fx_alert",
        "fx_glitch",
        "synth",
    )
    assert compatible_semantic_families("Drums/Kick Drums/Sub Kick/One Shots")[:2] == (
        "drum_kick",
        "drums",
    )


def test_fx_role_can_remain_compatible_with_its_audible_source_material() -> None:
    assessment = assess_semantic_label_compatibility(
        "FX/Structural and Transitional FX/Reverses and Tails/Generic Reverse/One Shots",
        predicted_family="drum_backbeat",
        top_score=0.19,
        family_scores={"drum_backbeat": 0.19, "fx_transition": 0.05},
    )

    assert assessment.contradictory is False
    assert assessment.reason == "semantic_family_compatible"
