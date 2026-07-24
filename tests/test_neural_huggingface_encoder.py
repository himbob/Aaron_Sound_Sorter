from __future__ import annotations

from types import SimpleNamespace

from aaron_sound_sorter.neural_audio.providers.huggingface_encoder import prepare_audio_model_config


def test_prepare_audio_model_config_adds_historical_mert_hubert_default() -> None:
    config = SimpleNamespace(model_type="mert_model")

    returned = prepare_audio_model_config(config)

    assert returned is config
    assert config.conv_pos_batch_norm is False


def test_prepare_audio_model_config_preserves_explicit_or_unrelated_settings() -> None:
    explicit = SimpleNamespace(model_type="mert_model", conv_pos_batch_norm=True)
    unrelated = SimpleNamespace(model_type="other_audio_model")

    prepare_audio_model_config(explicit)
    prepare_audio_model_config(unrelated)

    assert explicit.conv_pos_batch_norm is True
    assert not hasattr(unrelated, "conv_pos_batch_norm")
