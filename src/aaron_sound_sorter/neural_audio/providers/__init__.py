"""Pluggable frozen embedding providers."""

from .base import EmbeddingProvider
from .huggingface_clap import DEFAULT_CLAP_MODEL, HuggingFaceClapProvider
from .huggingface_encoder import HuggingFaceAudioEncoderProvider

__all__ = [
    "DEFAULT_CLAP_MODEL",
    "EmbeddingProvider",
    "HuggingFaceAudioEncoderProvider",
    "HuggingFaceClapProvider",
]
