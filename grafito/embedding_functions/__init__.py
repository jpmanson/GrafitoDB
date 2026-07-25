"""Embedding function registry."""

from __future__ import annotations

from typing import Any

from .base import EmbeddingFunction
from .sentence_transformer import SentenceTransformerEmbeddingFunction
from .huggingface import HuggingFaceEmbeddingFunction
from .openai import OpenAIEmbeddingFunction
from .cohere import CohereEmbeddingFunction
from .ollama import OllamaEmbeddingFunction
from .amazon_bedrock import AmazonBedrockEmbeddingFunction
from .jina import JinaEmbeddingFunction
from .voyageai import VoyageAIEmbeddingFunction
from .mistral import MistralEmbeddingFunction
from .google_genai import GoogleGenAIEmbeddingFunction
from .tfhub import TensorFlowHubEmbeddingFunction
from .together_ai import TogetherAIEmbeddingFunction


_EMBEDDING_FUNCTIONS: dict[str, type[EmbeddingFunction]] = {
    SentenceTransformerEmbeddingFunction.name(): SentenceTransformerEmbeddingFunction,
    HuggingFaceEmbeddingFunction.name(): HuggingFaceEmbeddingFunction,
    OpenAIEmbeddingFunction.name(): OpenAIEmbeddingFunction,
    CohereEmbeddingFunction.name(): CohereEmbeddingFunction,
    OllamaEmbeddingFunction.name(): OllamaEmbeddingFunction,
    AmazonBedrockEmbeddingFunction.name(): AmazonBedrockEmbeddingFunction,
    JinaEmbeddingFunction.name(): JinaEmbeddingFunction,
    VoyageAIEmbeddingFunction.name(): VoyageAIEmbeddingFunction,
    MistralEmbeddingFunction.name(): MistralEmbeddingFunction,
    GoogleGenAIEmbeddingFunction.name(): GoogleGenAIEmbeddingFunction,
    TensorFlowHubEmbeddingFunction.name(): TensorFlowHubEmbeddingFunction,
    TogetherAIEmbeddingFunction.name(): TogetherAIEmbeddingFunction,
}


def register_embedding_function_class(cls: type[EmbeddingFunction]) -> None:
    """Register an embedding function class."""
    _EMBEDDING_FUNCTIONS[cls.name()] = cls


def get_embedding_function_class(name: str) -> type[EmbeddingFunction]:
    """Look up a registered embedding function class by name.

    The counterpart to :func:`create_embedding_function` for callers that want to
    build via the class ``__init__`` (and its defaults) rather than the stricter
    ``build_from_config`` — e.g. ``get_embedding_function_class("sentence_transformer")()``
    yields the default MiniLM model with no config. Raises ``ValueError`` naming
    the available functions when ``name`` is unknown.
    """
    if name not in _EMBEDDING_FUNCTIONS:
        raise ValueError(
            f"Unknown embedding function '{name}'. Available: {list_embedding_functions()}"
        )
    return _EMBEDDING_FUNCTIONS[name]


def create_embedding_function(name: str, config: dict[str, Any]) -> EmbeddingFunction:
    """Create an embedding function from a config dict."""
    if name not in _EMBEDDING_FUNCTIONS:
        raise ValueError(f"Unknown embedding function '{name}'")
    cls = _EMBEDDING_FUNCTIONS[name]
    cls.validate_config(config)
    return cls.build_from_config(config)


def list_embedding_functions() -> list[str]:
    return sorted(_EMBEDDING_FUNCTIONS.keys())


__all__ = [
    "EmbeddingFunction",
    "SentenceTransformerEmbeddingFunction",
    "HuggingFaceEmbeddingFunction",
    "OpenAIEmbeddingFunction",
    "CohereEmbeddingFunction",
    "OllamaEmbeddingFunction",
    "AmazonBedrockEmbeddingFunction",
    "JinaEmbeddingFunction",
    "VoyageAIEmbeddingFunction",
    "MistralEmbeddingFunction",
    "GoogleGenAIEmbeddingFunction",
    "TensorFlowHubEmbeddingFunction",
    "TogetherAIEmbeddingFunction",
    "register_embedding_function_class",
    "create_embedding_function",
    "get_embedding_function_class",
    "list_embedding_functions",
]
