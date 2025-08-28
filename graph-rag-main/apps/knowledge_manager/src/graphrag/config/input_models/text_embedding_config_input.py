"""Parameterization settings for the default configuration."""

from typing_extensions import NotRequired

from libs.python.schemas.enums import (
    TextEmbeddingTarget,
)

from .llm_config_input import LLMConfigInput


class TextEmbeddingConfigInput(LLMConfigInput):
    """Configuration section for text embeddings."""

    batch_size: NotRequired[int | str | None]
    batch_max_tokens: NotRequired[int | str | None]
    target: NotRequired[TextEmbeddingTarget | str | None]
    skip: NotRequired[list[str] | str | None]
    vector_store: NotRequired[dict | None]
    strategy: NotRequired[dict | None]
