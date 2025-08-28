"""Common types for the GraphRAG knowledge model."""

from collections.abc import Callable

TextEmbedder = Callable[[str], list[float]]
