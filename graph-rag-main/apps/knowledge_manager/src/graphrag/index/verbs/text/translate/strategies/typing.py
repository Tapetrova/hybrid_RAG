"""A module containing 'TextTranslationResult' model."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from datashaper import VerbCallbacks

from apps.knowledge_manager.src.graphrag.index.cache import PipelineCache


@dataclass
class TextTranslationResult:
    """Text translation result class definition."""

    translations: list[str]


TextTranslationStrategy = Callable[
    [list[str], dict[str, Any], VerbCallbacks, PipelineCache],
    Awaitable[TextTranslationResult],
]
