"""A module containing ChunkStrategy definition."""

from collections.abc import Callable, Iterable
from typing import Any

from datashaper import ProgressTicker

from apps.knowledge_manager.src.graphrag.index.verbs.text.chunk.typing import TextChunk

# Given a list of document texts, return a list of tuples of (source_doc_indices, text_chunk)

ChunkStrategy = Callable[
    [list[str], dict[str, Any], ProgressTicker], Iterable[TextChunk]
]
