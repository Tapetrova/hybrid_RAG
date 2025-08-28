"""Parameterization settings for the default configuration."""

from libs.python.schemas.basic_models import BaseModelUpd, BaseEnum, Field

import libs.python.schemas.defaults as defs


class ChunkingConfig(BaseModelUpd):
    """Configuration section for chunking."""

    size: int = Field(description="The chunk size to use.", default=defs.CHUNK_SIZE)
    overlap: int = Field(
        description="The chunk overlap to use.", default=defs.CHUNK_OVERLAP
    )
    group_by_columns: list[str] = Field(
        description="The chunk by columns to use.",
        default=defs.CHUNK_GROUP_BY_COLUMNS,
    )
    strategy: dict | None = Field(
        description="The chunk strategy to use, overriding the default tokenization strategy",
        default=None,
    )

    def resolved_strategy(self) -> dict:
        """Get the resolved chunking strategy."""
        from apps.knowledge_manager.src.graphrag.index.verbs.text.chunk import (
            ChunkStrategyType,
        )

        return self.strategy or {
            "type": ChunkStrategyType.tokens,
            "chunk_size": self.size,
            "chunk_overlap": self.overlap,
            "group_by_columns": self.group_by_columns,
        }
