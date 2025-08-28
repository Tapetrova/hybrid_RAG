"""Parameterization settings for the default configuration."""

from pathlib import Path

from pydantic import Field

import libs.python.schemas.defaults as defs

from .llm_config import LLMConfig
from ...prompts.graphrag.summarize_descriptions.summarize_descriptions_prompt import (
    SUMMARIZE_DESCRIPTIONS_PROMPT,
)


class SummarizeDescriptionsConfig(LLMConfig):
    """Configuration section for description summarization."""

    prompt: str | None = Field(
        description="The description summarization prompt to use.",
        default=SUMMARIZE_DESCRIPTIONS_PROMPT,
    )
    max_length: int = Field(
        description="The description summarization maximum length.",
        default=defs.SUMMARIZE_DESCRIPTIONS_MAX_LENGTH,
    )
    strategy: dict | None = Field(
        description="The override strategy to use.", default=None
    )

    def resolved_strategy(self, root_dir: str) -> dict:
        """Get the resolved description summarization strategy."""
        from apps.knowledge_manager.src.graphrag.index.verbs.entities.summarize import (
            SummarizeStrategyType,
        )

        return self.strategy or {
            "type": SummarizeStrategyType.graph_intelligence,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "summarize_prompt": self.prompt if self.prompt else None,
            "max_summary_length": self.max_length,
        }
