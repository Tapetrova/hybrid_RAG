"""Parameterization settings for the default configuration."""

from pathlib import Path

from pydantic import Field

import libs.python.schemas.defaults as defs

from .llm_config import LLMConfig
from ...prompts.graphrag.claim_extraction.claim_extraction_prompt import (
    CLAIM_EXTRACTION_PROMPT,
)


class ClaimExtractionConfig(LLMConfig):
    """Configuration section for claim extraction."""

    enabled: bool = Field(
        description="Whether claim extraction is enabled.", default=True
    )
    prompt: str | None = Field(
        description="The claim extraction prompt to use.",
        default=CLAIM_EXTRACTION_PROMPT,
    )
    description: str = Field(
        description="The claim description to use.",
        default=defs.CLAIM_DESCRIPTION,
    )
    max_gleanings: int = Field(
        description="The maximum number of entity gleanings to use.",
        default=defs.CLAIM_MAX_GLEANINGS,
    )
    strategy: dict | None = Field(
        description="The override strategy to use.", default=None
    )

    def resolved_strategy(self, root_dir: str) -> dict:
        """Get the resolved claim extraction strategy."""
        from apps.knowledge_manager.src.graphrag.index.verbs.covariates.extract_covariates import (
            ExtractClaimsStrategyType,
        )

        return self.strategy or {
            "type": ExtractClaimsStrategyType.graph_intelligence,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "extraction_prompt": self.prompt if self.prompt else None,
            "claim_description": self.description,
            "max_gleanings": self.max_gleanings,
        }
