"""Parameterization settings for the default configuration."""

from libs.python.schemas.basic_models import BaseModelUpd, BaseEnum, Field

import libs.python.schemas.defaults as defs

from .llm_parameters import LLMParameters
from .parallelization_parameters import ParallelizationParameters
from ..enums import AsyncType


class LLMConfig(BaseModelUpd):
    """Base class for LLM-configured steps."""

    llm: LLMParameters = Field(
        description="The LLM configuration to use.", default=LLMParameters()
    )
    parallelization: ParallelizationParameters = Field(
        description="The parallelization configuration to use.",
        default=ParallelizationParameters(),
    )
    async_mode: AsyncType = Field(
        description="The async mode to use.", default=defs.ASYNC_MODE
    )
