"""LLM Parameters model."""

from libs.python.schemas.basic_models import BaseModelUpd, BaseEnum, Field

import libs.python.schemas.defaults as defs


class ParallelizationParameters(BaseModelUpd):
    """LLM Parameters model."""

    stagger: float = Field(
        description="The stagger to use for the LLM service.",
        default=defs.PARALLELIZATION_STAGGER,
    )
    num_threads: int = Field(
        description="The number of threads to use for the LLM service.",
        default=defs.PARALLELIZATION_NUM_THREADS,
    )
