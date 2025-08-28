"""Parameterization settings for the default configuration."""

from libs.python.prompts.graphrag.global_search.map_system_prompt import (
    MAP_SYSTEM_PROMPT,
)
from libs.python.prompts.graphrag.global_search.reduce_system_prompt import (
    REDUCE_SYSTEM_PROMPT,
    GENERAL_KNOWLEDGE_INSTRUCTION,
)
from libs.python.schemas.basic_models import BaseModelUpd, Field
from libs.python.schemas.graphrag.llm_config import LLMConfig
from libs.python.schemas.graphrag.llm_parameters import (
    LLMParametersReduced,
)


class ContextBuilderParams(BaseModelUpd):
    use_community_summary: bool = Field(
        description="False means using full community reports. "
        "True means using community short summaries.",
        default=False,
    )
    shuffle_data: bool = Field(default=True)
    include_community_rank: bool = Field(default=True)
    min_community_rank: int = Field(default=0)
    community_rank_name: str = Field(default="rank")
    include_community_weight: bool = Field(default=True)
    community_weight_name: str = Field(default="occurrence weight")
    normalize_community_weight: bool = Field(default=True)
    max_tokens: int = Field(default=12_000)
    context_name: str = Field(default="Reports")
    conversation_history_user_turns_only: bool = Field(default=False)
    conversation_history_max_turns: int | None = Field(default=200)


class GlobalSearchConfig(BaseModelUpd):
    """The default configuration section for Cache."""

    run_reduce_response: bool = Field(
        default=False,
        description="Return last reduce llm call to get final result "
        "or return (if False) only map analyst responses ",
    )
    llm_model: LLMConfig = Field(default=LLMConfig())
    context_builder_params: ContextBuilderParams = Field(default=ContextBuilderParams())
    max_data_tokens: int = Field(
        description="The data llm maximum tokens. "
        "Change this based on the token limit you have on your model "
        "(if you are using a model with 8k limit, a good setting could be 5000)",
        default=12_000,
    )
    map_llm_params: LLMParametersReduced = Field(
        default=LLMParametersReduced(max_tokens=1000, temperature=0.0, n=1, top_p=1)
    )
    map_system_prompt: str = Field(default=MAP_SYSTEM_PROMPT)
    reduce_system_prompt: str = Field(default=REDUCE_SYSTEM_PROMPT)
    general_knowledge_inclusion_prompt: str = Field(
        default=GENERAL_KNOWLEDGE_INSTRUCTION
    )
    reduce_llm_params: LLMParametersReduced = Field(
        default=LLMParametersReduced(max_tokens=1000, temperature=0.0, n=1, top_p=1)
    )
    allow_general_knowledge: bool = Field(
        description="Set this to True will add instruction to encourage the LLM to "
        "incorporate general knowledge in the response, "
        "which may increase hallucinations, but could be useful in some use cases.",
        default=False,
    )
    json_mode: bool = Field(
        description="set this to False if your LLM model does not support JSON mode.",
        default=True,
    )
    concurrency: int = Field(
        description="The number of concurrent requests.",
        default=32,
    )
    response_type: str = Field(
        description="Free form text describing the response type and format, "
        "can be anything, e.g. prioritized list, "
        "single paragraph, multiple paragraphs, multiple-page report",
        default="multiple paragraphs",
    )
