import time
from typing import List, Optional, Type, TypeVar

from langchain_community.callbacks.openai_info import (
    standardize_model_name,
    MODEL_COST_PER_1K_TOKENS,
    get_openai_token_cost_for_model,
)
from langchain_core.messages import BaseMessage
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

from libs.python.schemas.basic_models import LLMModel, BaseModelUpd
from libs.python.schemas.metrics import (
    LLMProcessMetrics,
    InputLLMProcessMetrics,
    OutputLLMProcessMetrics,
    TotalLLMProcessMetrics,
)
from libs.python.utils.logger import logger
from libs.python.utils.token_managment import TokenManager

T = TypeVar("T", bound=BaseModel)


class StructureChainOpenaiOutput(BaseModelUpd):
    output: Optional[BaseModel]
    process_metrics: LLMProcessMetrics
    context_data_from_tool: Optional[str] = None


class BaseStructureChainOpenai:
    """Base class for OpenAI structured output chains to reduce code duplication."""

    def __init__(self, chain_name: str, llm_model: LLMModel):
        self._async_client = AsyncOpenAI()
        self._client = OpenAI()
        self._llm_model = llm_model
        self._chain_name = chain_name

    def _calculate_metrics(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        time_processed: float,
        count_input_tokens_exceeded: int = 0,
    ) -> LLMProcessMetrics:
        """Calculate token usage metrics."""
        input_cost = 0
        output_cost = 0

        if model_name in MODEL_COST_PER_1K_TOKENS:
            input_cost = get_openai_token_cost_for_model(
                model_name, input_tokens, is_completion=False
            )
            output_cost = get_openai_token_cost_for_model(
                model_name, output_tokens, is_completion=True
            )

        return LLMProcessMetrics(
            name=self._chain_name,
            input_metrics=InputLLMProcessMetrics(
                input_tokens=input_tokens,
                input_cost=input_cost,
            ),
            output_metrics=OutputLLMProcessMetrics(
                output_tokens=output_tokens,
                output_cost=output_cost,
            ),
            total_metrics=TotalLLMProcessMetrics(
                total_tokens=input_tokens + output_tokens,
                total_cost=input_cost + output_cost,
            ),
            count_input_tokens_exceeded=count_input_tokens_exceeded,
            time_exe=time_processed,
        )

    def _calculate_input_tokens(
        self,
        system_prompt: str,
        input_message: str,
        chat_history: Optional[List[BaseMessage]] = None,
        tool_output: Optional[str] = None,
    ) -> int:
        """Calculate total input tokens."""
        model_name = standardize_model_name(self._llm_model.value)

        tokens = TokenManager.num_tokens_from_string(
            system_prompt, model_name=self._llm_model
        ) + TokenManager.num_tokens_from_string(
            input_message, model_name=self._llm_model
        )

        if chat_history:
            tokens += sum(
                TokenManager.num_tokens_from_string(
                    mes.content, model_name=self._llm_model
                )
                for mes in chat_history
            )

        if tool_output:
            tokens += TokenManager.num_tokens_from_string(
                tool_output, model_name=self._llm_model
            )

        return tokens

    def _build_messages(
        self,
        system_prompt: str,
        input_message: str,
        chat_history: Optional[List[BaseMessage]] = None,
        tool_output: Optional[str] = None,
    ) -> List[dict]:
        """Build messages for OpenAI API."""
        messages = [{"role": "system", "content": system_prompt}]

        if chat_history:
            messages.extend(
                [
                    {
                        "role": "user" if mes.type == "human" else "assistant",
                        "content": mes.content,
                    }
                    for mes in chat_history
                ]
            )

        messages.append({"role": "user", "content": input_message})

        if tool_output:
            messages.append(
                {
                    "role": "system",
                    "content": f"USE THE FOLLOWING TOOL OUTPUT TO ANSWER TO HUMAN (`HUMAN` is the same as `user`): \n{tool_output}",
                }
            )

        return messages
