import time
from typing import List, Optional, Type

from langchain_community.callbacks.openai_info import standardize_model_name
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from apps.agent_flow.src.cat_main_agent.openai_helper_strict_schema_output.base_structure_chain import (
    BaseStructureChainOpenai,
    StructureChainOpenaiOutput,
)
from libs.python.schemas.basic_models import LLMModel
from libs.python.utils.logger import logger
from libs.python.utils.token_managment import TokenManager


class UniversalStructureChainOpenai(BaseStructureChainOpenai):
    """Universal chain for structured output with any Pydantic schema."""

    async def ainvoke(
        self,
        system_prompt: str,
        input_message: str,
        temperature: float,
        schema_output: Type[BaseModel],
        timeout: int = 120,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> StructureChainOpenaiOutput:
        input_tokens = self._calculate_input_tokens(
            system_prompt, input_message, chat_history
        )
        messages = self._build_messages(system_prompt, input_message, chat_history)

        st = time.time()
        completion = await self._async_client.beta.chat.completions.parse(
            model=self._llm_model.value,
            messages=messages,
            response_format=schema_output,
            temperature=temperature,
            timeout=timeout,
        )
        time_processed = time.time() - st

        message = completion.choices[0].message
        output_tokens = TokenManager.num_tokens_from_string(
            message.content, model_name=self._llm_model
        )

        process_metrics = self._calculate_metrics(
            standardize_model_name(self._llm_model.value),
            input_tokens,
            output_tokens,
            time_processed,
        )

        if message.parsed:
            return StructureChainOpenaiOutput(
                output=message.parsed,
                process_metrics=process_metrics,
            )
        else:
            logger.info(
                f"message.parsed: {message.parsed}; message.refusal: {message.refusal};"
            )

            return StructureChainOpenaiOutput(
                output=None,
                process_metrics=process_metrics,
            )

    def invoke(
        self,
        system_prompt: str,
        input_message: str,
        temperature: float,
        schema_output: Type[BaseModel],
        timeout: int = 120,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> StructureChainOpenaiOutput:
        input_tokens = self._calculate_input_tokens(
            system_prompt, input_message, chat_history
        )
        messages = self._build_messages(system_prompt, input_message, chat_history)

        st = time.time()
        completion = self._client.beta.chat.completions.parse(
            model=self._llm_model.value,
            messages=messages,
            response_format=schema_output,
            temperature=temperature,
            timeout=timeout,
        )
        time_processed = time.time() - st

        message = completion.choices[0].message
        output_tokens = TokenManager.num_tokens_from_string(
            message.content, model_name=self._llm_model
        )

        process_metrics = self._calculate_metrics(
            standardize_model_name(self._llm_model.value),
            input_tokens,
            output_tokens,
            time_processed,
        )

        if message.parsed:
            return StructureChainOpenaiOutput(
                output=message.parsed,
                process_metrics=process_metrics,
            )
        else:
            logger.info(
                f"message.parsed: {message.parsed}; message.refusal: {message.refusal};"
            )

            return StructureChainOpenaiOutput(
                output=None,
                process_metrics=process_metrics,
            )
