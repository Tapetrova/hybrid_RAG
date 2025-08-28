import time
from typing import List, Optional

from langchain_community.callbacks.openai_info import standardize_model_name
from langchain_core.messages import BaseMessage

from apps.agent_flow.src.cat_main_agent.openai_helper_strict_schema_output.schemas import (
    MainAgentFinishOutput,
    References,
)
from apps.agent_flow.src.cat_main_agent.openai_helper_strict_schema_output.base_structure_chain import (
    BaseStructureChainOpenai,
    StructureChainOpenaiOutput as BaseStructureChainOpenaiOutput,
)
from libs.python.schemas.basic_models import LLMModel
from libs.python.utils.logger import logger
from libs.python.utils.token_managment import TokenManager


class StructureChainOpenaiOutput(BaseStructureChainOpenaiOutput):
    output: MainAgentFinishOutput


class StructureChainOpenai(BaseStructureChainOpenai):
    def __init__(self, llm_model: LLMModel):
        super().__init__(chain_name="main_agent", llm_model=llm_model)

    async def ainvoke(
        self,
        system_prompt: str,
        input_message: str,
        chat_history: List[BaseMessage],
        tool_output: str,
        temperature: float,
        timeout: int = 120,
    ) -> StructureChainOpenaiOutput:
        input_tokens = self._calculate_input_tokens(
            system_prompt, input_message, chat_history, tool_output
        )
        messages = self._build_messages(
            system_prompt, input_message, chat_history, tool_output
        )

        st = time.time()
        completion = await self._async_client.beta.chat.completions.parse(
            model=self._llm_model.value,
            messages=messages,
            response_format=MainAgentFinishOutput,
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
                output=MainAgentFinishOutput(
                    agent_answer=message.refusal,
                    references=References(
                        ids_reports=[], ids_entities=[], ids_relationships=[]
                    ),
                ),
                process_metrics=process_metrics,
            )

    def invoke(
        self,
        system_prompt: str,
        input_message: str,
        chat_history: List[BaseMessage],
        tool_output: str,
        temperature: float,
        timeout: int = 120,
    ) -> StructureChainOpenaiOutput:
        input_tokens = self._calculate_input_tokens(
            system_prompt, input_message, chat_history, tool_output
        )
        messages = self._build_messages(
            system_prompt, input_message, chat_history, tool_output
        )

        st = time.time()
        completion = self._client.beta.chat.completions.parse(
            model=self._llm_model.value,
            messages=messages,
            response_format=MainAgentFinishOutput,
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
                output=MainAgentFinishOutput(
                    agent_answer=message.refusal,
                    references=References(
                        ids_reports=[], ids_entities=[], ids_relationships=[]
                    ),
                ),
                process_metrics=process_metrics,
            )
