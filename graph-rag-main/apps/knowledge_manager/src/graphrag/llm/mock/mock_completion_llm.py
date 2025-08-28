"""LLM Static Response method definition."""

import logging

from typing_extensions import Unpack

from apps.knowledge_manager.src.graphrag.llm.base import BaseLLM
from apps.knowledge_manager.src.graphrag.llm.types import (
    CompletionInput,
    CompletionOutput,
    LLMInput,
)

from libs.python.utils.logger import logger

log = logger


class MockCompletionLLM(
    BaseLLM[
        CompletionInput,
        CompletionOutput,
    ]
):
    """Mock Completion LLM for testing purposes."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self._on_error = None

    async def _execute_llm(
        self,
        input: CompletionInput,
        **kwargs: Unpack[LLMInput],
    ) -> CompletionOutput:
        return self.responses[0]
