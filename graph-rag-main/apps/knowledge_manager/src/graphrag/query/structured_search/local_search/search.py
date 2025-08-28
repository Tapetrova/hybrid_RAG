import asyncio
import json
import time
from typing import Any

import tiktoken

from apps.knowledge_manager.src.graphrag.query.context_builder.builders import (
    LocalContextBuilder,
)
from apps.knowledge_manager.src.graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from apps.knowledge_manager.src.graphrag.query.llm.base import BaseLLM, BaseLLMCallback
from apps.knowledge_manager.src.graphrag.query.llm.text_utils import num_tokens
from apps.knowledge_manager.src.graphrag.query.structured_search.base import (
    BaseSearch,
    SearchResult,
)
from libs.python.prompts.graphrag.local_search.system_prompt import (
    LOCAL_SEARCH_SYSTEM_PROMPT,
)
from libs.python.utils.logger import logger

DEFAULT_LLM_PARAMS = {
    "max_tokens": 1500,
    "temperature": 0.0,
}


log = logger


class LocalSearch(BaseSearch):
    """Search orchestration for local search mode."""

    def __init__(
        self,
        llm: BaseLLM,
        context_builder: LocalContextBuilder,
        token_encoder: tiktoken.Encoding | None = None,
        system_prompt: str = LOCAL_SEARCH_SYSTEM_PROMPT,
        response_type: str = "multiple paragraphs",
        callbacks: list[BaseLLMCallback] | None = None,
        llm_params: dict[str, Any] = DEFAULT_LLM_PARAMS,
        context_builder_params: dict | None = None,
    ):
        super().__init__(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            llm_params=llm_params,
            context_builder_params=context_builder_params or {},
        )
        self.system_prompt = system_prompt
        self.callbacks = callbacks
        self.response_type = response_type

    async def asearch(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        run_reduce_response: bool = True,
        **kwargs,
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user query."""
        start_time = time.time()
        search_prompt = ""

        logger.info(
            f"self.context_builder_params: {json.dumps(self.context_builder_params, indent=2)}"
        )

        context_text, context_records = await self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )
        if not run_reduce_response:
            log.info(
                "not `run_reduce_response` == True: return `SearchResult` with `response=context_text`"
            )
            return SearchResult(
                response=context_text,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=0,
                prompt_tokens=0,
                resources=None,
            )

        log.info("GENERATE ANSWER: %s. QUERY: %s", start_time, query)
        try:
            search_prompt = self.system_prompt.format(
                context_data=context_text, response_type=self.response_type
            )
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]

            response = await self.llm.agenerate(
                messages=search_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            return SearchResult(
                response=response,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                resources=None,
            )

        except Exception:
            log.exception("Exception in _asearch")
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                resources=None,
            )

    def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        run_reduce_response: bool = True,
        **kwargs,
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user question."""
        start_time = time.time()
        search_prompt = ""
        context_text, context_records = asyncio.run(
            self.context_builder.build_context(
                query=query,
                conversation_history=conversation_history,
                **kwargs,
                **self.context_builder_params,
            )
        )
        if not run_reduce_response:
            log.info(
                "not `run_reduce_response` == True: return `SearchResult` with `response=context_text`"
            )
            return SearchResult(
                response=context_text,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=0,
                prompt_tokens=0,
                resources=None,
            )

        log.info("GENERATE ANSWER: %d. QUERY: %s", start_time, query)
        try:
            search_prompt = self.system_prompt.format(
                context_data=context_text, response_type=self.response_type
            )
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]

            response = self.llm.generate(
                messages=search_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            return SearchResult(
                response=response,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                resources=None,
            )

        except Exception:
            log.exception("Exception in _map_response_single_batch")
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
                resources=None,
            )
