from __future__ import annotations

import json
import os
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import (
    Generator,
    Optional,
)
from typing import List, Dict, Any

import aioredis
from langchain_community.callbacks.openai_info import (
    standardize_model_name,
    MODEL_COST_PER_1K_TOKENS,
    get_openai_token_cost_for_model,
)
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook

from libs.python.schemas.basic_models import LLMModel
from libs.python.schemas.events import (
    Event,
    ServiceType,
    PubSubStatusChannel,
    AgentEvent,
    MessageChannel,
)

from libs.python.utils.logger import logger
from libs.python.utils.token_managment import TokenManager

REDIS_CLIENT_CALLBACK = aioredis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=os.getenv("REDIS_PORT", 6379),
    password=os.getenv("REDIS_PASSWORD_LLM", "redispassword"),
    db=os.getenv("REDIS_DB_PUB_SUB", 2),
    decode_responses=True,
)


START_RETRIEVE_INFO = "INFORMATION THAT YOU HAVE TO ANALYSE AND USE: \n\n"
NO_FOUND_MESSAGE_RETRIEVE_INFO = "There is no knowledge data found!"


class BaseUsageCallbackHandler(AsyncCallbackHandler):
    started_datetime: datetime = None
    output_tool: dict = dict()
    tool_exec: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0
    output_cost: float = 0
    total_cost: float = 0
    model_name: str = ""
    pubsub_channel: str = None

    def __init__(self, pubsub_channel: str, started_datetime: datetime) -> None:
        super().__init__()
        self.pubsub_channel = pubsub_channel
        self.started_datetime = started_datetime

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""
        diff = datetime.now() - self.started_datetime
        message_to_channel = MessageChannel(
            event=Event(
                status=PubSubStatusChannel.NOTIFICATION,
                message=AgentEvent.GOTO_TOOL_TO_KNOWLEDGE_MANAGER,
                service_type=ServiceType.AGENT,
                time_exe_from_start=diff.seconds + (diff.microseconds // 1000) / 1000,
                started_datetime=self.started_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                executed_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            ),
            response={"serialized": serialized, "input_str": input_str},
        )
        await REDIS_CLIENT_CALLBACK.publish(
            channel=self.pubsub_channel,
            message=message_to_channel.json().encode("utf8"),
        )

    async def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        self.tool_exec = True
        failure = False

        if (START_RETRIEVE_INFO in output) and (
            output != NO_FOUND_MESSAGE_RETRIEVE_INFO
        ):
            if output.startswith(START_RETRIEVE_INFO):
                output = output[len(START_RETRIEVE_INFO) :]
                try:
                    self.output_tool = json.loads(output)
                except json.JSONDecodeError as e:
                    failure = True
                    exc = traceback.format_exc()
                    logger.exception(exc)
                    diff = datetime.now() - self.started_datetime
                    message_to_channel = MessageChannel(
                        event=Event(
                            status=PubSubStatusChannel.FAILURE,
                            message=exc,
                            service_type=ServiceType.AGENT,
                            time_exe_from_start=diff.seconds
                            + (diff.microseconds // 1000) / 1000,
                            started_datetime=self.started_datetime.strftime(
                                "%Y-%m-%d %H:%M:%S.%f"
                            ),
                            executed_datetime=datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S.%f"
                            ),
                        ),
                        response={"error": str(e)},
                    )
                    await REDIS_CLIENT_CALLBACK.publish(
                        channel=self.pubsub_channel,
                        message=message_to_channel.json().encode("utf8"),
                    )

        if not failure:
            diff = datetime.now() - self.started_datetime
            message_to_channel = MessageChannel(
                event=Event(
                    status=PubSubStatusChannel.NOTIFICATION,
                    message=AgentEvent.RECEIVED_RESULTS_FROM_TOOL_TO_KNOWLEDGE_MANAGER,
                    service_type=ServiceType.AGENT,
                    time_exe_from_start=diff.seconds
                    + (diff.microseconds // 1000) / 1000,
                    started_datetime=self.started_datetime.strftime(
                        "%Y-%m-%d %H:%M:%S.%f"
                    ),
                    executed_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                ),
                response=self.output_tool,
            )
            await REDIS_CLIENT_CALLBACK.publish(
                channel=self.pubsub_channel,
                message=message_to_channel.json().encode("utf8"),
            )


class UsageOpenAICallbackHandler(BaseUsageCallbackHandler):
    """Callback Handler that tracks OpenAI info."""

    def __init__(
        self,
        pubsub_channel: str,
        started_datetime: datetime,
        model_name: str = "",
    ) -> None:
        super().__init__(
            pubsub_channel=pubsub_channel, started_datetime=started_datetime
        )
        self.model_name = model_name

    def __repr__(self) -> str:
        return (
            f"Total Tokens: {self.total_tokens}\n"
            f"\tInput Tokens: {self.input_tokens}\n"
            f"\tOutput Tokens: {self.output_tokens}\n"
            f"Total Cost (USD): ${self.total_cost}\n"
            f"\tInput Cost (USD): ${self.input_cost}\n"
            f"\tOutput Cost (USD): ${self.output_cost}\n"
        )

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        formatted_prompts = "\n".join(prompts)
        logger.info(f"Prompt:\n{formatted_prompts}")
        self.input_tokens = TokenManager.num_tokens_from_string(
            formatted_prompts, model_name=self.model_name
        )

        model_name = standardize_model_name(self.model_name)
        if model_name in MODEL_COST_PER_1K_TOKENS:
            self.input_cost = get_openai_token_cost_for_model(
                model_name, self.input_tokens
            )

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print out the token."""
        pass

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""

        if len(response.generations) == 0:
            return None

        if len(response.generations[0]) == 0:
            return None

        if response.generations[0][0].type != "ChatGenerationChunk":
            return None

        self.output_tokens = TokenManager.num_tokens_from_string(
            response.generations[0][0].text, model_name=self.model_name
        )

        model_name = standardize_model_name(self.model_name)
        if model_name in MODEL_COST_PER_1K_TOKENS:
            self.output_cost = get_openai_token_cost_for_model(
                model_name, self.output_tokens, is_completion=True
            )
        else:
            self.output_cost = 0

        self.total_tokens = self.input_tokens + self.output_tokens
        self.total_cost = self.input_cost + self.output_cost
        logger.info(self)

    def __copy__(self) -> "UsageOpenAICallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "UsageOpenAICallbackHandler":
        """Return a deep copy of the callback handler."""
        return self


usage_openai_callback_var: ContextVar[Optional[UsageOpenAICallbackHandler]] | None = (
    ContextVar("usage_openai_callback", default=None)
)


register_configure_hook(usage_openai_callback_var, True)


@contextmanager
def get_usage_openai_callback(
    model_name: str,
    started_datetime: datetime,
    pubsub_channel: str = None,
) -> Generator[UsageOpenAICallbackHandler, None, None]:
    """Get the OpenAI callback handler in a context manager.
    which conveniently exposes token and cost information.

    Returns:
        UsageOpenAICallbackHandler: The OpenAI callback handler.

    Example:
        >>> with get_usage_openai_callback() as cb:
        ...     # Use the OpenAI callback handler
    """
    cb = UsageOpenAICallbackHandler(
        model_name=model_name,
        pubsub_channel=pubsub_channel,
        started_datetime=started_datetime,
    )
    usage_openai_callback_var.set(cb)
    yield cb
    usage_openai_callback_var.set(None)


MAP_USAGE_CALLBACK = {
    LLMModel.GPT_4O_MINI: get_usage_openai_callback,
    LLMModel.GPT_4O: get_usage_openai_callback,
    LLMModel.GPT_4_0613: get_usage_openai_callback,
    LLMModel.GPT_35_TURBO_16k: get_usage_openai_callback,
}
