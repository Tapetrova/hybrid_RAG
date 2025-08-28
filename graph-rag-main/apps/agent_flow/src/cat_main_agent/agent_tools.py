import asyncio
import json
import os
from typing import Dict, Any, Tuple, Optional
from urllib.parse import urljoin

from libs.python.prompts.graphrag.global_search.reduce_system_prompt import (
    NO_DATA_ANSWER,
    REDUCE_SYSTEM_PROMPT_RULE,
)
from libs.python.prompts.graphrag.local_search.system_prompt import (
    LOCAL_SEARCH_SUPPORT_INSTRUCTIONS_AS_TOOL_OUTPUT,
)
from libs.python.schemas.basic_models import (
    TOKEN_PERCENTAGE_FOR_RAG,
    TOKEN_THRESHOLDS,
    PERCENTAGE_FROM_TOKEN_RAG_SLICE_TOOL_RESPONSE,
    LLMModel,
)
from libs.python.schemas.configuration import (
    Config,
)
from libs.python.schemas.graphrag.graph_rag_config import SearchMode
from libs.python.utils.cacher import acache_data
from libs.python.utils.callbacks import (
    START_RETRIEVE_INFO,
    NO_FOUND_MESSAGE_RETRIEVE_INFO,
)
from libs.python.utils.logger import logger
from libs.python.utils.request_utils import asend_request, send_request
from libs.python.utils.token_managment import TokenManager, hash_string

# logger.info(f"os.getenv(ENDPOINT_RETRIEVEMENT): {os.getenv('ENDPOINT_RETRIEVEMENT')}")
ENDPOINT_RETRIEVEMENT = os.getenv(
    "ENDPOINT_RETRIEVEMENT", "http://localhost:8098/retrievement/vector_retrieve"
)
BASEURL_RETRIEVEMENT = os.getenv("BASEURL_RETRIEVEMENT", "http://localhost:8098")
ENDPOINT_RETRIEVEMENT_GRAPHRAG = urljoin(
    BASEURL_RETRIEVEMENT, "/retrievement/graphrag/retrieve"
)


# logger.info(f"ENDPOINT_RETRIEVEMENT: {ENDPOINT_RETRIEVEMENT}")


# Helper functions to reduce code duplication
def _build_request_data(
    natural_query: str, user_id: str, pubsub_channel_name: str, config: Config
) -> Dict[str, Any]:
    """Build request data for API calls."""
    return {
        "natural_query": natural_query,
        "pubsub_channel_name": pubsub_channel_name,
        "user_id": user_id,
        "session_id": f"session_{user_id}",
        "dialog_id": f"dialog_{user_id}",
        "config": json.loads(config.json()),
    }


async def _fetch_knowledge_content(
    request_data: Dict[str, Any], endpoint: str, caller_name: str
) -> Dict[str, Any]:
    """Fetch knowledge content from API asynchronously."""
    response = await asend_request(request_data, endpoint=endpoint)
    if response is None:
        return {}

    response_dict = response.json()
    logger.info(
        f"[AGENT-CHAIN-TOOL:{caller_name}] Response keys: {list(response_dict.keys())}"
    )
    return response_dict


def _fetch_knowledge_content_sync(
    request_data: Dict[str, Any], endpoint: str, caller_name: str
) -> Dict[str, Any]:
    """Fetch knowledge content from API synchronously."""
    response = send_request(request_data, endpoint=endpoint)
    if response is None:
        return {}

    response_dict = response.json()
    logger.info(
        f"[AGENT-CHAIN-TOOL:{caller_name}] Response keys: {list(response_dict.keys())}"
    )
    return response_dict


def _extract_knowledge_content(
    response_dict: Dict[str, Any]
) -> Dict[str, Dict[str, str]]:
    """Extract knowledge content from response."""
    knowledge_content_json = {}

    if "knowledge_content" in response_dict:
        knowledge_content = response_dict.get("knowledge_content", [])
        logger.info(
            f"[AGENT-CHAIN-TOOL] Found {len(knowledge_content)} knowledge items"
        )

        for i, content_item in enumerate(knowledge_content):
            text = content_item.get("text", "")
            source = content_item.get("src", "")

            logger.info(
                f"[AGENT-CHAIN-TOOL] Item {i + 1}: "
                f"Content length: {len(text)}, Source: {source}"
            )

            knowledge_content_json[
                f"Information_that_you_have_to_analyse_and_use_{i + 1}"
            ] = {
                "Content": text,
                "Source": source,
            }

    return knowledge_content_json


def _format_knowledge_content(knowledge_content_json: Dict[str, Dict[str, str]]) -> str:
    """Format knowledge content for output."""
    return (
        f"{START_RETRIEVE_INFO}"
        f"{json.dumps(knowledge_content_json, indent=2, ensure_ascii=False)}\n"
    )


async def _manage_token_limits(
    knowledge_content_json: Dict[str, Dict[str, str]],
    formatted_content: str,
    llm_model: LLMModel,
) -> Tuple[str, Dict[str, Any]]:
    """Manage token limits by truncating content if necessary."""
    initial_tokens = TokenManager.num_tokens_from_string(
        formatted_content, model_name=llm_model
    )
    token_threshold = TOKEN_PERCENTAGE_FOR_RAG * TOKEN_THRESHOLDS.get(llm_model)
    slice_amount = int(PERCENTAGE_FROM_TOKEN_RAG_SLICE_TOOL_RESPONSE * token_threshold)

    current_tokens = initial_tokens
    truncated_json = knowledge_content_json.copy()

    while token_threshold < current_tokens:
        logger.info(
            f"[TOOL] Token limit exceeded for {llm_model.value}: "
            f"threshold[{token_threshold}] < current[{current_tokens}]"
        )

        # Truncate content
        truncated_json = {
            key: {
                "Content": (
                    val["Content"][:-slice_amount]
                    if len(val["Content"]) > slice_amount
                    else ""
                ),
                "Source": val["Source"],
            }
            for key, val in truncated_json.items()
            if len(val["Content"]) > slice_amount
        }

        if not truncated_json:
            logger.info("[TOOL] All content truncated - no data remaining")
            return NO_FOUND_MESSAGE_RETRIEVE_INFO, {
                "tokens_tool_executed_prompt": 0,
                "count_agent_tool_input_tokens_exceeded": initial_tokens,
                "tokens_tool_executed_prompt_zero": initial_tokens,
                "percentage_from_token_rag_slice_tool_response": PERCENTAGE_FROM_TOKEN_RAG_SLICE_TOOL_RESPONSE,
                "tool_tokens_threshold": token_threshold,
            }

        formatted_content = _format_knowledge_content(truncated_json)
        current_tokens = TokenManager.num_tokens_from_string(
            formatted_content, model_name=llm_model
        )

    return formatted_content, {
        "tokens_tool_executed_prompt": current_tokens,
        "count_agent_tool_input_tokens_exceeded": initial_tokens - current_tokens,
        "tokens_tool_executed_prompt_zero": initial_tokens,
        "percentage_from_token_rag_slice_tool_response": PERCENTAGE_FROM_TOKEN_RAG_SLICE_TOOL_RESPONSE,
        "tool_tokens_threshold": token_threshold,
    }


def _manage_token_limits_sync(
    knowledge_content_json: Dict[str, Dict[str, str]],
    formatted_content: str,
    llm_model: LLMModel,
) -> Tuple[str, Dict[str, Any]]:
    """Synchronous version of _manage_token_limits."""
    # Use the async version synchronously since it doesn't have any async operations
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _manage_token_limits(knowledge_content_json, formatted_content, llm_model)
        )
    finally:
        loop.close()


async def _cache_tool_metrics(
    pubsub_channel_name: str, content: str, metrics: Dict[str, Any]
) -> None:
    """Cache tool execution metrics."""
    content_hash = hash_string(content)
    await acache_data(
        f"cache_helper_{pubsub_channel_name}_{content_hash}",
        data=metrics,
        ex=7200,
    )


def _cache_tool_metrics_sync(
    pubsub_channel_name: str, content: str, metrics: Dict[str, Any]
) -> None:
    """Cache tool execution metrics synchronously."""
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _cache_tool_metrics(pubsub_channel_name, content, metrics)
        )
    finally:
        loop.close()


def _format_graph_rag_answer(graph_rag_answer: str, config: Config) -> str:
    """Format GraphRAG answer based on search mode."""
    search_mode = config.config_knowledge_manager.config_retrieval_mode.search_mode

    if search_mode == SearchMode.global_mode:
        if graph_rag_answer != NO_DATA_ANSWER:
            response_type = (
                config.config_knowledge_manager.config_retrieval_mode.global_search.response_type
            )
            return REDUCE_SYSTEM_PROMPT_RULE.format(
                report_data=graph_rag_answer, response_type=response_type
            )
        else:
            logger.info(f"GraphRAG returned NO_DATA_ANSWER: {NO_DATA_ANSWER}")
            return graph_rag_answer
    else:
        response_type = (
            config.config_knowledge_manager.config_retrieval_mode.local_search.response_type
        )
        return LOCAL_SEARCH_SUPPORT_INSTRUCTIONS_AS_TOOL_OUTPUT.format(
            context_data=graph_rag_answer, response_type=response_type
        )


async def _cache_graph_rag_metrics(
    pubsub_channel_name: str, answer: str, response: Optional[Dict[str, Any]]
) -> None:
    """Cache GraphRAG metrics."""
    answer_hash = hash_string(answer)
    await acache_data(
        f"cache_helper_{pubsub_channel_name}_{answer_hash}",
        data={
            "tokens_tool_executed_prompt": 0.0,
            "count_agent_tool_input_tokens_exceeded": 0.0,
            "tokens_tool_executed_prompt_zero": 0.0,
            "percentage_from_token_rag_slice_tool_response": 0.05,
            "tool_tokens_threshold": 0.0,
            "graph_rag_answer_response": response,
        },
        ex=7200,
    )


def _cache_graph_rag_metrics_sync(
    pubsub_channel_name: str, answer: str, response: Optional[Dict[str, Any]]
) -> None:
    """Cache GraphRAG metrics synchronously."""
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _cache_graph_rag_metrics(pubsub_channel_name, answer, response)
        )
    finally:
        loop.close()


async def aget_information_from_internet(
    natural_query: str,
    user_id: str,
    pubsub_channel_name: str,
    config: Config,
) -> str:
    request_data = _build_request_data(
        natural_query, user_id, pubsub_channel_name, config
    )
    response_dict = await _fetch_knowledge_content(
        request_data, ENDPOINT_RETRIEVEMENT, "aget_information_from_internet"
    )
    knowledge_content_retrieve_json = _extract_knowledge_content(response_dict)

    if not knowledge_content_retrieve_json:
        return NO_FOUND_MESSAGE_RETRIEVE_INFO

    knowledge_content_retrieve = _format_knowledge_content(
        knowledge_content_retrieve_json
    )

    # Token management and truncation
    truncated_content, metrics = await _manage_token_limits(
        knowledge_content_retrieve_json,
        knowledge_content_retrieve,
        config.config_agent_flow.llm_model,
    )

    # Cache the results
    await _cache_tool_metrics(pubsub_channel_name, truncated_content, metrics)

    return truncated_content


def get_information_from_internet(
    natural_query: str,
    user_id: str,
    pubsub_channel_name: str,
    config: Config,
) -> str:
    request_data = _build_request_data(
        natural_query, user_id, pubsub_channel_name, config
    )
    response_dict = _fetch_knowledge_content_sync(
        request_data, ENDPOINT_RETRIEVEMENT, "get_information_from_internet"
    )
    knowledge_content_retrieve_json = _extract_knowledge_content(response_dict)

    if not knowledge_content_retrieve_json:
        return NO_FOUND_MESSAGE_RETRIEVE_INFO

    knowledge_content_retrieve = _format_knowledge_content(
        knowledge_content_retrieve_json
    )

    # Token management and truncation
    truncated_content, metrics = _manage_token_limits_sync(
        knowledge_content_retrieve_json,
        knowledge_content_retrieve,
        config.config_agent_flow.llm_model,
    )

    # Cache the results
    _cache_tool_metrics_sync(pubsub_channel_name, truncated_content, metrics)

    return truncated_content


async def aget_information_from_internet_graphrag(
    natural_query: str,
    user_id: str,
    pubsub_channel_name: str,
    config: Config,
    run_reduce_response: bool,
) -> str:
    request_data = _build_request_data(
        natural_query, user_id, pubsub_channel_name, config
    )
    response_dict = await _fetch_knowledge_content(
        request_data,
        ENDPOINT_RETRIEVEMENT_GRAPHRAG,
        "aget_information_from_internet_graphrag",
    )

    graph_rag_answer = ""
    graph_rag_answer_response = response_dict.get("graph_rag_answer")

    if graph_rag_answer_response and "response" in graph_rag_answer_response:
        graph_rag_answer = graph_rag_answer_response.get("response")

        if not run_reduce_response:
            graph_rag_answer = _format_graph_rag_answer(graph_rag_answer, config)

    # Cache the results
    await _cache_graph_rag_metrics(
        pubsub_channel_name, graph_rag_answer, graph_rag_answer_response
    )

    return graph_rag_answer


def get_information_from_internet_graphrag(
    natural_query: str,
    user_id: str,
    pubsub_channel_name: str,
    config: Config,
    run_reduce_response: bool,
) -> str:
    request_data = _build_request_data(
        natural_query, user_id, pubsub_channel_name, config
    )
    response_dict = _fetch_knowledge_content_sync(
        request_data,
        ENDPOINT_RETRIEVEMENT_GRAPHRAG,
        "get_information_from_internet_graphrag",
    )

    graph_rag_answer = ""
    graph_rag_answer_response = response_dict.get("graph_rag_answer")

    if graph_rag_answer_response and "response" in graph_rag_answer_response:
        graph_rag_answer = graph_rag_answer_response.get("response")

        if not run_reduce_response:
            graph_rag_answer = _format_graph_rag_answer(graph_rag_answer, config)

    # Cache the results
    _cache_graph_rag_metrics_sync(
        pubsub_channel_name, graph_rag_answer, graph_rag_answer_response
    )

    return graph_rag_answer
