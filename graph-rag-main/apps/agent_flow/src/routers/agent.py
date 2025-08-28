import json
import os
import time
import traceback
import uuid
from datetime import datetime
from typing import Union, Dict, Optional, Any, List

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from langchain_core.messages import AIMessage, HumanMessage

from apps.agent_flow.src.cat_main_agent.agent_chain import CarAgentMain, PromptManager
from apps.agent_flow.src.chains import FinalAgentAnswerSummarizer
from apps.agent_flow.src.gen_next_questions_chain.gen_next_questions_chain import (
    NextRelatedUserStepsGenerator,
)
from apps.agent_flow.src.redis_helper import REDIS_CLIENT_AGENT  # , redis_subscriber
from apps.agent_flow.src.schemas.schema_agent import (
    ResponseAgentProcessUserMessage,
    RequestAgentProcessUserMessage,
    ResponseGetConfig,
    ResponseAgentProcessUserMessageChannel,
    ResponseAgentProcessUserMessageDefault,
    AgentMetricsSummary,
    ResponseAgentProcessUserPublishChannel,
)
from apps.agent_flow.src.utils import convert_dialogs_to_chat_history_lc
from libs.python.schemas.basic_models import TOKEN_THRESHOLDS, TOKEN_PERCENTAGE_FOR_RAG
from libs.python.schemas.config_presets import get_config, ConfigPreSet
from libs.python.schemas.configuration import Config
from libs.python.schemas.events import (
    Event,
    PubSubStatusChannel,
    ServiceType,
    AgentEvent,
)
from libs.python.utils.logger import logger
from libs.python.utils.token_managment import TokenManager

router = APIRouter()


class AgentEventPublisher:
    """Handles publishing agent events to Redis channels."""

    @staticmethod
    async def send_event(
        channel_name: str,
        status_event: PubSubStatusChannel,
        message: str,
        agent_answer: str,
        metrics: Dict,
        started_datetime: datetime,
        generated_next_user_steps: List[str] = None,
        final_agent_answer_summarized: str = None,
        agent_full_executed_prompt: Optional[str] = None,
        agent_output_tool: Optional[Dict[str, Any]] = None,
    ):
        """Send an agent event to the specified Redis channel."""
        event = AgentEventPublisher._create_event(
            status_event, message, started_datetime
        )

        response = AgentEventPublisher._create_response(
            agent_answer=agent_answer,
            metrics=metrics,
            agent_full_executed_prompt=agent_full_executed_prompt,
            agent_output_tool=agent_output_tool,
            generated_next_user_steps=generated_next_user_steps,
            final_agent_answer_summarized=final_agent_answer_summarized,
        )

        message_to_channel = ResponseAgentProcessUserMessageChannel(
            event=event, response=response
        )

        await REDIS_CLIENT_AGENT.publish(
            channel=channel_name, message=message_to_channel.json().encode("utf8")
        )

        logger.info(
            f"PUBLISH MESSAGE AGENT: "
            f"channel_name: {channel_name}, "
            f"status_event: {status_event}, "
            f"message: {message}"
        )

    @staticmethod
    def _create_event(
        status: PubSubStatusChannel, message: str, started_datetime: datetime
    ) -> Event:
        """Create an Event object with timing information."""
        diff = datetime.now() - started_datetime
        return Event(
            status=status,
            message=message,
            service_type=ServiceType.AGENT,
            time_exe_from_start=diff.seconds + (diff.microseconds // 1000) / 1000,
            started_datetime=started_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
            executed_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        )

    @staticmethod
    def _create_response(
        agent_answer: str,
        metrics: Dict,
        agent_full_executed_prompt: Optional[str] = None,
        agent_output_tool: Optional[Dict[str, Any]] = None,
        generated_next_user_steps: Optional[List[str]] = None,
        final_agent_answer_summarized: Optional[str] = None,
    ) -> ResponseAgentProcessUserMessageDefault:
        """Create a response object with agent execution results."""
        return ResponseAgentProcessUserMessageDefault(
            agent_answer=agent_answer,
            metrics=AgentMetricsSummary(**metrics),
            agent_full_executed_prompt=agent_full_executed_prompt,
            agent_output_tool=agent_output_tool,
            generated_next_user_steps=generated_next_user_steps,
            final_agent_answer_summarized=final_agent_answer_summarized,
        )


# Create a singleton instance for backward compatibility
_agent_event_publisher = AgentEventPublisher()


# Keep the original function for backward compatibility
async def send_message_agent_event(**kwargs):
    """Backward compatible wrapper for AgentEventPublisher."""
    await _agent_event_publisher.send_event(**kwargs)


async def execute_agent(channel_name: str, request: RequestAgentProcessUserMessage):
    started_datetime = datetime.now()
    main_start_process = time.time()
    generated_next_user_steps = None
    final_agent_answer_summarized = None

    # fake metrics
    metrics = {
        "additional_metrics": {},
        "time_exe": 0,
        "agent_flow": {
            "additional_metrics": {},
            "time_exe": 0,
            "processes": [],
        },
        "knowledge_manager": {
            "additional_metrics": {},
            "time_exe": 0,
            "processes": [
                {
                    "additional_metrics": {},
                    "time_exe": 0,
                    "name": "vector_semantic_retrieval",
                    "type_process": "retrieval_process",
                },
                {
                    "additional_metrics": {},
                    "time_exe": 0,
                    "name": "vector_record",
                    "type_process": "building_process",
                },
            ],
            "knowledge_depth": 0,
            "knowledge_width": 0,
        },
        "content_scraper": {
            "additional_metrics": {},
            "time_exe": 0,
            "processes": [
                {
                    "additional_metrics": {},
                    "time_exe": 0,
                    "name": "website_content_scrapping_playwright",
                    "type_process": "parsing_process",
                }
            ],
        },
    }
    try:
        agent_full_executed_prompt = ""
        output_tool = dict()

        await AgentEventPublisher.send_event(
            channel_name=channel_name,
            status_event=PubSubStatusChannel.STARTED,
            message=PubSubStatusChannel.STARTED.value,
            agent_answer="",
            metrics=metrics,
            agent_full_executed_prompt=(
                agent_full_executed_prompt
                if request.return_agent_full_executed_prompt
                else None
            ),
            agent_output_tool=(
                output_tool if request.return_agent_tool_output else None
            ),
            started_datetime=started_datetime,
        )

        logger.info(
            f"[user_id={request.user_id}] Received request to Process User Message;"
        )
        logger.info(
            f"[user_id={request.user_id}] return_agent_full_executed_prompt={request.return_agent_full_executed_prompt};"
        )

        if request.config_preset_name is not None:
            config = get_config(preset=request.config_preset_name)
            logger.info(
                f"[user_id={request.user_id}] Config loaded from PRESET config: \n"
                f"config_preset_name={request.config_preset_name};\n"
            )
        elif request.config is not None:
            logger.info(
                f"[user_id={request.user_id}] Config loaded from REQUEST FIELD PAYLOAD;"
            )
            config = request.config
        else:
            detail = "Payload must contain a 'request.config_preset_name' OR 'request.config' field as not None"
            await send_message_agent_event(
                channel_name=channel_name,
                metrics=metrics,
                status_event=PubSubStatusChannel.FAILURE,
                message=detail,
                started_datetime=started_datetime,
                agent_answer="",
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail,
            )

        logger.info(f"[user_id={request.user_id}] Config={config};")

        sys_prompt = config.config_agent_flow.sys_prompt

        st = time.time()

        prompt_manager = PromptManager(
            knowledge_manager_mode=config.config_knowledge_manager.mode,
            sys_prompt=sys_prompt,
        )

        logger.info(
            f"Init CarAgentMain as car_agent_main in `process_user_message`, time: {time.time() - st} sec."
        )

        if request.dialog_content is not None:
            tokens_threshold = TOKEN_THRESHOLDS.get(config.config_agent_flow.llm_model)
            tokens_threshold = (1 - TOKEN_PERCENTAGE_FOR_RAG) * tokens_threshold

            dialog_content = request.dialog_content
            k = config.config_agent_flow.window_k_dialog_messages

            input_message, chat_history = await convert_dialogs_to_chat_history_lc(
                dialog_content=dialog_content, k=k
            )

            agent_full_executed_prompt: str = await prompt_manager.format_prompt(
                input_message=input_message,
                chat_history=chat_history,
            )

            len_agent_full_executed_prompt_zero = TokenManager.num_tokens_from_string(
                agent_full_executed_prompt,
                model_name=config.config_agent_flow.llm_model,
            )

            tokens_agent_full_executed_prompt = len_agent_full_executed_prompt_zero
            while tokens_threshold < tokens_agent_full_executed_prompt:
                logger.info(
                    f"Detect token limit for {config.config_agent_flow.llm_model.value}: "
                    f"tokens_threshold[{tokens_threshold}] < "
                    f"tokens_agent_full_executed_prompt[{tokens_agent_full_executed_prompt}]"
                )
                if len(dialog_content) > 1:
                    dialog_content = dialog_content[1:]
                else:
                    break

                input_message, chat_history = await convert_dialogs_to_chat_history_lc(
                    dialog_content=dialog_content, k=k
                )

                agent_full_executed_prompt: str = await prompt_manager.format_prompt(
                    input_message=input_message,
                    chat_history=chat_history,
                )
                tokens_agent_full_executed_prompt = TokenManager.num_tokens_from_string(
                    agent_full_executed_prompt,
                    model_name=config.config_agent_flow.llm_model,
                )

            tokens_agent_full_executed_prompt = TokenManager.num_tokens_from_string(
                agent_full_executed_prompt,
                model_name=config.config_agent_flow.llm_model,
            )
            if tokens_threshold > tokens_agent_full_executed_prompt:
                logger.info(
                    f"Token limit is OK: "
                    f"tokens_threshold[{tokens_threshold}] > "
                    f"tokens_agent_full_executed_prompt[{tokens_agent_full_executed_prompt}]"
                )
            else:
                logger.info(
                    f"Detect token limit for {config.config_agent_flow.llm_model.value}: "
                    f"tokens_threshold[{tokens_threshold}] <= "
                    f"tokens_agent_full_executed_prompt[{tokens_agent_full_executed_prompt}]"
                )

            count_agent_input_tokens_exceeded = (
                len_agent_full_executed_prompt_zero - tokens_agent_full_executed_prompt
            )

            car_agent_main = CarAgentMain(
                user_id=request.user_id,
                llm_model=config.config_agent_flow.llm_model,
                tool_get_info_config=config,
                pubsub_channel_name=channel_name,
                started_datetime=started_datetime,
                prompt_manager=prompt_manager,
                chat_history=chat_history,
                input_message=input_message,
            )

            (
                agent_answer,
                usage_model_metrics_info,
                output_tool,
                context_data_from_tool,
            ) = await car_agent_main.inference(
                input=input_message,
                chat_history=chat_history,
                count_agent_input_tokens_exceeded=count_agent_input_tokens_exceeded,
            )
            metrics["agent_flow"]["processes"].append(
                json.loads(usage_model_metrics_info.json())
            )
            await send_message_agent_event(
                channel_name=channel_name,
                status_event=PubSubStatusChannel.UPDATED,
                message=f"===[PARTIAL RESULT field: 'agent_answer']{AgentEvent.MAIN_AGENT_ANSWER.value}===\n{agent_answer[:250]}......",
                agent_answer=agent_answer,
                metrics=metrics,
                agent_full_executed_prompt=(
                    agent_full_executed_prompt
                    if request.return_agent_full_executed_prompt
                    else None
                ),
                agent_output_tool=(
                    output_tool if request.return_agent_tool_output else None
                ),
                started_datetime=started_datetime,
            )

            logger.info(
                f"IS Next Related Generation Available: {config.config_agent_flow.next_gen_user_steps.available}?"
            )

            if config.config_agent_flow.next_gen_user_steps.available and (
                ((len(chat_history) + 2) > 7) or (context_data_from_tool != "")
            ):
                logger.info(f"START NextRelatedUserStepsGenerator!")
                generator_next_user_steps = NextRelatedUserStepsGenerator(
                    config=config,
                    pubsub_channel_name=channel_name,
                    started_datetime=started_datetime,
                )

                chat_history.append(HumanMessage(content=input_message))
                chat_history.append(AIMessage(content=agent_answer))
                generated_next_user_steps, usage_model_metrics_info = (
                    await generator_next_user_steps.generate(
                        context_data=context_data_from_tool,
                        chat_history=chat_history,
                        chat_history_sys_prompt=prompt_manager.sys_prompt,
                    )
                )
                metrics["agent_flow"]["processes"].append(
                    json.loads(usage_model_metrics_info.json())
                )
                formatted_generated_next_user_steps = "\n".join(
                    (
                        f"{_num_ + 1}. {_}"
                        for _num_, _ in enumerate(
                            generated_next_user_steps.generated_next_potential_user_steps
                        )
                    )
                )
                await send_message_agent_event(
                    channel_name=channel_name,
                    status_event=PubSubStatusChannel.UPDATED,
                    # message=AgentEvent.GENERATED_NEXT_QUESTIONS,
                    message=f"===[PARTIAL RESULT field: 'generated_next_user_steps']{AgentEvent.GENERATED_NEXT_USER_STEPS.value}===\n{formatted_generated_next_user_steps}",
                    agent_answer=agent_answer,
                    metrics=metrics,
                    agent_full_executed_prompt=(
                        agent_full_executed_prompt
                        if request.return_agent_full_executed_prompt
                        else None
                    ),
                    agent_output_tool=(
                        output_tool if request.return_agent_tool_output else None
                    ),
                    generated_next_user_steps=generated_next_user_steps.generated_next_potential_user_steps,
                    started_datetime=started_datetime,
                )
                logger.info(f"FINISH NextRelatedQuestionsGenerator!")

            logger.info(
                f"IS `FinalAgentAnswerSummarizer` Available: {config.config_agent_flow.summarizer_final_answer_setup.available}?"
            )

            agent_answer_chars_len = len(agent_answer)
            summarizer_chars_threshold = (
                config.config_agent_flow.summarizer_final_answer_setup.chars_threshold
            )
            summarizer_condition = summarizer_chars_threshold < agent_answer_chars_len

            logger.info(
                f"[FinalAgentAnswerSummarizer] Check if answer above chars_threshold. "
                f"agent_answer_chars_len={agent_answer_chars_len} {'<' if summarizer_condition else '=>'} "
                f"summarizer_chars_threshold={summarizer_chars_threshold}"
            )

            if (
                config.config_agent_flow.summarizer_final_answer_setup.available
                and summarizer_condition
            ):
                logger.info(f"START `FinalAgentAnswerSummarizer`!")
                final_agent_answer_summarizer = FinalAgentAnswerSummarizer(
                    config=config,
                    pubsub_channel_name=channel_name,
                    started_datetime=started_datetime,
                )
                final_agent_answer_summarized, usage_model_metrics_info_summarized = (
                    await final_agent_answer_summarizer.generate(
                        final_agent_answer=agent_answer,
                    )
                )
                metrics["agent_flow"]["processes"].append(
                    json.loads(usage_model_metrics_info_summarized.json())
                )
                await send_message_agent_event(
                    channel_name=channel_name,
                    status_event=PubSubStatusChannel.UPDATED,
                    # message=AgentEvent.GENERATED_NEXT_QUESTIONS,
                    message=f"===[PARTIAL RESULT field: 'final_agent_answer_summarized'][Reduced from {agent_answer_chars_len} to {len(final_agent_answer_summarized)}]{AgentEvent.SUMMARIZED_FINAL_AGENT_ANSWER.value}===\n{final_agent_answer_summarized}",
                    agent_answer=agent_answer,
                    metrics=metrics,
                    agent_full_executed_prompt=(
                        agent_full_executed_prompt
                        if request.return_agent_full_executed_prompt
                        else None
                    ),
                    agent_output_tool=(
                        output_tool if request.return_agent_tool_output else None
                    ),
                    generated_next_user_steps=generated_next_user_steps.generated_next_potential_user_steps,
                    final_agent_answer_summarized=final_agent_answer_summarized,
                    started_datetime=started_datetime,
                )
                logger.info(f"FINISH FinalAgentAnswerSummarizer!")

            await send_message_agent_event(
                channel_name=channel_name,
                status_event=PubSubStatusChannel.SUCCESS,
                message=PubSubStatusChannel.SUCCESS.value,
                agent_answer=agent_answer,
                metrics=metrics,
                agent_full_executed_prompt=(
                    agent_full_executed_prompt
                    if request.return_agent_full_executed_prompt
                    else None
                ),
                agent_output_tool=(
                    output_tool if request.return_agent_tool_output else None
                ),
                generated_next_user_steps=(
                    generated_next_user_steps.generated_next_potential_user_steps
                    if generated_next_user_steps is not None
                    else generated_next_user_steps
                ),
                started_datetime=started_datetime,
            )

            # token_len_agent_full_executed_prompt = TokenManager.num_tokens_from_string(
            #     string=agent_full_executed_prompt
            # )
            # logger.info(
            #     f"Token count in agent full executed prompt: {token_len_agent_full_executed_prompt};"
            # )
        elif isinstance(request.dialog_id, str) and request.dialog_id is not None:
            # TODO: Process **dialog_id**
            # content = ""  # process request.dialog_id
            # agent_answer = await car_agent_main.inference(input=content)
            agent_answer = ""
        else:
            detail = "Payload must contain a 'request.dialog_content' OR 'request.dialog_id' field as not None"
            await send_message_agent_event(
                channel_name=channel_name,
                metrics=metrics,
                status_event=PubSubStatusChannel.FAILURE,
                message=detail,
                started_datetime=started_datetime,
                agent_answer="",
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=detail,
            )

    except Exception as e:

        exc = traceback.format_exc()
        logger.exception(exc)
        metrics["time_exe"] = time.time() - main_start_process

        await send_message_agent_event(
            channel_name=channel_name,
            metrics=metrics,
            status_event=PubSubStatusChannel.FAILURE,
            message=exc,
            started_datetime=started_datetime,
            agent_answer="",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    time_exe = time.time() - main_start_process
    logger.info(f"Processed User Message, time: {time_exe} sec.")
    return (
        agent_answer,
        metrics,
        agent_full_executed_prompt,
        output_tool,
        generated_next_user_steps,
        final_agent_answer_summarized,
    )


@router.post(
    "/process_user_message",
    response_model=Union[
        ResponseAgentProcessUserMessage, ResponseAgentProcessUserPublishChannel
    ],
)
async def process_user_message(
    request: RequestAgentProcessUserMessage, background_tasks: BackgroundTasks
):
    agent_events = []
    if request.pubsub_channel_name is None:
        channel_name = f"AGENT-PROCESS-USER-MESSAGE--{str(uuid.uuid4())}"
    else:
        channel_name = request.pubsub_channel_name

    if request.as_task:
        background_tasks.add_task(
            execute_agent, channel_name=channel_name, request=request
        )
        return {
            "pubsub_channel_name": channel_name,
        }
    else:
        # background_tasks.add_task(
        #     redis_subscriber, channel_name=channel_name, collected_events=agent_events
        # )
        (
            agent_answer,
            metrics,
            agent_full_executed_prompt,
            output_tool,
            generated_next_user_steps,
            final_agent_answer_summarized,
        ) = await execute_agent(channel_name=channel_name, request=request)

        return {
            "agent_answer": agent_answer,
            "metrics": metrics,
            "agent_full_executed_prompt": (
                agent_full_executed_prompt
                if request.return_agent_full_executed_prompt
                else None
            ),
            "agent_output_tool": (
                output_tool if request.return_agent_tool_output else None
            ),
            "agent_events": agent_events,
            "generated_next_user_steps": generated_next_user_steps,
            "final_agent_answer_summarized": final_agent_answer_summarized,
        }


@router.get("/config/{config_name}", response_model=ResponseGetConfig)
async def get_config_endpoint(config_name: ConfigPreSet):
    logger.info(f"Received `get_config_endpoint` with config_name={config_name}")
    config2return: Config = get_config(preset=config_name)
    return {
        "config_name": config_name,
        "config": config2return,
        "commit_hash": os.getenv("COMMIT_HASH", ""),
    }
