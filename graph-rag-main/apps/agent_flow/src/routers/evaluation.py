import json
import os
import time
import traceback
import uuid
from datetime import datetime
from typing import List, Dict, Union

from fastapi import APIRouter, BackgroundTasks

from apps.agent_flow.src.eval_lib import (
    relevance_calc,
    consistency_calc,
    # plagiarism_calc,
    directness_calc,
    comprehensiveness_calc,
    diversity_calc,
    empowerment_calc,
)
from apps.agent_flow.src.eval_lib.plagiarism_calc import plagiarism_calc
from apps.agent_flow.src.redis_helper import REDIS_CLIENT_EVAL
from apps.agent_flow.src.schemas.schema_eval import (
    ResponseEvaluationScoreDialog,
    RequestEvaluationScoreDialog,
    VersionEvaluationScoreDialog,
    ResponseEvaluationScoreDialogPublishChannel,
    ResponseEvaluationScoreDialogMessageChannel,
    EvalMetricsSummary,
)
from libs.python.schemas.events import (
    PubSubStatusChannel,
    EvalMetricEvent,
    ServiceType,
    Event,
)
from libs.python.utils.logger import logger

MAX_WORKERS_PROMETHEUS = 2
from enum import Enum
from typing import Callable, Dict, Any, Optional, Tuple, List


class MetricName(str, Enum):
    """Enumeration of available metric names."""

    ACCURACY_FACT_KDB = "accuracy_compare_to_factKDB"
    PLAGIARISM = "plagiarism"
    CONSISTENCY = "consistency"
    RELEVANCE = "relevance"
    COMPREHENSIVENESS = "comprehensiveness"
    DIRECTNESS = "directness"
    DIVERSITY = "diversity"
    EMPOWERMENT = "empowerment"


METRIC_NAMES = [metric.value for metric in MetricName]

# Metric calculator registry
METRIC_CALCULATORS: Dict[str, Callable] = {
    MetricName.RELEVANCE: relevance_calc,
    MetricName.CONSISTENCY: consistency_calc,
    MetricName.PLAGIARISM: plagiarism_calc,
    MetricName.DIRECTNESS: directness_calc,
    MetricName.COMPREHENSIVENESS: comprehensiveness_calc,
    MetricName.DIVERSITY: diversity_calc,
    MetricName.EMPOWERMENT: empowerment_calc,
}

router = APIRouter()


def sliding_window(lst, n):
    for i in range(len(lst) - n + 1):
        yield lst[i : i + n]


async def send_metrics_evaluation(
    channel_name: str,
    metrics,
    version_of_eval_score_dialog: VersionEvaluationScoreDialog,
    event_status: PubSubStatusChannel,
    event_message: Union[str, EvalMetricEvent],
    started_datetime: datetime,
):
    logger.info(f"Sending metrics to channel: {channel_name}")
    diff = datetime.now() - started_datetime
    message_to_channel = ResponseEvaluationScoreDialogMessageChannel(
        event=Event(
            status=event_status,
            message=event_message,
            service_type=ServiceType.EVALUATION,
            time_exe_from_start=diff.seconds + (diff.microseconds // 1000) / 1000,
            started_datetime=started_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
            executed_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        response=ResponseEvaluationScoreDialog(
            metrics=EvalMetricsSummary(**metrics),
            version_of_eval_score_dialog=version_of_eval_score_dialog,
            commit_hash=os.getenv("COMMIT_HASH", ""),
        ),
    )
    try:
        logger.debug("Starting send_metrics_evaluation")
        await REDIS_CLIENT_EVAL.publish(
            channel=channel_name, message=message_to_channel.json().encode("utf8")
        )
        logger.debug("Completed send_metrics_evaluation")
    except Exception as e:
        logger.debug(f"An error occurred in send_metrics_evaluation: {e}")
    logger.info(
        f"PUBLISH MESSAGE: "
        f"    channel_name: {channel_name},"
        f"    metrics,"
        f"    version_of_eval_score_dialog: {version_of_eval_score_dialog},"
        f"    event_status: {event_status},"
        f"    event_message: {event_message},"
    )


async def calculate_metric(
    metric_name: str,
    instruction: str,
    response_answer: str,
    content_sources: Dict[str, Dict[str, str]],
    agent_executions_to_eval_consistency: List,
    llm_judge_config: Any,
) -> Tuple[Optional[float], Dict[str, Any], List, float, Optional[str]]:
    """Calculate a specific evaluation metric."""
    start_process_metrics = time.time()

    # Handle special cases first
    if metric_name == MetricName.ACCURACY_FACT_KDB:
        return _handle_accuracy_metric(start_process_metrics)

    if metric_name == MetricName.PLAGIARISM:
        return await _handle_plagiarism_metric(
            content_sources, response_answer, start_process_metrics
        )

    if metric_name == MetricName.CONSISTENCY:
        return await _handle_consistency_metric(
            instruction,
            response_answer,
            agent_executions_to_eval_consistency,
            llm_judge_config,
            start_process_metrics,
        )

    # Handle standard metrics
    calculator = METRIC_CALCULATORS.get(metric_name)
    if not calculator:
        end_process_metrics = time.time() - start_process_metrics
        return None, {}, [], end_process_metrics, f"Unknown metric: `{metric_name}`"

    try:
        result = await calculator(
            instruction=instruction,
            response=response_answer,
            params=llm_judge_config,
        )

        score, stdev_score, scores, best_feedback, process_metrics = result

        additional_metrics = {
            "std_score": stdev_score,
            "scores": scores,
            "best_feedback": best_feedback,
            "llm_judge_config": json.loads(llm_judge_config.json()),
        }

        end_process_metrics = time.time() - start_process_metrics
        return score, additional_metrics, process_metrics, end_process_metrics, None

    except Exception as e:
        logger.error(f"Error calculating {metric_name}: {str(e)}")
        end_process_metrics = time.time() - start_process_metrics
        return None, {}, [], end_process_metrics, str(e)


def _handle_accuracy_metric(start_time: float) -> Tuple[None, Dict, List, float, None]:
    """Handle accuracy metric (placeholder for now)."""
    end_time = time.time() - start_time
    return None, {}, [], end_time, None


async def _handle_plagiarism_metric(
    content_sources: Dict[str, Dict[str, str]], response_answer: str, start_time: float
) -> Tuple[Optional[float], Dict[str, Any], List, float, Optional[str]]:
    """Handle plagiarism metric with special validation."""
    if not content_sources:
        end_time = time.time() - start_time
        return (
            None,
            {},
            [],
            end_time,
            "!!! PLAGIARISM WAS NOT CALCULATED, because: len(content_sources) == 0",
        )

    score, additional_metrics, process_metrics = await plagiarism_calc(
        content_sources=content_sources,
        response=response_answer,
        chunk_size=10,
        overlap_size=4,
        dimensions=10000,
    )

    if score is None:
        end_time = time.time() - start_time
        return (
            None,
            additional_metrics,
            process_metrics,
            end_time,
            "!!! PLAGIARISM WAS NOT CALCULATED, because: "
            "In `parser_results` SQL Table there were not any Original `content_sources`",
        )

    end_time = time.time() - start_time
    return score, additional_metrics, process_metrics, end_time, None


async def _handle_consistency_metric(
    instruction: str,
    response_answer: str,
    agent_executions: List,
    llm_judge_config: Any,
    start_time: float,
) -> Tuple[Optional[float], Dict[str, Any], List, float, Optional[str]]:
    """Handle consistency metric with special validation."""
    # Validate input data
    if not _is_valid_consistency_data(agent_executions):
        end_time = time.time() - start_time
        return (
            None,
            {},
            [],
            end_time,
            "!!! CONSISTENCY WAS NOT CALCULATED due to invalid input data",
        )

    result = await consistency_calc(
        instruction=instruction,
        response=response_answer,
        params=llm_judge_config,
        responses_to_eval_consistency=agent_executions,
    )

    score, stdev_score, scores, best_feedback, process_metrics = result

    additional_metrics = {
        "std_score": stdev_score,
        "scores": scores,
        "best_feedback": best_feedback,
        "llm_judge_config": json.loads(llm_judge_config.json()),
    }

    end_time = time.time() - start_time
    return score, additional_metrics, process_metrics, end_time, None


def _is_valid_consistency_data(agent_executions: List) -> bool:
    """Validate consistency evaluation data."""
    if not agent_executions:
        return False

    return all(len(execution.llm_responses) == 1 for execution in agent_executions)


async def calc_metric_full(
    channel_name,
    started_datetime,
    metrics,
    eval_metric_toggle,
    metric_name,
    instruction,
    response_answer,
    content_sources,
    agent_executions_to_eval_consistency,
    llm_judge_config,
    version_of_eval_score_dialog,
):
    if getattr(eval_metric_toggle, metric_name, False):
        (
            score,
            additional_metrics,
            process_metrics,
            end_process_metrics,
            error_message,
        ) = await calculate_metric(
            metric_name,
            instruction,
            response_answer,
            content_sources,
            agent_executions_to_eval_consistency,
            llm_judge_config,
        )

        if error_message is None:
            metrics["process_metrics"]["time_exe"] += end_process_metrics
            metrics["quality_metrics"]["absolute_metrics"][metric_name]["score"] = score
            metrics["quality_metrics"]["absolute_metrics"][metric_name][
                "additional_metrics"
            ] = additional_metrics
            metrics["process_metrics"][metric_name] = process_metrics

            await send_metrics_evaluation(
                channel_name=channel_name,
                metrics=metrics,
                version_of_eval_score_dialog=version_of_eval_score_dialog,
                event_status=PubSubStatusChannel.UPDATED,
                event_message=f"UPDATED_METRIC_{metric_name.upper()}",
                started_datetime=started_datetime,
            )
        else:
            await send_metrics_evaluation(
                channel_name=channel_name,
                metrics=metrics,
                version_of_eval_score_dialog=version_of_eval_score_dialog,
                event_status=PubSubStatusChannel.NOTIFICATION,
                event_message=error_message,
                started_datetime=started_datetime,
            )
    else:
        await send_metrics_evaluation(
            channel_name=channel_name,
            metrics=metrics,
            version_of_eval_score_dialog=version_of_eval_score_dialog,
            event_status=PubSubStatusChannel.NOTIFICATION,
            event_message=f"METRIC_{metric_name.upper()}_WAS_OFF",
            started_datetime=started_datetime,
        )
    return metrics


async def process_request(channel_name, request, metrics, started_datetime):
    main_start_process = time.time()
    await send_metrics_evaluation(
        channel_name=channel_name,
        metrics=metrics,
        version_of_eval_score_dialog=request.version_of_eval_score_dialog,
        event_status=PubSubStatusChannel.STARTED,
        event_message=PubSubStatusChannel.STARTED.value,
        started_datetime=started_datetime,
    )
    try:

        eval_metric_toggle = request.eval_score_cfg.eval_metric_toggle
        llm_judge_config = request.eval_score_cfg.llm_judge_config

        if len(request.llm_execution_main.llm_responses) == 0:

            metrics["time_exe"] = time.time() - main_start_process

            await send_metrics_evaluation(
                channel_name=channel_name,
                metrics=metrics,
                version_of_eval_score_dialog=request.version_of_eval_score_dialog,
                event_status=PubSubStatusChannel.FAILURE,
                event_message="`llm_responses` has to be >0!",
                started_datetime=started_datetime,
            )

        elif len(request.llm_execution_main.llm_responses) == 1:

            instruction = request.llm_execution_main.llm_full_executed_prompt
            response_answer = request.llm_execution_main.llm_responses[0].answer
            content_sources = request.llm_execution_main.llm_tool_response
            agent_executions_to_eval_consistency = (
                request.llm_executions_to_eval_consistency
            )

            # for metric_names_window_step in sliding_window(
            #     METRIC_NAMES, MAX_WORKERS_PROMETHEUS
            # ):
            # tasks = [
            #     calc_metric_full(
            #         channel_name,
            #         started_datetime,
            #         metrics,
            #         eval_metric_toggle,
            #         metric_name,
            #         instruction,
            #         response_answer,
            #         content_sources,
            #         agent_executions_to_eval_consistency,
            #         llm_judge_config,
            #         request.version_of_eval_score_dialog,
            #     )
            #     # for metric_name in metric_names_window_step
            #     for metric_name in METRIC_NAMES
            # ]
            # results = await asyncio.gather(*tasks)
            # for result in tqdm(results, desc="Processing metrics..."):
            #     metrics.update(result)

            for metric_name in METRIC_NAMES:
                metrics = await calc_metric_full(
                    channel_name=channel_name,
                    started_datetime=started_datetime,
                    metrics=metrics,
                    eval_metric_toggle=eval_metric_toggle,
                    metric_name=metric_name,
                    instruction=instruction,
                    response_answer=response_answer,
                    content_sources=content_sources,
                    agent_executions_to_eval_consistency=agent_executions_to_eval_consistency,
                    llm_judge_config=llm_judge_config,
                    version_of_eval_score_dialog=request.version_of_eval_score_dialog,
                )

            metrics["time_exe"] = time.time() - main_start_process
            await send_metrics_evaluation(
                channel_name=channel_name,
                metrics=metrics,
                version_of_eval_score_dialog=request.version_of_eval_score_dialog,
                event_status=PubSubStatusChannel.SUCCESS,
                event_message=PubSubStatusChannel.SUCCESS.value,
                started_datetime=started_datetime,
            )

        else:

            metrics["time_exe"] = time.time() - main_start_process

            await send_metrics_evaluation(
                channel_name=channel_name,
                metrics=metrics,
                version_of_eval_score_dialog=request.version_of_eval_score_dialog,
                event_status=PubSubStatusChannel.FAILURE,
                event_message=f"len(request.llm_execution_main.llm_responses)"
                f"[{len(request.llm_execution_main.llm_responses)}] > 1: "
                f"Relative Eval Calc NOT IMPLEMENTED! "
                f"len(request.llm_execution_main.llm_responses) HAS TO BE 1!",
                started_datetime=started_datetime,
            )

    except Exception as e:

        metrics["time_exe"] = time.time() - main_start_process

        exc = traceback.format_exc()
        logger.exception(exc)

        await send_metrics_evaluation(
            channel_name=channel_name,
            metrics=metrics,
            version_of_eval_score_dialog=request.version_of_eval_score_dialog,
            event_status=PubSubStatusChannel.FAILURE,
            event_message=exc,
            started_datetime=started_datetime,
        )

    return metrics


@router.post(
    "/score_calc_pipeline",
    response_model=Union[
        ResponseEvaluationScoreDialog, ResponseEvaluationScoreDialogPublishChannel
    ],
)
async def score_calc_pipeline(
    request: RequestEvaluationScoreDialog, background_tasks: BackgroundTasks
):
    started_datetime = datetime.now()
    logger.info(
        f"[user_id={request.user_id}] "
        f"[EVALUATION] "
        f"[/score_calc_pipeline] "
        f"Received POST request;\n"
        # f"PAYLOAD: {json.dumps(json.loads(request.json()), indent=2)}"
    )
    metrics = {
        "additional_metrics": {},
        "time_exe": 0.0,
        "process_metrics": {
            "additional_metrics": {},
            "time_exe": 0.0,
            "relevance": [],
            "plagiarism": [],
            "accuracy_compare_to_factKDB": [],
            "consistency": [],
            "directness": [],
            "comprehensiveness": [],
            "diversity": [],
            "empowerment": [],
        },
        "quality_metrics": {
            "absolute_metrics": {
                "relevance": {
                    "additional_metrics": {},
                    "score": None,
                },
                "plagiarism": {
                    "additional_metrics": {},
                    "score": None,
                },
                "accuracy_compare_to_factKDB": {
                    "additional_metrics": {},
                    "score": None,
                },
                "consistency": {"additional_metrics": {}, "score": None},
                "empowerment": {
                    "additional_metrics": {},
                    "score": None,
                },
                "diversity": {
                    "additional_metrics": {},
                    "score": None,
                },
                "comprehensiveness": {
                    "additional_metrics": {},
                    "score": None,
                },
                "directness": {"additional_metrics": {}, "score": None},
            }
        },
    }
    if request.pubsub_channel_name is None:
        channel_name = f"CALC-METRICS--{str(uuid.uuid4())}"
    else:
        channel_name = request.pubsub_channel_name

    if request.as_task:
        logger.info(
            f"[user_id={request.user_id}] "
            f"[EVALUATION] "
            f"[/score_calc_pipeline] "
            f"PROCESS AS TASK: request.as_task: {request.as_task}"
        )

        background_tasks.add_task(
            process_request,
            channel_name=channel_name,
            request=request,
            metrics=metrics,
            started_datetime=started_datetime,
        )
        return {
            "pubsub_channel_name": channel_name,
            "version_of_eval_score_dialog": request.version_of_eval_score_dialog.value,
            "commit_hash": os.getenv("COMMIT_HASH", ""),
        }

    else:
        logger.info(
            f"[user_id={request.user_id}] "
            f"[EVALUATION] "
            f"[/score_calc_pipeline] "
            f"PROCESS DEFAULT: request.as_task: {request.as_task}"
        )

        metrics = await process_request(
            channel_name=channel_name,
            request=request,
            metrics=metrics,
            started_datetime=started_datetime,
        )

        return {
            "metrics": metrics,
            "version_of_eval_score_dialog": request.version_of_eval_score_dialog.value,
            "commit_hash": os.getenv("COMMIT_HASH", ""),
        }
