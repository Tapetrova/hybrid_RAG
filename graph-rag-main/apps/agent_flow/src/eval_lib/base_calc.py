import json
import os

# import statistics
import numpy as np
import time

from tqdm import tqdm

from apps.agent_flow.src.schemas.eval_config import LLMJudgeConfig
from libs.python.utils.logger import logger
from libs.python.utils.request_utils import asend_request

MAX_WORKERS = 2
ENDPOINT_PROMETHEUS_EVAL = os.getenv(
    "ENDPOINT_PROMETHEUS_EVAL", "http://localhost:5000/predictions"
)


async def process_time_iter(time_iter: int, payload: dict):
    scores = []
    feedbacks = []

    prometheus_eval_process_st_time = time.time()
    logger.info(f"Execute step [{time_iter+1}]")
    result = await asend_request(data=payload, endpoint=ENDPOINT_PROMETHEUS_EVAL)

    result_data = result.json()
    logger.info(f"RESULT:\n{result.json()}")

    if result_data.get("status") == "succeeded":
        scores = result_data.get("output").get("scores")
        feedbacks = result_data.get("output").get("feedbacks")
        # if result_data.get("output").get("stdev_score") is not None:
        #     list_of_stdev_score.append(result_data.get("output").get("stdev_score"))

        total_input_tokens = [
            process_metrics_batch_el["input_metrics"]["input_tokens"]
            for process_metrics_batch_el in result_data["output"]["process_metrics"]
        ] + [
            process_metrics_batch_el["input_metrics"]["input_tokens"]
            for process_metrics_batch_el in result_data["output"][
                "process_retries_metrics"
            ]
        ]
        total_output_tokens = [
            process_metrics_batch_el["output_metrics"]["output_tokens"]
            for process_metrics_batch_el in result_data["output"]["process_metrics"]
        ] + [
            process_metrics_batch_el["output_metrics"]["output_tokens"]
            for process_metrics_batch_el in result_data["output"][
                "process_retries_metrics"
            ]
        ]
        total_total_tokens = [
            process_metrics_batch_el["total_metrics"]["total_tokens"]
            for process_metrics_batch_el in result_data["output"]["process_metrics"]
        ] + [
            process_metrics_batch_el["total_metrics"]["total_tokens"]
            for process_metrics_batch_el in result_data["output"][
                "process_retries_metrics"
            ]
        ]

        processes_metrics = {
            "additional_metrics": {},
            "time_exe": time.time() - prometheus_eval_process_st_time,
            "name": f"prometheus_eval_step_{time_iter}",
            "type_process": "llm_process",
            "input_metrics": {
                "input_cost": 0,
                "input_tokens": sum(total_input_tokens),
            },
            "output_metrics": {
                "output_cost": 0,
                "output_tokens": sum(total_output_tokens),
            },
            "total_metrics": {
                "total_cost": 0,
                "total_tokens": sum(total_total_tokens),
            },
        }

    else:
        processes_metrics = {
            "additional_metrics": {},
            "time_exe": time.time() - prometheus_eval_process_st_time,
            "name": f"prometheus_eval_step_{time_iter}",
            "type_process": "llm_process",
            "input_metrics": {"input_cost": 0, "input_tokens": 0},
            "output_metrics": {"output_cost": 0, "output_tokens": 0},
            "total_metrics": {"total_cost": 0, "total_tokens": 0},
        }

        logger.error(f'!!! result_data.get("status")={result_data.get("status")} !!!')
        logger.error(f'!!! result_data.get("status") != "succeeded" !!!')
        logger.error(
            f'!!! result_data.get("error"):\n\n {result_data.get("error")}\n\n'
        )

    return scores, feedbacks, processes_metrics


class BasePrometheusEval:
    def __init__(self):
        pass

    async def acalc_metric(
        self, instruction: str, response: str, rubric: str, params: LLMJudgeConfig
    ):
        payload = {
            "input": {
                "instruction": instruction,
                "response": response,
                "rubric": rubric,
                "times2judge": params.batch_size_to_eval,
                "temperature": params.temperature,
                "max_tokens": params.max_tokens,
                "repetition_penalty": params.repetition_penalty,
                "top_p": params.top_p,
                "best_of": params.best_of,
            }
        }

        scores = []
        # list_of_stdev_score = []
        feedbacks = []
        processes_metrics = []
        for time_iter in tqdm(
            range(params.times_to_eval),
            desc=f"Processing `times_to_eval` one of metrics...",
        ):
            result_ = await process_time_iter(time_iter, payload)
            scores_, feedbacks_, processes_metrics_ = result_
            scores.append(scores_)
            feedbacks.append(feedbacks_)
            # list_of_stdev_score.extend(list_of_stdev_score_)
            processes_metrics.append(processes_metrics_)

        # tasks = [
        #     process_time_iter(time_iter, payload)
        #     for time_iter in range(params.times_to_eval)
        # ]
        # results = await asyncio.gather(*tasks)
        # for result in tqdm(
        #     results, desc=f"Processing `times_to_eval` one of metrics..."
        # ):
        #     scores_, feedbacks_, list_of_stdev_score_, processes_metrics_ = result
        #     scores.extend(scores_)
        #     feedbacks.extend(feedbacks_)
        #     list_of_stdev_score.extend(list_of_stdev_score_)
        #     processes_metrics.extend(processes_metrics_)

        # scores_flatten = np.array(scores).reshape(-1).tolist()
        # feedbacks_flatten = np.array(feedbacks).reshape(-1).tolist()
        scores_flatten = [ss for s in scores for ss in s]
        feedbacks_flatten = [fff for ff in feedbacks for fff in ff]

        if len(feedbacks_flatten) == 0:
            best_feedback = ""
            min_score_index = None
        else:
            min_score_index = min(
                range(len(feedbacks_flatten)), key=lambda k: scores_flatten[k]
            )
            best_feedback = feedbacks_flatten[min_score_index]

        if len(scores_flatten) == 0:
            avg_score = None
            stdev_score = None
        elif len(scores_flatten) == 1:
            avg_score = scores_flatten[0]
            stdev_score = 0.0
        else:
            # avg_score = statistics.mean(scores)
            # stdev_score = statistics.stdev(scores)
            avg_score = float(np.mean(scores_flatten))
            stdev_score = float(np.std(scores_flatten))

        logger.info(f"feedbacks: {json.dumps(feedbacks)}")
        logger.info(f"scores: {json.dumps(scores)}")
        logger.info(f"min_score_index: {min_score_index}")
        logger.info(f"avg_score: {avg_score}")
        logger.info(f"stdev_score: {stdev_score}")
        # logger.info(f"list_of_stdev_score: {list_of_stdev_score}")
        logger.info(f"best_feedback: {best_feedback}")
        return (
            avg_score,
            stdev_score,
            # list_of_stdev_score,
            scores,
            best_feedback,
            processes_metrics,
        )
