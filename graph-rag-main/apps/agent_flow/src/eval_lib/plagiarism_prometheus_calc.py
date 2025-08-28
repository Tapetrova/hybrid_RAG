import json
import os

from typing import List, Dict


from apps.agent_flow.src.eval_lib.base_calc import (
    BasePrometheusEval,
)
from apps.agent_flow.src.eval_lib.rubrics import (
    generate_plagiarism_rubric_json_string,
)
from apps.agent_flow.src.schemas.eval_config import LLMJudgeConfig
from libs.python.utils.request_utils import asend_request

ENDPOINT_GET_SOURCE_CONTENT = os.getenv(
    "ENDPOINT_GET_SOURCE_CONTENT",
    "http://localhost:8099/data/get_content_source_by_urls",
)


async def plagiarism_calc(
    instruction: str,
    response: str,
    content_sources: Dict[str, Dict[str, str]],
    params: LLMJudgeConfig = LLMJudgeConfig(),
):
    list_of_urls_of_sources = list(
        set(
            (
                content_source.get("Source")
                for content_source in content_sources.values()
            )
        )
    )  # get uniq list of sources

    response_source_contents_ = await asend_request(
        data={"urls": list_of_urls_of_sources}, endpoint=ENDPOINT_GET_SOURCE_CONTENT
    )

    source_retrival_info = list()
    if response_source_contents_:
        response_source_contents = response_source_contents_.json()
        for response_source_content in response_source_contents.get("results"):
            url_source = response_source_content.get("url")
            content_source = response_source_content.get("content")

            source_retrival_info.append(
                {"Content": content_source, "Source": url_source}
            )

    str_source_retrival_info = json.dumps(source_retrival_info, indent=2)

    plagiarism_rubric_json_string = generate_plagiarism_rubric_json_string(
        source_retrival_info=str_source_retrival_info
    )
    calculator = BasePrometheusEval()
    (
        avg_score,
        stdev_score,
        # list_of_stdev_score,
        scores,
        best_feedback,
        processes_relevance_metrics,
    ) = await calculator.acalc_metric(
        instruction=instruction,
        response=response,
        rubric=plagiarism_rubric_json_string,
        params=params,
    )
    return (
        avg_score,
        stdev_score,
        # list_of_stdev_score,
        scores,
        best_feedback,
        processes_relevance_metrics,
    )
