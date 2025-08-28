from typing import List, Tuple

from apps.agent_flow.src.eval_lib.base_eval import BaseEvaluationMetric
from apps.agent_flow.src.eval_lib.rubrics import generate_consistency_rubric_json_string
from apps.agent_flow.src.schemas import LLMExecutionData
from apps.agent_flow.src.schemas.eval_config import LLMJudgeConfig


# Create a reusable consistency metric instance
_consistency_metric = BaseEvaluationMetric(
    metric_name="consistency",
    rubric_generator_func=generate_consistency_rubric_json_string,
)


async def consistency_calc(
    instruction: str,
    response: str,
    responses_to_eval_consistency: List[LLMExecutionData],
    params: LLMJudgeConfig = LLMJudgeConfig(),
):
    """Calculate consistency metric for the given instruction and responses."""
    # Parse instruction to extract system prompt and dialog
    instruction_split = instruction.split("\n\nCHAT HISTORY:\n\n")
    sys_prompt_main = instruction_split[0] + "\n\nCHAT HISTORY:\n\n"
    dialog_string_main = instruction_split[1]

    # Process consistency evaluation data
    new_dialogs, new_answers = _process_consistency_data(responses_to_eval_consistency)

    # Format the response for evaluation
    response_updated = _format_consistency_response(response, new_answers)

    # Calculate the metric
    return await _consistency_metric.calculate(
        instruction=sys_prompt_main,
        response=response_updated,
        params=params,
        dialog_string_main=dialog_string_main,
        new_dialogs_to_eval_consistency=new_dialogs,
    )


def _process_consistency_data(
    responses_to_eval: List[LLMExecutionData],
) -> Tuple[str, str]:
    """Process consistency evaluation data to extract dialogs and answers."""
    new_dialogs = ""
    new_answers = ""

    for num, response_data in enumerate(responses_to_eval):
        prompt = response_data.llm_full_executed_prompt
        answer = response_data.llm_responses[0].answer

        # Extract dialog from prompt
        dialog = prompt.split("\n\nCHAT HISTORY:\n\n")[1]

        # Format dialogs and answers
        new_dialogs += f"New Dialogue with dialog number {num+1}:\n" f"{dialog}\n\n"
        new_answers += (
            f"New Answer is Based on slightly modified dialogue number {num+1}:\n"
            f"{answer}\n"
        )

    return new_dialogs, new_answers


def _format_consistency_response(main_response: str, new_answers: str) -> str:
    """Format the response for consistency evaluation."""
    return (
        f"\n\nMain Response is based on main dialogue:\n{main_response}\n\n\n"
        f"\n\nResponses are based on slightly modified dialogues:\n{new_answers}\n\n"
    )
