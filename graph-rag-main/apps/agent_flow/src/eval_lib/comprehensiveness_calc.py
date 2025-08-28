from apps.agent_flow.src.eval_lib.base_eval import SimpleEvaluationMetric
from apps.agent_flow.src.eval_lib.rubrics import (
    generate_comprehensiveness_rubric_json_string,
)
from apps.agent_flow.src.schemas.eval_config import LLMJudgeConfig

# Create a reusable comprehensiveness metric instance
_comprehensiveness_metric = SimpleEvaluationMetric(
    metric_name="comprehensiveness",
    rubric_generator_func=generate_comprehensiveness_rubric_json_string,
)


async def comprehensiveness_calc(
    instruction: str, response: str, params: LLMJudgeConfig = LLMJudgeConfig()
):
    """Calculate comprehensiveness metric for the given instruction and response."""
    return await _comprehensiveness_metric.calculate(instruction, response, params)
