from apps.agent_flow.src.eval_lib.base_eval import SimpleEvaluationMetric
from apps.agent_flow.src.eval_lib.rubrics import generate_directness_rubric_json_string
from apps.agent_flow.src.schemas.eval_config import LLMJudgeConfig

# Create a reusable directness metric instance
_directness_metric = SimpleEvaluationMetric(
    metric_name="directness",
    rubric_generator_func=generate_directness_rubric_json_string,
)


async def directness_calc(
    instruction: str, response: str, params: LLMJudgeConfig = LLMJudgeConfig()
):
    """Calculate directness metric for the given instruction and response."""
    return await _directness_metric.calculate(instruction, response, params)
