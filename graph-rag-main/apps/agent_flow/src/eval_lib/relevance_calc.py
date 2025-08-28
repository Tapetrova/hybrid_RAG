from apps.agent_flow.src.eval_lib.base_eval import SimpleEvaluationMetric
from apps.agent_flow.src.eval_lib.rubrics import generate_relevance_rubric_json_string
from apps.agent_flow.src.schemas.eval_config import LLMJudgeConfig

# Create a reusable relevance metric instance
_relevance_metric = SimpleEvaluationMetric(
    metric_name="relevance", rubric_generator_func=generate_relevance_rubric_json_string
)


async def relevance_calc(
    instruction: str, response: str, params: LLMJudgeConfig = LLMJudgeConfig()
):
    """Calculate relevance metric for the given instruction and response."""
    return await _relevance_metric.calculate(instruction, response, params)
