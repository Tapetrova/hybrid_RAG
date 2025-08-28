from apps.agent_flow.src.eval_lib.base_eval import SimpleEvaluationMetric
from apps.agent_flow.src.eval_lib.rubrics import generate_empowerment_rubric_json_string
from apps.agent_flow.src.schemas.eval_config import LLMJudgeConfig

# Create a reusable empowerment metric instance
_empowerment_metric = SimpleEvaluationMetric(
    metric_name="empowerment",
    rubric_generator_func=generate_empowerment_rubric_json_string,
)


async def empowerment_calc(
    instruction: str, response: str, params: LLMJudgeConfig = LLMJudgeConfig()
):
    """Calculate empowerment metric for the given instruction and response."""
    return await _empowerment_metric.calculate(instruction, response, params)
