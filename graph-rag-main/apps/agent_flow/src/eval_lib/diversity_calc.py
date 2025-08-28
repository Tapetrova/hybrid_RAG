from apps.agent_flow.src.eval_lib.base_eval import SimpleEvaluationMetric
from apps.agent_flow.src.eval_lib.rubrics import generate_diversity_rubric_json_string
from apps.agent_flow.src.schemas.eval_config import LLMJudgeConfig

# Create a reusable diversity metric instance
_diversity_metric = SimpleEvaluationMetric(
    metric_name="diversity", rubric_generator_func=generate_diversity_rubric_json_string
)


async def diversity_calc(
    instruction: str, response: str, params: LLMJudgeConfig = LLMJudgeConfig()
):
    """Calculate diversity metric for the given instruction and response."""
    return await _diversity_metric.calculate(instruction, response, params)
