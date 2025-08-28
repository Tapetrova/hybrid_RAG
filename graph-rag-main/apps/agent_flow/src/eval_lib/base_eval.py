from typing import Tuple, List, Optional, Dict, Any
from apps.agent_flow.src.eval_lib.base_calc import BasePrometheusEval
from apps.agent_flow.src.schemas.eval_config import LLMJudgeConfig


class BaseEvaluationMetric:
    """Base class for all evaluation metrics to reduce code duplication."""

    def __init__(self, metric_name: str, rubric_generator_func):
        self.metric_name = metric_name
        self.rubric_generator_func = rubric_generator_func
        self.calculator = BasePrometheusEval()

    async def calculate(
        self,
        instruction: str,
        response: str,
        params: LLMJudgeConfig = LLMJudgeConfig(),
        **kwargs
    ) -> Tuple[float, float, List, str, List[Dict[str, Any]]]:
        """Calculate the metric score."""
        rubric_json_string = self._generate_rubric(**kwargs)

        return await self.calculator.acalc_metric(
            instruction=instruction,
            response=response,
            rubric=rubric_json_string,
            params=params,
        )

    def _generate_rubric(self, **kwargs) -> str:
        """Generate the rubric JSON string."""
        if kwargs:
            return self.rubric_generator_func(**kwargs)
        return self.rubric_generator_func()


class SimpleEvaluationMetric(BaseEvaluationMetric):
    """Evaluation metric that doesn't require additional parameters."""

    def __init__(self, metric_name: str, rubric_generator_func):
        super().__init__(metric_name, rubric_generator_func)

    async def calculate(
        self, instruction: str, response: str, params: LLMJudgeConfig = LLMJudgeConfig()
    ) -> Tuple[float, float, List, str, List[Dict[str, Any]]]:
        """Calculate the metric score without additional parameters."""
        return await super().calculate(instruction, response, params)
