from pydantic import Field

from libs.python.schemas.basic_models import BaseModelUpd


class LLMJudgeConfig(BaseModelUpd):
    times_to_eval: int = Field(default=2, gt=0, lt=16)
    batch_size_to_eval: int = Field(default=5, gt=0, lt=16)

    temperature: float = Field(
        description="Temperature of LLM.", default=0.8, ge=0.0, le=1.0
    )
    max_tokens: int = Field(
        description="Max Tokens of LLM.",
        default=1024,
        ge=1,
    )

    repetition_penalty: float = Field(
        description="Repetition Penalty for LLM.",
        default=1.03,
    )

    top_p: float = Field(
        description="Top P for LLM.",
        default=0.9,
    )

    best_of: int = Field(
        description="`Best of` for LLM.",
        default=1,
    )


class EvalMetricToggle(BaseModelUpd):
    consistency: bool = True
    plagiarism: bool = True
    relevance: bool = True
    accuracy_compare_to_factKDB: bool = True

    directness: bool = True
    diversity: bool = True
    empowerment: bool = True
    comprehensiveness: bool = True


class EvaluationScoreDialogConfig(BaseModelUpd):
    llm_judge_config: LLMJudgeConfig = LLMJudgeConfig()
    eval_metric_toggle: EvalMetricToggle = EvalMetricToggle()
