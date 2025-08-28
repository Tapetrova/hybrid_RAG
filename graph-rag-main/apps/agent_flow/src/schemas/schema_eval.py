from enum import Enum
from typing import List, Optional, Dict, Union

from apps.agent_flow.src.schemas.eval_config import EvaluationScoreDialogConfig
from libs.python.schemas.basic_models import BaseModelUpd, BaseEnum

from apps.agent_flow.src.schemas.schema_agent import AgentAnswer
from libs.python.schemas.events import MessageChannel
from libs.python.schemas.metrics import (
    ProcessBasicMetrics,
    LLMProcessMetrics,
    BasicMetrics,
)


class LLMExecutionData(BaseModelUpd):
    llm_full_executed_prompt: str
    llm_tool_response: Dict
    llm_responses: List[AgentAnswer]


class VersionEvaluationScoreDialog(str, BaseEnum):
    V0 = "V0"


class RequestEvaluationScoreDialog(BaseModelUpd):
    user_id: str
    llm_execution_main: LLMExecutionData
    version_of_eval_score_dialog: VersionEvaluationScoreDialog = (
        VersionEvaluationScoreDialog.V0
    )
    llm_executions_to_eval_consistency: List[LLMExecutionData] = None
    pubsub_channel_name: Optional[str] = None
    as_task: Optional[bool] = True
    eval_score_cfg: Optional[EvaluationScoreDialogConfig] = (
        EvaluationScoreDialogConfig()
    )


class ResponseEvaluationScoreDialogPublishChannel(BaseModelUpd):
    pubsub_channel_name: str
    version_of_eval_score_dialog: VersionEvaluationScoreDialog
    commit_hash: str


class RelevanceMetric(BasicMetrics):
    score: Optional[float] = None


class PlagiarismMetric(BasicMetrics):
    score: Optional[float] = None


class ConsistencyMetric(BasicMetrics):
    score: Optional[float] = None


class AccuracyCompareToFactKDB(BasicMetrics):
    score: Optional[float] = None


class DirectnessMetric(BasicMetrics):
    score: Optional[float] = None


class DiversityMetric(BasicMetrics):
    score: Optional[float] = None


class EmpowermentMetric(BasicMetrics):
    score: Optional[float] = None


class ComprehensivenessMetric(BasicMetrics):
    score: Optional[float] = None


class AbsoluteMetrics(BaseModelUpd):
    relevance: RelevanceMetric
    plagiarism: PlagiarismMetric
    accuracy_compare_to_factKDB: AccuracyCompareToFactKDB
    consistency: ConsistencyMetric

    directness: DirectnessMetric
    diversity: DiversityMetric
    empowerment: EmpowermentMetric
    comprehensiveness: ComprehensivenessMetric


class QualityMetrics(BaseModelUpd):
    absolute_metrics: AbsoluteMetrics


class PrometheusEvalMetricsSummary(ProcessBasicMetrics):
    relevance: List[LLMProcessMetrics]
    plagiarism: List[LLMProcessMetrics]
    accuracy_compare_to_factKDB: List[LLMProcessMetrics]
    consistency: List[LLMProcessMetrics]

    directness: List[LLMProcessMetrics]
    diversity: List[LLMProcessMetrics]
    empowerment: List[LLMProcessMetrics]
    comprehensiveness: List[LLMProcessMetrics]


class EvalMetricsSummary(ProcessBasicMetrics):
    process_metrics: PrometheusEvalMetricsSummary
    quality_metrics: QualityMetrics


class ResponseEvaluationScoreDialog(BaseModelUpd):
    metrics: EvalMetricsSummary
    version_of_eval_score_dialog: VersionEvaluationScoreDialog
    commit_hash: str


class ResponseEvaluationScoreDialogMessageChannel(MessageChannel):
    response: ResponseEvaluationScoreDialog
