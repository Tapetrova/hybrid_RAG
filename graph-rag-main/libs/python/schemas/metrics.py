from enum import Enum
from typing import Dict, Any, Optional, List, Union

from libs.python.schemas.basic_models import BaseModelUpd, BaseEnum


class BasicMetrics(BaseModelUpd):
    additional_metrics: Optional[Dict[str, Any]] = dict()


class ProcessBasicMetrics(BasicMetrics):
    time_exe: float


class InputLLMProcessMetrics(BaseModelUpd):
    input_cost: float  # input cost in cents
    input_tokens: int


class OutputLLMProcessMetrics(BaseModelUpd):
    output_cost: float  # output cost in cents
    output_tokens: int


class TotalLLMProcessMetrics(BaseModelUpd):
    total_cost: float  # output cost in cents
    total_tokens: int


class TypeProcess(str, BaseEnum):
    LLM_PROCESS = "llm_process"
    PARSING_PROCESS = "parsing_process"
    RETRIEVAL_PROCESS = "retrieval_process"
    BUILDING_PROCESS = "building_process"


class LLMProcessMetrics(ProcessBasicMetrics):
    name: str
    type_process: TypeProcess = TypeProcess.LLM_PROCESS
    input_metrics: InputLLMProcessMetrics
    output_metrics: OutputLLMProcessMetrics
    total_metrics: TotalLLMProcessMetrics
    count_input_tokens_exceeded: int = 0


class ParsingProcessMetrics(ProcessBasicMetrics):
    name: str
    type_process: TypeProcess = TypeProcess.PARSING_PROCESS


class RetrievalProcessMetrics(ProcessBasicMetrics):
    name: str
    type_process: TypeProcess = TypeProcess.RETRIEVAL_PROCESS


class BuildingProcessMetrics(ProcessBasicMetrics):
    name: str
    type_process: TypeProcess = TypeProcess.BUILDING_PROCESS


class AgentFlowMetrics(ProcessBasicMetrics):
    processes: List[
        Union[LLMProcessMetrics]
    ]  # Named Pipeline, Sequential of LLM processes


class ContentScraperMetrics(ProcessBasicMetrics):
    processes: List[
        Union[ParsingProcessMetrics]
    ]  # Named Pipeline, Sequential of LLM processes


class KnowledgeManagerMetrics(ProcessBasicMetrics):
    processes: List[
        Union[RetrievalProcessMetrics, BuildingProcessMetrics]
    ]  # Named Pipeline, Sequential of LLM processes
    knowledge_depth: int  # count of sources
    knowledge_width: int  # count of make/model names
