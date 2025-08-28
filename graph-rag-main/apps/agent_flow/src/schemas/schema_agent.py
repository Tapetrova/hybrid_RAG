from enum import Enum
from typing import List, Optional, Dict, Any

from libs.python.schemas.basic_models import BaseModelUpd, BaseEnum

from libs.python.schemas.config_presets import ConfigPreSet
from libs.python.schemas.configuration import Config
from libs.python.schemas.events import Event, MessageChannel
from libs.python.schemas.metrics import (
    AgentFlowMetrics,
    KnowledgeManagerMetrics,
    ContentScraperMetrics,
    ProcessBasicMetrics,
)


class MessageType(str, BaseEnum):
    AI = "AI"
    HUMAN = "HUMAN"


class Message(BaseModelUpd):
    type_message: MessageType
    content: str


class RequestAgentProcessUserMessage(BaseModelUpd):
    user_id: str
    dialog_content: List[Message]
    dialog_id: Optional[str] = None
    config_preset_name: Optional[ConfigPreSet] = (
        ConfigPreSet.VDB_INET_SCRAP_USE_VERIF_GPT4O_MINI
    )
    config: Optional[Config] = None
    return_agent_full_executed_prompt: Optional[bool] = False
    return_agent_tool_output: Optional[bool] = False
    pubsub_channel_name: Optional[str] = None
    as_task: Optional[bool] = False


class ResponseGetConfig(BaseModelUpd):
    config_name: ConfigPreSet
    config: Config
    commit_hash: str


class VersionEvaluationScoreDialog(str, BaseEnum):
    V0 = "V0"


class AgentAnswer(BaseModelUpd):
    agent_name: str = "vector_db_content_scraper_baseline_v0"
    answer: str


class AgentMetricsSummary(ProcessBasicMetrics):
    agent_flow: AgentFlowMetrics
    knowledge_manager: KnowledgeManagerMetrics
    content_scraper: ContentScraperMetrics


class ResponseAgentProcessUserMessageDefault(BaseModelUpd):
    agent_answer: str
    metrics: AgentMetricsSummary
    agent_full_executed_prompt: Optional[str] = None
    agent_output_tool: Optional[Dict[str, Any]] = None
    generated_next_user_steps: Optional[List[str]] = None
    final_agent_answer_summarized: Optional[str] = None


class ResponseAgentProcessUserMessage(ResponseAgentProcessUserMessageDefault):
    agent_events: List[Event]


class ResponseAgentProcessUserPublishChannel(BaseModelUpd):
    pubsub_channel_name: str


class ResponseAgentProcessUserMessageChannel(MessageChannel):
    response: Optional[ResponseAgentProcessUserMessageDefault]
