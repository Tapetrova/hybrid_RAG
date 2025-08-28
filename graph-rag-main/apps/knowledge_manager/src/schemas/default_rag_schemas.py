from typing import List, Optional, Union, Any, Dict

from pydantic.v1 import Field

from apps.knowledge_manager.src.schemas.basic_models import (
    KnowledgeContent,
    RequestRecordContent,
    ResponseRecordContent,
)
from libs.python.schemas.basic_models import BaseModelUpd
from libs.python.schemas.configuration import Config
from libs.python.schemas.events import KnowledgeManagerEvent


class SearchResultModel(BaseModelUpd):
    """A Structured Search Result."""

    response: str | dict[str, Any] | list[dict[str, Any]] | None
    context_data: str | list[dict[str, Any]] | dict[str, dict[str, Any]]
    # actual text strings that are in the context window, built from context_data
    context_text: str | list[str] | dict[str, str]
    completion_time: float
    llm_calls: int
    prompt_tokens: int
    resources: Optional[Dict] = None


class GlobalSearchResultModel(SearchResultModel):
    """A GlobalSearch result."""

    map_responses: list[SearchResultModel]
    reduce_context_data: str | list[dict[str, Any]] | dict[str, dict[str, Any]]
    reduce_context_text: str | list[str] | dict[str, str]


class QuestionResultModel(BaseModelUpd):
    """A Structured Question Result."""

    response: list[str]
    context_data: str | dict[str, Any]
    completion_time: float
    llm_calls: int
    prompt_tokens: int


class RequestKnowledgeManagerRetrieve(BaseModelUpd):
    user_id: str
    session_id: str
    dialog_id: str
    natural_query: str
    config: Config
    pubsub_channel_name: Optional[str] = None


class RequestDefaultRAGRetrieve(RequestKnowledgeManagerRetrieve):
    pass


class RequestGraphRAGRetrieve(RequestKnowledgeManagerRetrieve):
    # chat_history: Optional[List[Dict[str, str]]] = Field(
    #     default=None,
    #     description='Dict should be {"role": string, "content": string}; '
    #     '"role" should be in ["user", "assistant"]',
    # )
    pass


class ResponseVectorRetrieve(KnowledgeContent):
    knowledge_manager_events: List[KnowledgeManagerEvent]


class ResponseGraphRAGRetrieve(BaseModelUpd):
    graph_rag_answer: Union[
        GlobalSearchResultModel, SearchResultModel, QuestionResultModel
    ]
    knowledge_manager_events: List[KnowledgeManagerEvent]


class RequestRecordContentDefaultRAG(RequestRecordContent):
    pass


class ResponseRecordContentDefaultRAG(ResponseRecordContent):
    pass


class RequestRecordContentGraphRAG(RequestRecordContent):
    pass


class ResponseRecordContentGraphRAG(ResponseRecordContent):
    timestamp_folder: str
