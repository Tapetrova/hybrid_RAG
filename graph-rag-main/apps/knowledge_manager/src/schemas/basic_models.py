from typing import List, Optional

from libs.python.schemas.basic_models import BaseModelUpd, BaseEnum
from libs.python.schemas.configuration import Config


class Content(BaseModelUpd):
    text: str
    src: str
    title: Optional[str] = None


class KnowledgeContent(BaseModelUpd):
    knowledge_content: List[Content]


class RequestRecordContent(KnowledgeContent):
    config: Config


class ResponseRecordContent(BaseModelUpd):
    status: bool
    tasks: List[str]
