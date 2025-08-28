from enum import Enum
from devtools import pformat

from pydantic import BaseModel, Field

import json


class BaseEnum(Enum):
    def __str__(self):
        """Return the string representation of the enum value."""
        return self.value


class BaseModelUpd(BaseModel):
    # class BaseModelUpd(BaseModel):

    def __repr__(self) -> str:
        """Get a string representation."""
        return pformat(self, highlight=False)

    def __str__(self):
        """Get a string representation."""
        return self.model_dump_json()

    def model_dump_json(self, **kwargs) -> str:
        """Get a JSON string representation of the model."""
        return json.dumps(self.dict(), indent=2)


class LLMModel(str, BaseEnum):
    GPT_4_0613 = "gpt-4-0613"
    GPT_35_TURBO_16k = "gpt-3.5-turbo-16k"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

    def __str__(self):
        """Return the string representation of the enum value."""
        return self.value


TOKEN_PERCENTAGE_FOR_RAG = 0.3
PERCENTAGE_FROM_TOKEN_RAG_SLICE_TOOL_RESPONSE = 0.05

TOKEN_THRESHOLDS = {
    LLMModel.GPT_35_TURBO_16k: 13500,
    LLMModel.GPT_4_0613: 7400,
    LLMModel.GPT_4O: 126000,
    LLMModel.GPT_4O_MINI: 126000,
}
