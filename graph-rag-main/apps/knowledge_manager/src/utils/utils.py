import json
from typing import Union, Any

import pandas as pd
import requests

from apps.knowledge_manager.src.graphrag.query.question_gen.base import QuestionResult
from apps.knowledge_manager.src.graphrag.query.structured_search.base import (
    SearchResult,
)
from apps.knowledge_manager.src.graphrag.query.structured_search.global_search.search import (
    GlobalSearchResult,
)
from apps.knowledge_manager.src.schemas.default_rag_schemas import (
    SearchResultModel,
    QuestionResultModel,
    GlobalSearchResultModel,
)
from dataclasses import dataclass, asdict
from libs.python.utils.logger import logger
import uuid


async def send_request(data, endpoint: str):
    response = requests.post(endpoint, json=data)
    if response.status_code == 200:
        logger.info(f"Sent request to ***{endpoint}**")
        logger.info(response.json())
    else:
        logger.error(
            f"Error: Unable to send request to {endpoint}, "
            f"status code {response.status_code}; {json.dumps(response.json())}"
        )


def int_to_uuid(user_id):
    # Generate a random UUID
    random_uuid = uuid.uuid4()
    # Convert the UUID to a list of integers
    uuid_numbers = list(random_uuid.fields)
    # Embed the user_id into the first part of the UUID (for simplicity)
    uuid_numbers[0] = user_id
    # Generate a new UUID using modified numbers
    new_uuid = uuid.UUID(fields=tuple(uuid_numbers))
    return str(new_uuid)


def uuid_to_int(uuid_str):
    # Extract the user_id from the UUID
    user_uuid = uuid.UUID(uuid_str)
    # Return the first part of the UUID where the user_id was embedded
    return user_uuid.fields[0]


def hash_string_to_rank(string: str) -> int:
    # get signed 64-bit hash value
    signed_hash = hash(string)

    # reduce the hash value to a 64-bit range
    mask = (1 << 64) - 1
    signed_hash &= mask

    # convert the signed hash value to an unsigned 64-bit integer
    if signed_hash & (1 << 63):
        unsigned_hash = -((signed_hash ^ mask) + 1)
    else:
        unsigned_hash = signed_hash

    return unsigned_hash


def convert_field_graphrag_res_from_pd_to_json(
    field_res: str | list[pd.DataFrame] | dict[str, pd.DataFrame],
) -> str | list[dict[str, Any]] | dict[str, dict[str, Any]]:
    if isinstance(field_res, list):
        for el_i in range(len(field_res)):
            field_res[el_i] = json.loads(field_res[el_i].to_json(orient="columns"))
    elif isinstance(field_res, dict):
        for el_key in field_res.keys():
            field_res[el_key] = json.loads(field_res[el_key].to_json(orient="columns"))
    return field_res


def convert_graphrag_res_from_pd_to_json(
    graphrag_res: Union[GlobalSearchResult, QuestionResult, SearchResult]
) -> Union[GlobalSearchResultModel, QuestionResultModel, SearchResultModel]:
    if isinstance(graphrag_res, (SearchResult, GlobalSearchResult)):
        graphrag_res.context_data = convert_field_graphrag_res_from_pd_to_json(
            graphrag_res.context_data
        )
        graphrag_res_model = SearchResultModel(**asdict(graphrag_res))

        if isinstance(graphrag_res, GlobalSearchResult):
            graphrag_res.reduce_context_data = (
                convert_field_graphrag_res_from_pd_to_json(
                    graphrag_res.reduce_context_data
                )
            )
            for el_i in range(len(graphrag_res.map_responses)):
                graphrag_res.map_responses[el_i].context_data = (
                    convert_field_graphrag_res_from_pd_to_json(
                        graphrag_res.map_responses[el_i].context_data
                    )
                )
            graphrag_res_model = GlobalSearchResultModel(**asdict(graphrag_res))

    elif isinstance(graphrag_res, QuestionResult):
        graphrag_res_model = QuestionResultModel(**asdict(graphrag_res))
    else:
        raise ValueError(
            "`graphrag_res` should be `Union[GlobalSearchResult, QuestionResult, SearchResult]`"
        )

    return graphrag_res_model


async def aconvert_field_graphrag_res_from_pd_to_json(
    field_res: str | list[pd.DataFrame] | dict[str, pd.DataFrame],
) -> str | list[dict[str, Any]] | dict[str, dict[str, Any]]:
    if isinstance(field_res, list):
        for el_i in range(len(field_res)):
            field_res[el_i] = json.loads(field_res[el_i].to_json(orient="columns"))
    elif isinstance(field_res, dict):
        for el_key in field_res.keys():
            field_res[el_key] = json.loads(field_res[el_key].to_json(orient="columns"))
    return field_res


async def aconvert_graphrag_res_from_pd_to_json(
    graphrag_res: Union[GlobalSearchResult, QuestionResult, SearchResult]
) -> Union[GlobalSearchResultModel, QuestionResultModel, SearchResultModel]:
    if isinstance(graphrag_res, (SearchResult, GlobalSearchResult)):
        graphrag_res.context_data = convert_field_graphrag_res_from_pd_to_json(
            graphrag_res.context_data
        )
        graphrag_res_model = SearchResultModel(**asdict(graphrag_res))

        if isinstance(graphrag_res, GlobalSearchResult):
            graphrag_res.reduce_context_data = (
                await aconvert_field_graphrag_res_from_pd_to_json(
                    graphrag_res.reduce_context_data
                )
            )

            for el_i in range(len(graphrag_res.map_responses)):
                graphrag_res.map_responses[el_i].context_data = (
                    await aconvert_field_graphrag_res_from_pd_to_json(
                        graphrag_res.map_responses[el_i].context_data
                    )
                )

            graphrag_res_model = GlobalSearchResultModel(**asdict(graphrag_res))

    elif isinstance(graphrag_res, QuestionResult):
        graphrag_res_model = QuestionResultModel(**asdict(graphrag_res))
    else:
        raise ValueError(
            "`graphrag_res` should be `Union[GlobalSearchResult, QuestionResult, SearchResult]`"
        )

    return graphrag_res_model
