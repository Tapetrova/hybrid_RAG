import json

from fastapi import APIRouter
from apps.knowledge_manager.src.celery_workers import (
    test_from_s3_processed_to_sql_table_data_uploading,
)
from libs.python.schemas.basic_models import BaseModelUpd
from libs.python.schemas.config_presets import get_config, ConfigPreSet
from libs.python.schemas.configuration import Config
from libs.python.utils.logger import logger

router = APIRouter()


class RequestSQL(BaseModelUpd):
    config: Config = get_config(preset=ConfigPreSet.GRAPHRAG_TEST_GPT4O_MINI)


@router.post("/from_s3_to_sql_table_data_uploading")
async def from_s3_to_sql_table_data_uploading_func(request: RequestSQL):
    res = test_from_s3_processed_to_sql_table_data_uploading.delay(
        json.loads(str(request.config))
    )
    logger.info(res)
    return "OK"
