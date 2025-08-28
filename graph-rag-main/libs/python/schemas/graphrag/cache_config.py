"""Parameterization settings for the default configuration."""

import os

from libs.python.schemas.basic_models import BaseModelUpd, BaseEnum, Field

import libs.python.schemas.defaults as defs
from libs.python.schemas.enums import CacheType


class CacheConfig(BaseModelUpd):
    """The default configuration section for Cache."""

    type: CacheType = Field(
        description="The cache type to use.", default=defs.CACHE_TYPE
    )
    base_dir: str = Field(
        description="The base directory for the cache.", default=defs.CACHE_BASE_DIR
    )
    base_name: str = Field(
        description="The base name for the cache.", default=defs.CACHE_BASE_NAME
    )
    connection_string: str | None = Field(
        description="The cache conxnection string to use.", default=None
    )
    container_name: str | None = Field(
        description="The cache container name to use.", default=None
    )
    storage_account_blob_url: str | None = Field(
        description="The storage account blob url to use.", default=None
    )
    bucket_name: str | None = Field(
        description="The S3 bucket name for reporting", default="dev-1"
    )
    """The S3 bucket name for reporting"""

    region_name: str | None = Field(
        description="The region name for the S3 bucket",
        default=os.getenv("AWS_S3_REGION", None),
    )
    """The region name for the S3 bucket"""
