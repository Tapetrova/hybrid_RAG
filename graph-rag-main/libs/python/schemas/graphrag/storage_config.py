"""Parameterization settings for the default configuration."""

import os
import time

from libs.python.schemas.basic_models import BaseModelUpd, BaseEnum, Field

import libs.python.schemas.defaults as defs
from libs.python.schemas.enums import StorageType


class StorageConfig(BaseModelUpd):
    """The default configuration section for Storage."""

    type: StorageType = Field(
        description="The storage type to use.", default=defs.STORAGE_TYPE
    )
    base_dir: str = Field(
        description="The base directory for the storage.",
        default=defs.STORAGE_BASE_DIR,
    )
    connection_string: str | None = Field(
        description="The storage connection string to use.", default=None
    )
    container_name: str | None = Field(
        description="The storage container name to use.", default=None
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

    object_name: str | None = Field(
        description="Log file name in S3 bucket", default=None
    )
    """Log file name in S3 bucket"""
