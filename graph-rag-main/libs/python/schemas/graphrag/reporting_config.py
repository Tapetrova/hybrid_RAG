"""Parameterization settings for the default configuration."""

import os
import time

from libs.python.schemas.basic_models import BaseModelUpd, BaseEnum, Field

import libs.python.schemas.defaults as defs
from libs.python.schemas.enums import ReportingType


class ReportingConfig(BaseModelUpd):
    """The default configuration section for Reporting."""

    type: ReportingType = Field(
        description="The reporting type to use.", default=defs.REPORTING_TYPE
    )
    base_dir: str = Field(
        description="The base directory for reporting.",
        default=defs.REPORTING_BASE_DIR,
    )
    connection_string: str | None = Field(
        description="The reporting connection string to use.", default=None
    )
    container_name: str | None = Field(
        description="The reporting container name to use.", default=None
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
        default=os.getenv("AWS_S3_REGION", "eu-north-1"),
    )
    """The region name for the S3 bucket"""

    object_name: str | None = Field(
        description="Log file name in S3 bucket", default=None
    )
    """Log file name in S3 bucket"""
