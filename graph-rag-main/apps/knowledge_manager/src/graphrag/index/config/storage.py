"""A module containing 'PipelineStorageConfig', 'PipelineFileStorageConfig' and 'PipelineMemoryStorageConfig' models."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import Field as pydantic_Field
from libs.python.schemas.basic_models import BaseModelUpd, BaseEnum
from libs.python.schemas.enums import StorageType

T = TypeVar("T")


class PipelineStorageConfig(BaseModelUpd, Generic[T]):
    """Represent the storage configuration for the pipeline."""

    type: T


class PipelineFileStorageConfig(PipelineStorageConfig[Literal[StorageType.file]]):
    """Represent the file storage configuration for the pipeline."""

    type: Literal[StorageType.file] = StorageType.file
    """The type of storage."""

    base_dir: str | None = pydantic_Field(
        description="The base directory for the storage.", default=None
    )
    """The base directory for the storage."""


class PipelineMemoryStorageConfig(PipelineStorageConfig[Literal[StorageType.memory]]):
    """Represent the memory storage configuration for the pipeline."""

    type: Literal[StorageType.memory] = StorageType.memory
    """The type of storage."""


class PipelineBlobStorageConfig(PipelineStorageConfig[Literal[StorageType.blob]]):
    """Represents the blob storage configuration for the pipeline."""

    type: Literal[StorageType.blob] = StorageType.blob
    """The type of storage."""

    connection_string: str | None = pydantic_Field(
        description="The blob storage connection string for the storage.", default=None
    )
    """The blob storage connection string for the storage."""

    container_name: str | None = pydantic_Field(
        description="The container name for storage", default=None
    )
    """The container name for storage."""

    base_dir: str | None = pydantic_Field(
        description="The base directory for the storage.", default=None
    )
    """The base directory for the storage."""

    storage_account_blob_url: str | None = pydantic_Field(
        description="The storage account blob url.", default=None
    )
    """The storage account blob url."""


class PipelineS3StorageConfig(PipelineStorageConfig[Literal[StorageType.s3]]):
    """Represents the S3 storage configuration for the pipeline."""

    type: Literal[StorageType.s3] = StorageType.s3
    """The type of storage."""

    bucket_name: str | None = pydantic_Field(
        description="The S3 bucket name for storage", default="dev-1"
    )
    """The S3 bucket name for storage."""

    region_name: str | None = pydantic_Field(
        description="The region name for the S3 bucket", default=None
    )
    """The region name for the S3 bucket."""

    base_dir: str | None = pydantic_Field(
        description="The base directory for the storage.", default=None
    )
    """The base directory for the storage."""


PipelineStorageConfigTypes = (
    PipelineFileStorageConfig
    | PipelineMemoryStorageConfig
    | PipelineBlobStorageConfig
    | PipelineS3StorageConfig
)
