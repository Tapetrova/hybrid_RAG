"""A module containing 'PipelineCacheConfig', 'PipelineFileCacheConfig' and 'PipelineMemoryCacheConfig' models."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from libs.python.schemas.configuration import BaseModelUpd
from pydantic import Field as pydantic_Field

from libs.python.schemas.enums import CacheType

T = TypeVar("T")


class PipelineCacheConfig(BaseModelUpd, Generic[T]):
    """Represent the cache configuration for the pipeline."""

    type: T


class PipelineFileCacheConfig(PipelineCacheConfig[Literal[CacheType.file]]):
    """Represent the file cache configuration for the pipeline."""

    type: Literal[CacheType.file] = CacheType.file
    """The type of cache."""

    base_dir: str | None = pydantic_Field(
        description="The base directory for the cache.", default=None
    )
    """The base directory for the cache."""


class PipelineMemoryCacheConfig(PipelineCacheConfig[Literal[CacheType.memory]]):
    """Represent the memory cache configuration for the pipeline."""

    type: Literal[CacheType.memory] = CacheType.memory
    """The type of cache."""


class PipelineNoneCacheConfig(PipelineCacheConfig[Literal[CacheType.none]]):
    """Represent the none cache configuration for the pipeline."""

    type: Literal[CacheType.none] = CacheType.none
    """The type of cache."""


class PipelineBlobCacheConfig(PipelineCacheConfig[Literal[CacheType.blob]]):
    """Represents the blob cache configuration for the pipeline."""

    type: Literal[CacheType.blob] = CacheType.blob
    """The type of cache."""

    base_dir: str | None = pydantic_Field(
        description="The base directory for the cache.", default=None
    )
    """The base directory for the cache."""

    connection_string: str | None = pydantic_Field(
        description="The blob cache connection string for the cache.", default=None
    )
    """The blob cache connection string for the cache."""

    container_name: str | None = pydantic_Field(
        description="The container name for cache", default=None
    )
    """The container name for cache"""

    storage_account_blob_url: str | None = pydantic_Field(
        description="The storage account blob url for cache", default=None
    )
    """The storage account blob url for cache"""


class PipelineS3CacheConfig(PipelineCacheConfig[Literal[CacheType.s3]]):
    """Represents the S3 cache configuration for the pipeline."""

    type: Literal[CacheType.s3] = CacheType.s3
    """The type of cache."""

    base_dir: str | None = pydantic_Field(
        description="The base directory for the cache.", default=None
    )
    """The base directory for the cache."""

    bucket_name: str | None = pydantic_Field(
        description="The bucket name for the cache.", default="dev-1"
    )
    """The bucket name for the cache."""

    region_name: str | None = pydantic_Field(
        description="The AWS region name for the cache.", default=None
    )
    """The AWS region name for the cache."""


class PipelineRedisCacheConfig(PipelineCacheConfig[Literal[CacheType.redis]]):
    """Represents the Redis cache configuration for the pipeline."""

    type: Literal[CacheType.redis] = CacheType.redis
    """The type of cache."""

    base_name: str | None = pydantic_Field(
        description="The base name for the cache.", default=None
    )
    """The base name for the cache."""


PipelineCacheConfigTypes = (
    PipelineFileCacheConfig
    | PipelineMemoryCacheConfig
    | PipelineBlobCacheConfig
    | PipelineNoneCacheConfig
    | PipelineS3CacheConfig
    | PipelineRedisCacheConfig
)
