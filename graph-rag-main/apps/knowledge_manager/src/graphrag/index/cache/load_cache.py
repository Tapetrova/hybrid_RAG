"""A module containing load_cache method definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from libs.python.schemas.enums import CacheType
from apps.knowledge_manager.src.graphrag.index.config.cache import (
    PipelineBlobCacheConfig,
    PipelineFileCacheConfig,
    PipelineS3CacheConfig,
    PipelineRedisCacheConfig,
)
from apps.knowledge_manager.src.graphrag.index.storage import (
    BlobPipelineStorage,
    FilePipelineStorage,
    S3PipelineStorage,
)

if TYPE_CHECKING:
    from apps.knowledge_manager.src.graphrag.index.config import (
        PipelineCacheConfig,
    )

from .json_pipeline_cache import JsonPipelineCache
from .memory_pipeline_cache import create_memory_cache
from .noop_pipeline_cache import NoopPipelineCache
from .redis_pipeline_cache import RedisPipelineCache


def load_cache(config: PipelineCacheConfig | None, root_dir: str | None):
    """Load the cache from the given config."""
    if config is None:
        return NoopPipelineCache()

    match config.type:
        case CacheType.none:
            return NoopPipelineCache()
        case CacheType.memory:
            return create_memory_cache()
        case CacheType.file:
            config = cast(PipelineFileCacheConfig, config)
            storage = FilePipelineStorage(root_dir).child(config.base_dir)
            return JsonPipelineCache(storage)
        case CacheType.blob:
            config = cast(PipelineBlobCacheConfig, config)
            storage = BlobPipelineStorage(
                config.connection_string,
                config.container_name,
                storage_account_blob_url=config.storage_account_blob_url,
            ).child(config.base_dir)
            return JsonPipelineCache(storage)
        case CacheType.s3:
            config = cast(PipelineS3CacheConfig, config)
            storage = S3PipelineStorage(
                bucket_name=config.bucket_name,
                region_name=config.region_name,
                path_prefix=config.base_dir,
            )
            return JsonPipelineCache(storage)
        case CacheType.redis:
            config = cast(PipelineRedisCacheConfig, config)
            redis_pipeline_cache = RedisPipelineCache(name=config.base_name)
            return redis_pipeline_cache
        case _:
            msg = f"Unknown cache type: {config.type}"
            raise ValueError(msg)
