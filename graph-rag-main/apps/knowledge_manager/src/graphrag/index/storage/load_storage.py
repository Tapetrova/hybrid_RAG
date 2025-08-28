"""A module containing load_storage method definition."""

from __future__ import annotations

from typing import cast

from apps.knowledge_manager.src.graphrag.config import StorageType
from apps.knowledge_manager.src.graphrag.index.config.storage import (
    PipelineBlobStorageConfig,
    PipelineS3StorageConfig,
    PipelineFileStorageConfig,
    PipelineStorageConfig,
)

from .s3_pipeline_storage import create_s3_storage
from .blob_pipeline_storage import create_blob_storage
from .file_pipeline_storage import create_file_storage
from .memory_pipeline_storage import create_memory_storage


def load_storage(config: PipelineStorageConfig):
    """Load the storage for a pipeline."""
    match config.type:
        case StorageType.memory:
            return create_memory_storage()
        case StorageType.blob:
            config = cast(PipelineBlobStorageConfig, config)
            return create_blob_storage(
                config.connection_string,
                config.storage_account_blob_url,
                config.container_name,
                config.base_dir,
            )
        case StorageType.s3:
            config = cast(PipelineS3StorageConfig, config)
            return create_s3_storage(
                config.bucket_name,
                config.region_name,
                config.base_dir,
            )
        case StorageType.file:
            config = cast(PipelineFileStorageConfig, config)
            return create_file_storage(config.base_dir)
        case _:
            msg = f"Unknown storage type: {config.type}"
            raise ValueError(msg)
