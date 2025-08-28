"""A module containing load_input method definition."""

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pandas as pd

from apps.knowledge_manager.src.graphrag.config import InputConfig, InputType
from apps.knowledge_manager.src.graphrag.index.config import PipelineInputConfig
from apps.knowledge_manager.src.graphrag.index.progress import (
    NullProgressReporter,
    ProgressReporter,
)
from apps.knowledge_manager.src.graphrag.index.storage import (
    BlobPipelineStorage,
    FilePipelineStorage,
    S3PipelineStorage,
)

from .csv import input_type as csv
from .csv import load as load_csv
from .text import input_type as text
from .text import load as load_text

from libs.python.utils.logger import logger

log = logger
loaders: dict[str, Callable[..., Awaitable[pd.DataFrame]]] = {
    text: load_text,
    csv: load_csv,
}


async def load_input(
    config: PipelineInputConfig | InputConfig,
    progress_reporter: ProgressReporter | None = None,
    root_dir: str | None = None,
) -> pd.DataFrame:
    """Load the input data for a pipeline."""
    root_dir = root_dir or ""
    log.info(
        f"loading input from root_dir={config.base_dir}",
    )
    progress_reporter = progress_reporter or NullProgressReporter()

    if config is None:
        msg = "No input specified!"
        raise ValueError(msg)

    match config.type:
        case InputType.blob:
            log.info("using blob storage input")
            if config.container_name is None:
                msg = "Container name required for blob storage"
                raise ValueError(msg)
            if (
                config.connection_string is None
                and config.storage_account_blob_url is None
            ):
                msg = "Connection string or storage account blob url required for blob storage"
                raise ValueError(msg)
            storage = BlobPipelineStorage(
                connection_string=config.connection_string,
                storage_account_blob_url=config.storage_account_blob_url,
                container_name=config.container_name,
                path_prefix=config.base_dir,
            )
        case InputType.s3:
            log.info("using s3 storage input")
            if config.bucket_name is None:
                msg = "s3 required for `bucket_name`"
                raise ValueError(msg)
            if config.region_name is None:
                msg = "s3 required for `region_name`"
                raise ValueError(msg)
            storage = S3PipelineStorage(
                bucket_name=config.bucket_name,
                region_name=config.region_name,
                path_prefix=config.base_dir,
            )
        case InputType.file:
            log.info("using file storage for input")
            storage = FilePipelineStorage(
                root_dir=str(Path(root_dir) / (config.base_dir or ""))
            )
        case _:
            log.info("using file storage for input")
            storage = FilePipelineStorage(
                root_dir=str(Path(root_dir) / (config.base_dir or ""))
            )

    if config.file_type in loaders:
        progress = progress_reporter.child(
            f"Loading Input ({config.file_type})", transient=False
        )
        loader = loaders[config.file_type]
        results = await loader(config, progress, storage)
        return cast(pd.DataFrame, results)

    msg = f"Unknown input type {config.file_type}"
    raise ValueError(msg)
