"""Load pipeline reporter method."""

from pathlib import Path
from typing import cast

from datashaper import WorkflowCallbacks

from apps.knowledge_manager.src.graphrag.config import ReportingType
from apps.knowledge_manager.src.graphrag.index.config import (
    PipelineBlobReportingConfig,
    PipelineFileReportingConfig,
    PipelineReportingConfig,
    PipelineS3ReportingConfig,
)

from .s3_workflow_callbacks import S3WorkflowCallbacks
from .blob_workflow_callbacks import BlobWorkflowCallbacks
from .console_workflow_callbacks import ConsoleWorkflowCallbacks
from .file_workflow_callbacks import FileWorkflowCallbacks


def load_pipeline_reporter(
    config: PipelineReportingConfig | None, root_dir: str | None
) -> WorkflowCallbacks:
    """Create a reporter for the given pipeline config."""
    config = config or PipelineFileReportingConfig(base_dir="reports")

    match config.type:
        case ReportingType.file:
            config = cast(PipelineFileReportingConfig, config)
            return FileWorkflowCallbacks(
                str(Path(root_dir or "") / (config.base_dir or ""))
            )
        case ReportingType.console:
            return ConsoleWorkflowCallbacks()
        case ReportingType.blob:
            config = cast(PipelineBlobReportingConfig, config)
            return BlobWorkflowCallbacks(
                config.connection_string,
                config.container_name,
                base_dir=config.base_dir,
                storage_account_blob_url=config.storage_account_blob_url,
            )
        case ReportingType.s3:
            config = cast(PipelineS3ReportingConfig, config)
            return S3WorkflowCallbacks(
                base_dir=config.base_dir,
                region_name=config.region_name,
                object_name=config.object_name,
                bucket_name=config.bucket_name,
            )
        case _:
            msg = f"Unknown reporting type: {config.type}"
            raise ValueError(msg)
