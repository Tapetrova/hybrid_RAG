import json
import time
import uuid
from typing import List

from fastapi import APIRouter, HTTPException, status

from apps.knowledge_manager.src.schemas.default_rag_schemas import (
    RequestRecordContentDefaultRAG,
    ResponseRecordContentDefaultRAG,
    RequestRecordContentGraphRAG,
    ResponseRecordContentGraphRAG,
)
from libs.python.utils.logger import logger
from apps.knowledge_manager.src.celery_workers import (
    process_record_content,
    run_workflow_graph_building,
)

router = APIRouter()


class BuildingService:
    """Service class for handling building operations"""

    @staticmethod
    def generate_timestamp_folder() -> str:
        """Generate a unique timestamp folder name"""
        return f"{uuid.uuid4()}_{time.strftime('%Y%m%d-%H%M%S')}"

    @staticmethod
    def extract_sources(knowledge_content: List) -> List[str]:
        """Extract source URLs from knowledge content"""
        return [content.src for content in knowledge_content]

    @staticmethod
    def serialize_content(content: List) -> List[dict]:
        """Serialize knowledge content to JSON"""
        return [json.loads(item.json()) for item in content]


@router.post(
    "/record_content",
    response_model=ResponseRecordContentDefaultRAG,
    summary="Record content for default RAG processing",
    status_code=status.HTTP_200_OK,
)
async def record_content(request: RequestRecordContentDefaultRAG):
    """
    Record content for default RAG processing.

    Args:
        request: Request containing knowledge content and configuration

    Returns:
        ResponseRecordContentDefaultRAG: Response with task IDs and status

    Raises:
        HTTPException: If processing fails
    """
    try:
        # Extract sources for logging
        srcs = BuildingService.extract_sources(request.knowledge_content)
        logger.info(
            f"Received request to Record Content; SRCS: {json.dumps(srcs, indent=2)}"
        )

        # Process each knowledge content item
        tasks = []
        config_json = json.loads(request.config.json())

        for content_item in request.knowledge_content:
            result = process_record_content.delay(
                text_sample_knowledge_content=content_item.text,
                src_sample_knowledge_content=content_item.src,
                config=config_json,
            )
            tasks.append(result.id)

        logger.info(f"Tasks were sent to process_record_content: {tasks}")

        return ResponseRecordContentDefaultRAG(
            tasks=tasks,
            status=True,
        )

    except Exception as e:
        logger.exception(f"Error in record_content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process content: {str(e)}",
        )


@router.post(
    "/graphrag/build",
    response_model=ResponseRecordContentGraphRAG,
    summary="Build GraphRAG knowledge graph",
    status_code=status.HTTP_200_OK,
)
async def graphrag_build(request: RequestRecordContentGraphRAG):
    """
    Build GraphRAG knowledge graph from provided content.

    Args:
        request: Request containing knowledge content and configuration

    Returns:
        ResponseRecordContentGraphRAG: Response with task ID, status, and timestamp folder

    Raises:
        HTTPException: If graph building fails
    """
    try:
        # Extract sources for logging
        srcs = BuildingService.extract_sources(request.knowledge_content)
        logger.info(
            f"Received request to [/graphrag/build]; SRCS: {json.dumps(srcs, indent=2)}"
        )

        # Generate unique timestamp folder
        timestamp_folder = BuildingService.generate_timestamp_folder()

        # Serialize content and configuration
        serialized_content = BuildingService.serialize_content(
            request.knowledge_content
        )
        config_json = json.loads(request.config.json())

        # Submit task to Celery
        task_run = run_workflow_graph_building.delay(
            knowledge_content=serialized_content,
            config=config_json,
            timestamp_folder=timestamp_folder,
        )

        logger.info(f"Task was sent to run_workflow_graph_building: {task_run.id}")

        return ResponseRecordContentGraphRAG(
            tasks=[task_run.id],
            status=True,
            timestamp_folder=timestamp_folder,
        )

    except Exception as e:
        logger.exception(f"Error in graphrag_build: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build graph: {str(e)}",
        )
