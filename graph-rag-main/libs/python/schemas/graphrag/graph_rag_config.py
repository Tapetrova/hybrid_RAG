"""Parameterization settings for the default configuration."""

import os
from enum import Enum

from devtools import pformat
from pydantic import Field

import libs.python.schemas.defaults as defs

from .cache_config import CacheConfig
from .chunking_config import ChunkingConfig
from .claim_extraction_config import ClaimExtractionConfig
from .cluster_graph_config import ClusterGraphConfig
from .community_reports_config import CommunityReportsConfig
from .embed_graph_config import EmbedGraphConfig
from .entity_extraction_config import EntityExtractionConfig
from .global_search_config import GlobalSearchConfig
from .input_config import InputConfig
from .llm_config import LLMConfig
from .local_search_config import LocalSearchConfig
from .reporting_config import ReportingConfig
from .snapshots_config import SnapshotsConfig
from .storage_config import StorageConfig
from .summarize_descriptions_config import (
    SummarizeDescriptionsConfig,
)
from .text_embedding_config import TextEmbeddingConfig
from .umap_config import UmapConfig
from ..basic_models import BaseEnum


class SearchMode(str, BaseEnum):
    global_mode = "global_mode"
    local_mode = "local_mode"
    question_generation_mode = "question_generation_mode"


class GraphRagConfig(LLMConfig):
    """Base class for the Default-Configuration parameterization settings."""

    search_mode: SearchMode = Field(
        # default=SearchMode.global_mode,
        default=SearchMode.local_mode,
        description="Search Mode",
    )

    reporting: ReportingConfig = Field(
        description="The reporting configuration.", default=ReportingConfig()
    )
    """The reporting configuration."""

    storage: StorageConfig = Field(
        description="The storage configuration.", default=StorageConfig()
    )
    """The storage configuration."""

    cache: CacheConfig = Field(
        description="The cache configuration.", default=CacheConfig()
    )
    """The cache configuration."""

    input: InputConfig = Field(
        description="The input configuration.", default=InputConfig()
    )
    """The input configuration."""

    embed_graph: EmbedGraphConfig = Field(
        description="Graph embedding configuration.",
        default=EmbedGraphConfig(),
    )
    """Graph Embedding configuration."""

    embeddings: TextEmbeddingConfig = Field(
        description="The embeddings LLM configuration to use.",
        default=TextEmbeddingConfig(),
    )
    """The embeddings LLM configuration to use."""

    chunks: ChunkingConfig = Field(
        description="The chunking configuration to use.",
        default=ChunkingConfig(),
    )
    """The chunking configuration to use."""

    snapshots: SnapshotsConfig = Field(
        description="The snapshots configuration to use.",
        default=SnapshotsConfig(),
    )
    """The snapshots configuration to use."""

    entity_extraction: EntityExtractionConfig = Field(
        description="The entity extraction configuration to use.",
        default=EntityExtractionConfig(),
    )
    """The entity extraction configuration to use."""

    summarize_descriptions: SummarizeDescriptionsConfig = Field(
        description="The description summarization configuration to use.",
        default=SummarizeDescriptionsConfig(),
    )
    """The description summarization configuration to use."""

    community_reports: CommunityReportsConfig = Field(
        description="The community reports configuration to use.",
        default=CommunityReportsConfig(),
    )
    """The community reports configuration to use."""

    claim_extraction: ClaimExtractionConfig = Field(
        description="The claim extraction configuration to use.",
        default=ClaimExtractionConfig(),
    )
    """The claim extraction configuration to use."""

    cluster_graph: ClusterGraphConfig = Field(
        description="The cluster graph configuration to use.",
        default=ClusterGraphConfig(),
    )
    """The cluster graph configuration to use."""

    umap: UmapConfig = Field(
        description="The UMAP configuration to use.", default=UmapConfig()
    )
    """The UMAP configuration to use."""

    local_search: LocalSearchConfig = Field(
        description="The local search configuration.", default=LocalSearchConfig()
    )
    """The local search configuration."""

    global_search: GlobalSearchConfig = Field(
        description="The global search configuration.", default=GlobalSearchConfig()
    )
    """The global search configuration."""

    checkpoint_s3_folder_name: str = Field(
        description="{s3_prefix_path}/{checkpoint_s3_folder_name}/artifacts/{content_data}",
        default="4089a296-f4ce-4b21-830e-ca56b8419732_20240829-222856",
    )
    """{s3_prefix_path}/{checkpoint_s3_folder_name}/artifacts/{content_data}"""

    community_level: int = Field(
        description="Community Level in the Leiden community hierarchy from which we will load the community reports, "
        "higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)",
        default=2,
    )
    encoding_model: str = Field(
        description="The encoding model to use.", default=defs.ENCODING_MODEL
    )
    """The encoding model to use."""

    skip_workflows: list[str] = Field(
        description="The workflows to skip, usually for testing reasons.", default=[]
    )
    """The workflows to skip, usually for testing reasons."""
