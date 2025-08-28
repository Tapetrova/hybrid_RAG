import os
import time
from typing import List, Dict, Union, Optional, Tuple, Any
from abc import ABC, abstractmethod

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client import models
from qdrant_client.conversions import common_types as qd_types

from libs.python.schemas.configuration import EmbedderModel
from libs.python.utils.logger import logger


class BaseQdrantConfig:
    """Base configuration for Qdrant clients"""

    HOST = os.getenv("QDRANT_HOST", "localhost")
    PORT = int(os.getenv("QDRANT_PORT", 6333))
    TIMEOUT = 3600
    PREFER_GRPC = True
    HTTPS = False
    API_KEY = os.getenv("QDRANT_API_KEY")


def create_collection_if_not_exists(
    collection_name: str,
    qdrant_client: QdrantClient,
    vector_size: int = 1536,
    distance: models.Distance = models.Distance.COSINE,
) -> None:
    """Create collection with indexes if it doesn't exist"""
    if not qdrant_client.collection_exists(collection_name):
        logger.info(f"QDRANT COLLECTION_NAME {collection_name} does NOT exist")

        # Create collection
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
        )
        time.sleep(4.0)

        # Create indexes
        indexes = ["tag", "text", "src", "country", "locale"]
        schemas = ["keyword", "text", "text", "keyword", "keyword"]

        for field_name, field_schema in zip(indexes, schemas):
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )

        time.sleep(4.0)
        logger.info(f"QDRANT collection {collection_name} was created!")
    else:
        logger.info(f"QDRANT collection {collection_name} already exists!")


class BaseNeuralSearcher(ABC):
    """Abstract base class for neural searchers"""

    def __init__(
        self,
        collection_name: str,
        chunk_size: int = 8190,
        embedder_model: EmbedderModel = EmbedderModel.TEXT_EMBEDDING_ADA_002,
    ):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.embedder_model = embedder_model
        self.model = OpenAIEmbeddings(model=embedder_model.value, chunk_size=chunk_size)
        self._initialize_collection()

    @abstractmethod
    def _initialize_collection(self) -> None:
        """Initialize collection - to be implemented by subclasses"""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        query_filter: Optional[Any] = None,
        top_k: int = 5,
        score_threshold: float = 0.7,
        with_payload: Union[bool, List[str]] = None,
    ) -> List[qd_types.ScoredPoint]:
        """Search for similar vectors - to be implemented by subclasses"""
        pass
