from typing import List, Union, Optional

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client import models
from qdrant_client.conversions import common_types as qd_types

from libs.python.schemas.configuration import EmbedderModel
from libs.python.utils.logger import logger
from .base_qdrant_processor import (
    BaseQdrantConfig,
    BaseNeuralSearcher,
    create_collection_if_not_exists,
)

# Initialize async and sync Qdrant clients
QDRANT_CLIENT = AsyncQdrantClient(
    host=BaseQdrantConfig.HOST,
    port=BaseQdrantConfig.PORT,
    timeout=BaseQdrantConfig.TIMEOUT,
    prefer_grpc=BaseQdrantConfig.PREFER_GRPC,
    https=BaseQdrantConfig.HTTPS,
    api_key=BaseQdrantConfig.API_KEY,
)

QDRANT_CLIENT_SYNC = QdrantClient(
    host=BaseQdrantConfig.HOST,
    port=BaseQdrantConfig.PORT,
    timeout=BaseQdrantConfig.TIMEOUT,
    prefer_grpc=BaseQdrantConfig.PREFER_GRPC,
    https=BaseQdrantConfig.HTTPS,
    api_key=BaseQdrantConfig.API_KEY,
)


class AsyncNeuralSearcher(BaseNeuralSearcher):
    """Asynchronous neural searcher for Qdrant"""

    def __init__(
        self,
        collection_name: str,
        chunk_size: int = 8190,
        embedder_model: EmbedderModel = EmbedderModel.TEXT_EMBEDDING_ADA_002,
        qdrant_client: AsyncQdrantClient = QDRANT_CLIENT,
    ):
        self.qdrant_client = qdrant_client
        super().__init__(collection_name, chunk_size, embedder_model)

    def _initialize_collection(self) -> None:
        """Initialize collection using sync client"""
        try:
            create_collection_if_not_exists(
                collection_name=self.collection_name, qdrant_client=QDRANT_CLIENT_SYNC
            )
        except ValueError:
            # Retry once on ValueError
            create_collection_if_not_exists(
                collection_name=self.collection_name, qdrant_client=QDRANT_CLIENT_SYNC
            )

    async def retrieve_records(self, ids: List[str]) -> List[qd_types.Record]:
        """Retrieve records by IDs asynchronously"""
        results = await self.qdrant_client.retrieve(
            collection_name=self.collection_name, ids=ids
        )
        return results

    async def search(
        self,
        query: str,
        query_filter: Optional[models.Filter] = None,
        top_k: int = 5,
        score_threshold: float = 0.7,
        with_payload: Union[bool, List[str]] = None,
    ) -> List[qd_types.ScoredPoint]:
        """Search for similar vectors asynchronously"""
        if with_payload is None:
            with_payload = ["text"]

        # Convert text query into vector
        vector = await self.model.aembed_query(text=query)

        # Search for closest vectors
        search_result = await self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=with_payload,
        )
        return search_result
