from typing import List, Dict, Union, Tuple, Optional

from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.conversions import common_types as qd_types

from libs.python.schemas.configuration import EmbedderModel
from libs.python.utils.logger import logger
from .base_qdrant_processor import (
    BaseQdrantConfig,
    BaseNeuralSearcher,
    create_collection_if_not_exists,
)

# Initialize Qdrant client
QDRANT_CLIENT = QdrantClient(
    host=BaseQdrantConfig.HOST,
    port=BaseQdrantConfig.PORT,
    timeout=BaseQdrantConfig.TIMEOUT,
    prefer_grpc=BaseQdrantConfig.PREFER_GRPC,
    https=BaseQdrantConfig.HTTPS,
    api_key=BaseQdrantConfig.API_KEY,
)


class NeuralSearcher(BaseNeuralSearcher):
    """Synchronous neural searcher for Qdrant"""

    def __init__(
        self,
        collection_name: str,
        chunk_size: int = 8190,
        embedder_model: EmbedderModel = EmbedderModel.TEXT_EMBEDDING_ADA_002,
        qdrant_client: QdrantClient = QDRANT_CLIENT,
    ):
        self.qdrant_client = qdrant_client
        super().__init__(collection_name, chunk_size, embedder_model)

    def _initialize_collection(self) -> None:
        """Initialize collection with error handling"""
        try:
            create_collection_if_not_exists(
                collection_name=self.collection_name, qdrant_client=self.qdrant_client
            )
        except ValueError:
            # Retry once on ValueError
            create_collection_if_not_exists(
                collection_name=self.collection_name, qdrant_client=self.qdrant_client
            )

    def search_by_payload_scroll_must_values(
        self, must_value: Dict[str, str]
    ) -> Tuple[List[qd_types.Record], Optional[qd_types.PointId]]:
        """Search by payload with scroll filter"""
        results = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in must_value.items()
                ],
            ),
        )
        return results

    def retrieve_records(self, ids: List[str]) -> List[qd_types.Record]:
        """Retrieve records by IDs"""
        results = self.qdrant_client.retrieve(
            collection_name=self.collection_name, ids=ids
        )
        return results

    def upload(
        self,
        ids: List[str],
        payloads: List[Dict[str, str]],
        texts: List[str],
    ) -> None:
        """Upload data to Qdrant collection"""
        assert (
            len(ids) == len(payloads) == len(texts)
        ), f"Length mismatch: ids={len(ids)}, payloads={len(payloads)}, texts={len(texts)}"

        # Add text to payloads
        for payload, text in zip(payloads, texts):
            payload["text"] = text

        # Embed texts
        vectors = self.model.embed_documents(texts=texts)

        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                payloads=payloads,
                vectors=vectors,
            ),
        )

    def search(
        self,
        query: str,
        query_filter: Optional[models.Filter] = None,
        top_k: int = 5,
        score_threshold: float = 0.7,
        with_payload: Union[bool, List[str]] = None,
    ) -> List[qd_types.ScoredPoint]:
        """Search for similar vectors"""
        if with_payload is None:
            with_payload = ["text"]

        # Convert text query into vector
        vector = self.model.embed_query(text=query)

        # Search for closest vectors
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=with_payload,
        )
        return search_result
