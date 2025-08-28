import os
import time
from typing import Any, List
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition
from qdrant_openapi_client.models import models

from apps.knowledge_manager.src.graphrag.model.types import TextEmbedder
from libs.python.utils.logger import logger

from .base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)


class QdrantVectorStore(BaseVectorStore):
    """The Qdrant vector storage implementation."""

    def connect(self, **kwargs: Any) -> None:
        """Connect to the Qdrant vector storage."""
        st = time.time()
        logger.info(f"Start: `db_connection`;")
        self.db_connection_async = AsyncQdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=os.getenv("QDRANT_PORT", 6333),
            timeout=3600,
            prefer_grpc=True,
            # prefer_grpc=False,
            https=False,
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.db_connection = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=os.getenv("QDRANT_PORT", 6333),
            timeout=3600,
            prefer_grpc=True,
            # prefer_grpc=False,
            https=False,
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        logger.info(f"Finish: `db_connection`; time: {time.time() - st}")

    def load_documents(
        self, documents: List[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """Load documents into Qdrant vector storage."""
        st = time.time()
        logger.info(f"Start: `load_documents`;")
        points = [
            PointStruct(
                id=document.id, vector=document.vector, payload=document.attributes
            )
            for document in documents
            if document.vector is not None
        ]

        if overwrite:
            # Recreate the collection to overwrite it
            self.db_connection.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=len(points[0].vector), distance=models.Distance.COSINE
                ),
            )
        self.db_connection.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Finish: `load_documents`; time: {time.time() - st}")

    def filter_by_id(self, include_ids: List[str] | List[int]) -> Filter:
        """Build a query filter to filter documents by id."""
        st = time.time()
        logger.info(f"Start: `filter_by_id`;")
        conditions = [
            FieldCondition(key="id", match={"value": id_}) for id_ in include_ids
        ]
        output = Filter(must=conditions)
        logger.info(f"Finish: `filter_by_id`; time: {time.time() - st}")
        return output

    def similarity_search_by_vector(
        self, query_embedding: List[float], k: int = 10, **kwargs: Any
    ) -> List[VectorStoreSearchResult]:
        """Perform a vector-based similarity search."""
        st = time.time()
        logger.info(f"Start: `similarity_search_by_vector`")
        logger.info(
            f"[similarity_search_by_vector] self.query_filter: <{self.query_filter}>;"
        )
        search_result = self.db_connection.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True,
            query_filter=self.query_filter if self.query_filter else None,
            with_vectors=True,
        )
        output = [
            VectorStoreSearchResult(
                document=VectorStoreDocument(
                    id=point.id,
                    text=point.payload.get("text"),
                    vector=point.vector,
                    attributes=point.payload,
                ),
                score=point.score,
            )
            for point in search_result
        ]
        logger.info(f"Finish: `similarity_search_by_vector`; time: {time.time() - st}")
        return output

    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> List[VectorStoreSearchResult]:
        """Perform a similarity search using a given input text."""
        st = time.time()
        logger.info(f"Start: `similarity_search_by_text`")
        output = []
        query_embedding = text_embedder(text)
        if query_embedding:
            output = self.similarity_search_by_vector(query_embedding, k)
        logger.info(f"Finish: `similarity_search_by_text`; time: {time.time() - st}")
        return output

    async def aload_documents(
        self, documents: List[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """Load documents into Qdrant vector storage."""
        st = time.time()
        logger.info(f"Start: `load_documents`;")
        points = [
            PointStruct(
                id=document.id, vector=document.vector, payload=document.attributes
            )
            for document in documents
            if document.vector is not None
        ]

        if overwrite:
            # Recreate the collection to overwrite it
            await self.db_connection_async.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=len(points[0].vector), distance=models.Distance.COSINE
                ),
            )
        await self.db_connection_async.upsert(
            collection_name=self.collection_name, points=points
        )
        logger.info(f"Finish: `load_documents`; time: {time.time() - st}")

    async def afilter_by_id(self, include_ids: List[str] | List[int]) -> Filter:
        """Build a query filter to filter documents by id."""
        st = time.time()
        logger.info(f"Start: `filter_by_id`;")
        conditions = [
            FieldCondition(key="id", match={"value": id_}) for id_ in include_ids
        ]
        output = Filter(must=conditions)
        logger.info(f"Finish: `filter_by_id`; time: {time.time() - st}")
        return output

    async def asimilarity_search_by_vector(
        self, query_embedding: List[float], k: int = 10, **kwargs: Any
    ) -> List[VectorStoreSearchResult]:
        """Perform a vector-based similarity search."""
        st = time.time()
        logger.info(f"Start: `similarity_search_by_vector`")
        logger.info(
            f"[similarity_search_by_vector] self.query_filter: <{self.query_filter}>;"
        )
        search_result = await self.db_connection_async.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True,
            query_filter=self.query_filter if self.query_filter else None,
            with_vectors=True,
        )
        output = [
            VectorStoreSearchResult(
                document=VectorStoreDocument(
                    id=point.id,
                    text=point.payload.get("text"),
                    vector=point.vector,
                    attributes=point.payload,
                ),
                score=point.score,
            )
            for point in search_result
        ]
        logger.info(f"Finish: `similarity_search_by_vector`; time: {time.time() - st}")
        return output

    async def asimilarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> List[VectorStoreSearchResult]:
        """Perform a similarity search using a given input text."""
        st = time.time()
        logger.info(f"Start: `similarity_search_by_text`")
        output = []
        query_embedding = text_embedder(text)
        if query_embedding:
            output = await self.asimilarity_search_by_vector(query_embedding, k)
        logger.info(f"Finish: `similarity_search_by_text`; time: {time.time() - st}")
        return output
