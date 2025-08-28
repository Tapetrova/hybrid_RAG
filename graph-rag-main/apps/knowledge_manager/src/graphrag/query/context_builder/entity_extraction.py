"""Orchestration Context Builders."""

from enum import Enum

from apps.knowledge_manager.src.graphrag.model import Entity, Relationship
from apps.knowledge_manager.src.graphrag.query.input.retrieval.entities import (
    get_entity_by_key,
    get_entity_by_name,
)
from apps.knowledge_manager.src.graphrag.query.llm.base import BaseTextEmbedding

# from apps.knowledge_manager.src.graphrag.query.utils import (
#     get_all_entities,
#     get_all_relationships,
# )
from apps.knowledge_manager.src.graphrag.vector_stores import BaseVectorStore


class EntityVectorStoreKey(str, Enum):
    """Keys used as ids in the entity embedding vectorstores."""

    ID = "id"
    TITLE = "title"

    @staticmethod
    def from_string(value: str) -> "EntityVectorStoreKey":
        """Convert string to EntityVectorStoreKey."""
        if value == "id":
            return EntityVectorStoreKey.ID
        if value == "title":
            return EntityVectorStoreKey.TITLE

        msg = f"Invalid EntityVectorStoreKey: {value}"
        raise ValueError(msg)


async def map_query_to_entities(
    query: str,
    experiment_id: str,
    community_level: int,
    text_embedding_vectorstore: BaseVectorStore,
    text_embedder: BaseTextEmbedding,
    embedding_vectorstore_key: str = EntityVectorStoreKey.ID,
    include_entity_names: list[str] | None = None,
    exclude_entity_names: list[str] | None = None,
    k: int = 10,
    oversample_scaler: int = 2,
) -> list[Entity]:
    """Extract entities that match a given query using semantic similarity of text embeddings of query and entity descriptions."""
    if include_entity_names is None:
        include_entity_names = []
    if exclude_entity_names is None:
        exclude_entity_names = []
    matched_entities = []

    # get entities with highest semantic similarity to query
    # oversample to account for excluded entities
    search_results = await text_embedding_vectorstore.asimilarity_search_by_text(
        text=query,
        text_embedder=lambda t: text_embedder.embed(t),
        k=k * oversample_scaler,
    )
    if isinstance(embedding_vectorstore_key, EntityVectorStoreKey):
        embedding_vectorstore_key = str(embedding_vectorstore_key)

    for result in search_results:
        matched = await get_entity_by_key(
            experiment_id=experiment_id,
            community_level=community_level,
            key=embedding_vectorstore_key,
            value=result.document.id,
        )
        if matched:
            matched_entities.append(matched)

    # filter out excluded entities
    if exclude_entity_names:
        matched_entities = [
            entity
            for entity in matched_entities
            if entity.title not in exclude_entity_names
        ]

    # add entities in the include_entity list
    included_entities = []
    for entity_name in include_entity_names:
        ents_by_name = await get_entity_by_name(
            entity_name=entity_name,
            experiment_id=experiment_id,
            community_level=community_level,
        )
        included_entities.extend(ents_by_name)
    return included_entities + matched_entities


async def find_nearest_neighbors_by_graph_embeddings(
    experiment_id: str,
    community_level: int,
    entity_id: str,
    graph_embedding_vectorstore: BaseVectorStore,
    exclude_entity_names: list[str] | None = None,
    embedding_vectorstore_key: str = EntityVectorStoreKey.ID,
    k: int = 10,
    oversample_scaler: int = 2,
) -> list[Entity]:
    """Retrieve related entities by graph embeddings."""
    if exclude_entity_names is None:
        exclude_entity_names = []

    if isinstance(embedding_vectorstore_key, EntityVectorStoreKey):
        embedding_vectorstore_key = str(embedding_vectorstore_key)

    # find nearest neighbors of this entity using graph embedding
    query_entity = await get_entity_by_key(
        experiment_id=experiment_id,
        community_level=community_level,
        key=embedding_vectorstore_key,
        value=entity_id,
    )
    query_embedding = query_entity.graph_embedding if query_entity else None

    # oversample to account for excluded entities
    if query_embedding:
        matched_entities = []
        search_results = graph_embedding_vectorstore.similarity_search_by_vector(
            query_embedding=query_embedding, k=k * oversample_scaler
        )
        for result in search_results:
            matched = await get_entity_by_key(
                experiment_id=experiment_id,
                community_level=community_level,
                key=embedding_vectorstore_key,
                value=result.document.id,
            )
            if matched:
                matched_entities.append(matched)

        # filter out excluded entities
        if exclude_entity_names:
            matched_entities = [
                entity
                for entity in matched_entities
                if entity.title not in exclude_entity_names
            ]
        matched_entities.sort(key=lambda x: x.rank, reverse=True)
        return matched_entities[:k]

    return []


async def find_nearest_neighbors_by_entity_rank(
    all_entities: list[Entity],
    all_relationships: list[Relationship],
    entity_name: str,
    exclude_entity_names: list[str] | None = None,
    k: int | None = 10,
) -> list[Entity]:
    """Retrieve entities that have direct connections with the target entity, sorted by entity rank."""

    # all_entities = await get_all_entities(
    #     experiment_id=experiment_id, community_level=community_level, as_dict=False
    # )
    # all_relationships = await get_all_relationships(
    #     experiment_id=experiment_id, community_level=community_level, as_dict=False
    # )

    if exclude_entity_names is None:
        exclude_entity_names = []
    entity_relationships = [
        rel
        for rel in all_relationships
        if rel.source == entity_name or rel.target == entity_name
    ]
    source_entity_names = {rel.source for rel in entity_relationships}
    target_entity_names = {rel.target for rel in entity_relationships}
    related_entity_names = (source_entity_names.union(target_entity_names)).difference(
        set(exclude_entity_names)
    )
    top_relations = [
        entity for entity in all_entities if entity.title in related_entity_names
    ]
    top_relations.sort(key=lambda x: x.rank if x.rank else 0, reverse=True)
    if k:
        return top_relations[:k]
    return top_relations
