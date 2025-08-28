"""Util functions to get entities from a collection."""

import uuid
from typing import Any, cast

import pandas as pd

from apps.knowledge_manager.src.graphrag.model import Entity
from libs.python.databases.database import ExperimentEntities
from libs.python.utils.logger import logger


async def get_entity_by_key(
    key: str, value: str | int, experiment_id: str, community_level: int
) -> Entity | None:
    """Get entity by key."""

    key = key.replace(f"EntityVectorStoreKey.", "").lower()

    if key == "id":
        value = value.replace("-", "")

    list_ents = await ExperimentEntities.query.where(
        (ExperimentEntities.experiment_id == experiment_id)
        & (ExperimentEntities.community_level == community_level)
        & (getattr(ExperimentEntities, key) == value)
    ).gino.all()

    q_str = f"ExperimentEntities.query.where((ExperimentEntities.experiment_id == {experiment_id}) & (ExperimentEntities.community_level == {community_level}) & (ExperimentEntities[{key}] == {value})).gino.all()"
    if list_ents is None:
        logger.warning(f"NONE: {q_str}")
        return None

    if len(list_ents) == 1:
        logger.info(f"SUCCESS: {q_str}")
        ent = Entity(
            id=list_ents[0].id,
            short_id=list_ents[0].short_id,
            title=list_ents[0].title,
            type=list_ents[0].type,
            description=list_ents[0].description,
            description_embedding=list_ents[0].description_embedding,
            name_embedding=list_ents[0].name_embedding,
            graph_embedding=list_ents[0].graph_embedding,
            community_ids=list_ents[0].community_ids,
            text_unit_ids=list_ents[0].text_unit_ids,
            document_ids=list_ents[0].document_ids,
            rank=list_ents[0].rank,
            attributes=list_ents[0].attributes,
        )
        return ent

    elif len(list_ents) > 1:
        logger.warning(f"len(list_ents) > 1: {q_str}")
        return None

    else:
        logger.warning(f"len(list_ents) < 1: {q_str}")
        return None


async def get_entity_by_name(
    experiment_id: str, community_level: int, entity_name: str
) -> list[Entity]:
    """Get entities by name."""
    list_ents = await ExperimentEntities.query.where(
        (ExperimentEntities.experiment_id == experiment_id)
        & (ExperimentEntities.community_level == community_level)
        & (ExperimentEntities.title == entity_name)
    ).gino.all()

    q_str = f"ExperimentEntities.query.where((ExperimentEntities.experiment_id == {experiment_id}) & (ExperimentEntities.community_level == {community_level}) & (ExperimentEntities.title == {entity_name})).gino.all()"

    if list_ents is None:
        logger.warning(f"NONE: {q_str}")
        return []

    if len(list_ents) == 0:
        logger.warning(f"len(list_ents) == 0: {q_str}")
        return []

    entities = [
        Entity(
            id=ent.id,
            short_id=ent.short_id,
            title=ent.title,
            type=ent.type,
            description=ent.description,
            description_embedding=ent.description_embedding,
            name_embedding=ent.name_embedding,
            graph_embedding=ent.graph_embedding,
            community_ids=ent.community_ids,
            text_unit_ids=ent.text_unit_ids,
            document_ids=ent.document_ids,
            rank=ent.rank,
            attributes=ent.attributes,
        )
        for ent in list_ents
    ]
    logger.info(f"SUCCESS: {q_str}")
    return entities


async def get_entity_by_attribute(
    experiment_id: str, community_level: int, attribute_name: str, attribute_value: Any
) -> list[Entity]:
    """Get entities by attribute."""

    if ExperimentEntities.attributes is None:
        condit_ = ExperimentEntities.attributes is None
    else:
        condit_ = ExperimentEntities.attributes.get(attribute_name) == attribute_value

    list_ents = await ExperimentEntities.query.where(
        (ExperimentEntities.experiment_id == experiment_id)
        & (ExperimentEntities.community_level == community_level)
        & condit_
    ).gino.all()

    q_str = f"ExperimentEntities.query.where((ExperimentEntities.experiment_id == {experiment_id}) & (ExperimentEntities.community_level == {community_level}) & (ExperimentEntities.attributes.get(attribute_name) == {attribute_value})).gino.all()"

    if list_ents is None:
        logger.warning(f"NONE: {q_str}")
        return []

    if len(list_ents) == 0:
        logger.warning(f"len(list_ents) == 0: {q_str}")
        return []

    entities = [
        Entity(
            id=ent.id,
            short_id=ent.short_id,
            title=ent.title,
            type=ent.type,
            description=ent.description,
            description_embedding=ent.description_embedding,
            name_embedding=ent.name_embedding,
            graph_embedding=ent.graph_embedding,
            community_ids=ent.community_ids,
            text_unit_ids=ent.text_unit_ids,
            document_ids=ent.document_ids,
            rank=ent.rank,
            attributes=ent.attributes,
        )
        for ent in list_ents
    ]
    logger.info(f"SUCCESS: {q_str}")
    return entities


def to_entity_dataframe(
    entities: list[Entity],
    include_entity_rank: bool = True,
    rank_description: str = "number of relationships",
) -> pd.DataFrame:
    """Convert a list of entities to a pandas dataframe."""
    if len(entities) == 0:
        return pd.DataFrame()
    header = ["id", "entity", "description"]
    if include_entity_rank:
        header.append(rank_description)
    attribute_cols = (
        list(entities[0].attributes.keys()) if entities[0].attributes else []
    )
    attribute_cols = [col for col in attribute_cols if col not in header]
    header.extend(attribute_cols)

    records = []
    for entity in entities:
        new_record = [
            entity.short_id if entity.short_id else "",
            entity.title,
            entity.description if entity.description else "",
        ]
        if include_entity_rank:
            new_record.append(str(entity.rank))

        for field in attribute_cols:
            field_value = (
                str(entity.attributes.get(field))
                if entity.attributes and entity.attributes.get(field)
                else ""
            )
            new_record.append(field_value)
        records.append(new_record)
    return pd.DataFrame(records, columns=cast(Any, header))


def is_valid_uuid(value: str) -> bool:
    """Determine if a string is a valid UUID."""
    try:
        uuid.UUID(str(value))
    except ValueError:
        return False
    else:
        return True
