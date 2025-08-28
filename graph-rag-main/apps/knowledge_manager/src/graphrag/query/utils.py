from apps.knowledge_manager.src.graphrag.model import (
    CommunityReport,
    Entity,
    Relationship,
    Covariate,
)
from libs.python.databases.database import (
    ExperimentReports,
    ExperimentEntities,
    ExperimentRelationships,
    ExperimentCovariates,
)
from libs.python.utils.logger import logger


async def get_all_reports(
    experiment_id: str, community_level: int, as_dict: bool = True
) -> dict[str, CommunityReport] | list[CommunityReport]:
    community_reports_from_table = await ExperimentReports.query.where(
        (ExperimentReports.experiment_id == experiment_id)
        & (ExperimentReports.community_level == community_level)
    ).gino.all()
    q_str = f"ExperimentReports.query.where((ExperimentReports.experiment_id == {experiment_id}) & (ExperimentReports.community_level == {community_level})).gino.all()"
    if community_reports_from_table is None:
        logger.warning(f"NONE: {q_str}")
        return dict() if as_dict else list()
    elif len(community_reports_from_table) == 0:
        logger.warning(f"len(community_reports_from_table) == 0: {q_str}")
        return dict() if as_dict else list()
    else:
        logger.info(f"SUCCESS: {q_str}")
        return (
            {
                cr: CommunityReport(
                    id=cr.id,
                    short_id=cr.short_id,
                    title=cr.title,
                    community_id=cr.community_id,
                    summary=cr.summary,
                    full_content=cr.full_content,
                    rank=cr.rank,
                    summary_embedding=cr.summary_embedding,
                    full_content_embedding=cr.full_content_embedding,
                    attributes=cr.attributes,
                )
                for cr in community_reports_from_table
            }
            if as_dict
            else [
                CommunityReport(
                    id=cr.id,
                    short_id=cr.short_id,
                    title=cr.title,
                    community_id=cr.community_id,
                    summary=cr.summary,
                    full_content=cr.full_content,
                    rank=cr.rank,
                    summary_embedding=cr.summary_embedding,
                    full_content_embedding=cr.full_content_embedding,
                    attributes=cr.attributes,
                )
                for cr in community_reports_from_table
            ]
        )


async def get_all_entities(
    experiment_id: str, community_level: int, as_dict: bool = True
) -> dict[str, Entity] | list[Entity]:
    entities_from_table = await ExperimentEntities.query.where(
        (ExperimentEntities.experiment_id == experiment_id)
        & (ExperimentEntities.community_level == community_level)
    ).gino.all()

    q_str = f"ExperimentEntities.query.where((ExperimentEntities.experiment_id == {experiment_id}) & (ExperimentEntities.community_level == {community_level})).gino.all()"

    if entities_from_table is None:
        logger.warning(f"NONE: {q_str}")
        return dict() if as_dict else list()
    elif len(entities_from_table) == 0:
        logger.warning(f"len(entities_from_table) == 0: {q_str}")
        return dict() if as_dict else list()
    else:
        logger.info(f"SUCCESS: {q_str}")
        return (
            {
                ent: Entity(
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
                for ent in entities_from_table
            }
            if as_dict
            else [
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
                for ent in entities_from_table
            ]
        )


async def get_all_relationships(
    experiment_id: str, community_level: int, as_dict: bool = True
) -> dict[str, Relationship] | list[Relationship]:
    rels_from_table = await ExperimentRelationships.query.where(
        (ExperimentRelationships.experiment_id == experiment_id)
        & (ExperimentRelationships.community_level == community_level)
    ).gino.all()

    q_str = f"ExperimentRelationships.query.where((ExperimentRelationships.experiment_id == {experiment_id}) & (ExperimentRelationships.community_level == {community_level})).gino.all()"

    if rels_from_table is None:
        logger.warning(f"NONE: {q_str}")
        return dict() if as_dict else list()
    elif len(rels_from_table) == 0:
        logger.warning(f"len(rels_from_table) == 0: {q_str}")
        return dict() if as_dict else list()
    else:
        logger.info(f"SUCCESS: {q_str}")
        return (
            {
                rel: Relationship(
                    id=rel.id,
                    short_id=rel.short_id,
                    source=rel.source,
                    target=rel.target,
                    weight=rel.weight,
                    description=rel.description,
                    description_embedding=rel.description_embedding,
                    text_unit_ids=rel.text_unit_ids,
                    document_ids=rel.document_ids,
                    attributes=rel.attributes,
                )
                for rel in rels_from_table
            }
            if as_dict
            else [
                Relationship(
                    id=rel.id,
                    short_id=rel.short_id,
                    source=rel.source,
                    target=rel.target,
                    weight=rel.weight,
                    description=rel.description,
                    description_embedding=rel.description_embedding,
                    text_unit_ids=rel.text_unit_ids,
                    document_ids=rel.document_ids,
                    attributes=rel.attributes,
                )
                for rel in rels_from_table
            ]
        )


async def get_all_covariates_by_type(
    experiment_id: str, community_level: int, covariates_type: list[str] = None
) -> dict[str, list[Covariate]]:
    if covariates_type is None:
        covariates_type = ["claim"]

    output = dict()
    for covariate_type in covariates_type:
        covs_from_table = await ExperimentCovariates.query.where(
            (ExperimentCovariates.experiment_id == experiment_id)
            & (ExperimentCovariates.community_level == community_level)
            & (ExperimentCovariates.covariate_type == covariate_type)
        ).gino.all()

        q_str = f"ExperimentCovariates.query.where((ExperimentCovariates.experiment_id == {experiment_id}) & (ExperimentCovariates.community_level == {community_level}) & (ExperimentCovariates.covariate_type == {covariate_type})).gino.all()"

        if covs_from_table is None:
            logger.warning(f"NONE: {q_str}")

        elif len(covs_from_table) == 0:
            logger.warning(f"len(covs_from_table) == 0: {q_str}")

        if covariate_type == "claim":
            covariate_type = "claims"

        if covs_from_table is None:
            output[covariate_type] = list()
        else:
            logger.info(f"SUCCESS: {q_str}")
            output[covariate_type] = [
                Covariate(
                    id=c.id,
                    short_id=c.short_id,
                    subject_id=c.subject_id,
                    subject_type=c.subject_type,
                    covariate_type=c.covariate_type,
                    text_unit_ids=c.text_unit_ids,
                    document_ids=c.document_ids,
                    attributes=c.attributes,
                )
                for c in covs_from_table
            ]

    return output
