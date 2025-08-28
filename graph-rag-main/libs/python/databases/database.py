import os
import uuid
from datetime import datetime

from gino import Gino
from sqlalchemy.dialects.postgresql import JSONB

from libs.python.utils.logger import logger

db = Gino()


class GoogleSearchResults(db.Model):
    __tablename__ = "google_search_results"

    id = db.Column(db.String(), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(), nullable=True)
    session_id = db.Column(db.String(), nullable=True)
    dialog_id = db.Column(db.String(), nullable=True)
    type = db.Column(db.String())
    google_query = db.Column(db.String())
    results = db.Column(JSONB, nullable=False, server_default="{}")
    domain_filter = db.Column(JSONB, nullable=False, server_default="{}")

    top_k_url = db.Column(db.Integer())
    country = db.Column(db.String())
    locale = db.Column(db.String())
    datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())


class ParserResults(db.Model):
    __tablename__ = "parser_results"

    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = db.Column(db.String())
    url = db.Column(db.String())
    type = db.Column(db.String())

    batch_id = db.Column(db.String(), nullable=True)
    batch_input_count = db.Column(db.Integer(), nullable=True)

    html_content = db.Column(db.String(), nullable=True)

    content = db.Column(db.String(), nullable=True)

    is_parsed = db.Column(db.Boolean())

    google_search_results_id = db.Column(
        db.String, db.ForeignKey("google_search_results.id")
    )

    datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())
    datetime_updated = db.Column(db.DateTime())


class ParserVerificationResults(db.Model):
    __tablename__ = "parser_verification_results"

    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))

    batch_id = db.Column(db.String(), nullable=True)
    batch_input_count = db.Column(db.Integer(), nullable=True)

    verify_type = db.Column(db.String(), nullable=True)
    verified_content_summarization = db.Column(db.String(), nullable=True)
    is_provided_answer_useful = db.Column(db.Boolean(), nullable=True)
    google_query = db.Column(db.String())

    parser_results_id = db.Column(db.String, db.ForeignKey("parser_results.id"))
    datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())
    datetime_updated = db.Column(db.DateTime())


class GoogleSearchParserVerificationCalls(db.Model):
    __tablename__ = "google_search_parser_verification_calls"

    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))

    parser_results_id = db.Column(
        db.String, db.ForeignKey("parser_results.id"), nullable=True
    )
    google_search_results_id = db.Column(
        db.String, db.ForeignKey("google_search_results.id")
    )
    parser_verification_results_id = db.Column(
        db.String, db.ForeignKey("parser_verification_results.id"), nullable=True
    )
    behavior_graph_db = db.Column(db.String(), nullable=True)
    use_output_from_verif_as_content = db.Column(db.Boolean(), nullable=True)
    batch_id = db.Column(db.String(), nullable=True)


class GoogleSearchParserTitleSim(db.Model):
    """
    if pair (google_search_results_id, parser_results_id) already exists then do not add this pair again
    """

    __tablename__ = "google_search_parser_title_sim"

    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    parser_results_id = db.Column(db.String, db.ForeignKey("parser_results.id"))
    google_search_results_id = db.Column(
        db.String, db.ForeignKey("google_search_results.id")
    )
    google_query = db.Column(db.String())
    title = db.Column(db.String())
    title_query_sim_ada_2_cosine = db.Column(db.Float(), nullable=True)


class RedditResults(db.Model):
    __tablename__ = "reddit_results"

    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title_post = db.Column(db.String(), nullable=True)

    data_type = db.Column(db.String())

    community_name = db.Column(db.String())
    body = db.Column(db.String())
    number_of_comments = db.Column(db.Integer())

    username = db.Column(db.String())

    number_of_replies = db.Column(db.Integer())

    up_votes = db.Column(db.Integer())

    post_datetime_created = db.Column(db.DateTime(), nullable=True)

    body_query_sim_ada_2_cosine = db.Column(db.Float(), nullable=True)
    title_post_query_sim_ada_2_cosine = db.Column(db.Float(), nullable=True)

    datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())

    parser_results_id = db.Column(db.String, db.ForeignKey("parser_results.id"))


class GraphUpdateNotification(db.Model):
    __tablename__ = "graph_update_notification"

    id = db.Column(db.String(), primary_key=True)
    user_id = db.Column(db.String(), nullable=True)
    session_id = db.Column(db.String(), nullable=True)
    dialog_id = db.Column(db.String(), nullable=True)

    batch_id = db.Column(db.String(), nullable=True)
    batch_input_count = db.Column(db.Integer(), nullable=True)

    parser_results_id = db.Column(db.String, db.ForeignKey("parser_results.id"))

    src = db.Column(db.String(), nullable=True)
    status = db.Column(db.String(), nullable=True)
    verify_layer_flag = db.Column(db.Boolean())
    space_name = db.Column(db.String(), nullable=True)
    system_content_prompt = db.Column(db.String(), nullable=True)
    content = db.Column(db.String(), nullable=True)
    add_basic_nodes = db.Column(db.Boolean(), nullable=True)
    create_basic_structures = db.Column(db.Boolean(), nullable=True)
    behavior_graph_db = db.Column(db.String(), nullable=True)
    is_need_upd_scheme_during_extract_graph = db.Column(db.Boolean(), nullable=True)
    datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())
    datetime_updated = db.Column(db.DateTime(), default=datetime.utcnow())
    metadata = db.Column(JSONB, nullable=False, server_default="{}")


class SchemeUpdateNotification(db.Model):
    __tablename__ = "scheme_update_notification"

    id = db.Column(db.String(), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(), nullable=True)
    session_id = db.Column(db.String(), nullable=True)
    dialog_id = db.Column(db.String(), nullable=True)

    batch_id = db.Column(db.String(), nullable=True)

    parser_results_id = db.Column(db.String, db.ForeignKey("parser_results.id"))

    addon_scheme = db.Column(JSONB, nullable=False, server_default="{}")
    google_query = db.Column(db.String())
    status = db.Column(db.Boolean())
    verify_layer_flag = db.Column(db.Boolean())
    src = db.Column(db.String(), nullable=True)
    space_name = db.Column(db.String(), nullable=True)
    batch_input_count = db.Column(db.Integer(), nullable=True)
    is_used_upd_scheme_db = db.Column(db.Boolean())

    behavior_graph_db = db.Column(db.String(), nullable=True)

    request_to_update_graph = db.Column(JSONB, nullable=False, server_default="{}")

    datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())
    datetime_updated = db.Column(db.DateTime(), default=datetime.utcnow())


# class ExperimentCreateFinalNodes(db.Model):
#     __tablename__ = "create_final_nodes"
#
#     global_id = db.Column(
#         db.String(), primary_key=True, default=lambda: str(uuid.uuid4())
#     )
#     experiment_id = db.Column(db.String())
#
#     level = db.Column(db.Integer(), nullable=True)
#     title = db.Column(db.String(), nullable=True)
#     type = db.Column(db.String(), nullable=True)
#     description = db.Column(db.String(), nullable=True)
#     source_id = db.Column(db.String(), nullable=True)
#     community = db.Column(db.String(), nullable=True)
#     degree = db.Column(db.Integer(), nullable=True)
#     human_readable_id = db.Column(db.Integer(), nullable=True)
#     id = db.Column(db.String(), nullable=True)
#     size = db.Column(db.Integer(), nullable=True)
#     graph_embedding = db.Column(JSONB, nullable=False, server_default="[]")
#     entity_type = db.Column(db.String(), nullable=True)
#     top_level_node_id = db.Column(db.String(), nullable=True)
#     x = db.Column(db.Integer(), nullable=True)
#     y = db.Column(db.Integer(), nullable=True)
#
#     datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())
#     datetime_updated = db.Column(db.DateTime(), default=datetime.utcnow())
#
#
# class ExperimentCreateFinalCommunityReports(db.Model):
#     __tablename__ = "create_final_community_reports"
#
#     global_id = db.Column(
#         db.String(), primary_key=True, default=lambda: str(uuid.uuid4())
#     )
#     experiment_id = db.Column(db.String())
#
#     community = db.Column(db.String(), nullable=True)
#     full_content = db.Column(db.String(), nullable=True)
#     level = db.Column(db.Integer(), nullable=True)
#     rank = db.Column(db.Float(), nullable=True)
#     title = db.Column(db.String(), nullable=True)
#     rank_explanation = db.Column(db.String(), nullable=True)
#     summary = db.Column(db.String(), nullable=True)
#     findings = db.Column(JSONB, nullable=False, server_default="[]")
#     full_content_json = db.Column(JSONB, nullable=False, server_default="{}")
#     id = db.Column(db.String(), nullable=True)
#
#     datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())
#     datetime_updated = db.Column(db.DateTime(), default=datetime.utcnow())
#
#
# class ExperimentCreateFinalRelationships(db.Model):
#     __tablename__ = "create_final_relationships"
#
#     global_id = db.Column(
#         db.String(), primary_key=True, default=lambda: str(uuid.uuid4())
#     )
#     experiment_id = db.Column(db.String())
#
#     source = db.Column(db.String(), nullable=True)
#     target = db.Column(db.String(), nullable=True)
#     weight = db.Column(db.Float(), nullable=True)
#     description = db.Column(db.String(), nullable=True)
#     text_unit_ids = db.Column(JSONB, nullable=False, server_default="[]")
#     id = db.Column(db.String(), nullable=True)
#     human_readable_id = db.Column(db.String(), nullable=True)
#     source_degree = db.Column(db.Integer(), nullable=True)
#     target_degree = db.Column(db.Integer(), nullable=True)
#     rank = db.Column(db.Integer(), nullable=True)
#
#     datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())
#     datetime_updated = db.Column(db.DateTime(), default=datetime.utcnow())
#
#
# class ExperimentCreateFinalCovariates(db.Model):
#     __tablename__ = "create_final_covariates"
#
#     global_id = db.Column(
#         db.String(), primary_key=True, default=lambda: str(uuid.uuid4())
#     )
#     experiment_id = db.Column(db.String())
#
#     id = db.Column(db.String(), nullable=True)
#     human_readable_id = db.Column(db.String(), nullable=True)
#     covariate_type = db.Column(db.String(), nullable=True)
#     type = db.Column(db.String(), nullable=True)
#     description = db.Column(db.String(), nullable=True)
#     subject_id = db.Column(db.String(), nullable=True)
#     subject_type = db.Column(db.String(), nullable=True)
#     object_id = db.Column(db.String(), nullable=True)
#     object_type = db.Column(db.String(), nullable=True)
#     status = db.Column(db.String(), nullable=True)
#
#     start_date = db.Column(db.String(), nullable=True)
#     end_date = db.Column(db.String(), nullable=True)
#
#     source_text = db.Column(db.String(), nullable=True)
#     text_unit_id = db.Column(db.String(), nullable=True)
#     document_ids = db.Column(JSONB, nullable=False, server_default="[]")
#     n_tokens = db.Column(db.Integer(), nullable=True)
#
#     datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())
#     datetime_updated = db.Column(db.DateTime(), default=datetime.utcnow())
#
#
# class ExperimentCreateFinalTextUnits(db.Model):
#     __tablename__ = "create_final_text_units"
#
#     global_id = db.Column(
#         db.String(), primary_key=True, default=lambda: str(uuid.uuid4())
#     )
#     experiment_id = db.Column(db.String())
#
#     id = db.Column(db.String(), nullable=True)
#     text = db.Column(db.String(), nullable=True)
#     n_tokens = db.Column(db.Integer(), nullable=True)
#     document_ids = db.Column(JSONB, nullable=False, server_default="[]")
#     entity_ids = db.Column(JSONB, nullable=False, server_default="[]")
#     relationship_ids = db.Column(JSONB, nullable=False, server_default="[]")
#     covariate_ids = db.Column(JSONB, nullable=False, server_default="[]")
#
#     datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())
#     datetime_updated = db.Column(db.DateTime(), default=datetime.utcnow())


class ExperimentEntities(db.Model):
    __tablename__ = "entities"
    global_id = db.Column(
        db.String(), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    experiment_id = db.Column(db.String())
    community_level = db.Column(db.Integer())

    attributes = db.Column(JSONB, nullable=True)
    community_ids = db.Column(JSONB, nullable=True)
    description = db.Column(db.String(), nullable=True)
    description_embedding = db.Column(JSONB, nullable=True)
    document_ids = db.Column(JSONB, nullable=True)
    graph_embedding = db.Column(JSONB, nullable=True)
    id = db.Column(db.String(), nullable=True)
    name_embedding = db.Column(db.String(), nullable=True)
    rank = db.Column(db.Integer(), nullable=True)
    short_id = db.Column(db.String(), nullable=True)
    text_unit_ids = db.Column(JSONB, nullable=True)
    title = db.Column(db.String(), nullable=True)
    type = db.Column(db.String(), nullable=True)

    datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())
    datetime_updated = db.Column(db.DateTime(), default=datetime.utcnow())

    _idx1 = db.Index(
        "entities_idx_experiment_community", "experiment_id", "community_level"
    )
    _idx2 = db.Index("entities_idx_id", "id")
    _idx3 = db.Index("entities_idx_title", "title")
    _idx4 = db.Index(
        "entities_idx_experiment_community_id", "experiment_id", "community_level", "id"
    )
    _idx5 = db.Index(
        "entities_idx_attributes_gin", "attributes", postgresql_using="gin"
    )


class ExperimentReports(db.Model):
    __tablename__ = "reports"

    global_id = db.Column(
        db.String(), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    experiment_id = db.Column(db.String())
    community_level = db.Column(db.Integer())

    attributes = db.Column(JSONB, nullable=True, default=None)
    community_id = db.Column(db.String())
    full_content = db.Column(db.String(), default="")
    full_content_embedding = db.Column(JSONB, nullable=True, default=None)
    id = db.Column(db.String())
    rank = db.Column(db.Float(), nullable=True, default=1.0)
    short_id = db.Column(db.String(), nullable=True)
    summary = db.Column(db.String(), default="")
    summary_embedding = db.Column(JSONB, nullable=True, default=None)
    title = db.Column(db.String())

    datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())
    datetime_updated = db.Column(db.DateTime(), default=datetime.utcnow())

    _idx1 = db.Index(
        "reports_idx_experiment_community", "experiment_id", "community_level"
    )
    _idx2 = db.Index("reports_idx_community_id", "community_id")
    _idx3 = db.Index(
        "reports_idx_experiment_community_community_id",
        "experiment_id",
        "community_level",
        "community_id",
    )
    _idx4 = db.Index("reports_idx_attributes_gin", "attributes", postgresql_using="gin")


class ExperimentRelationships(db.Model):
    __tablename__ = "relationships"

    global_id = db.Column(
        db.String(), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    experiment_id = db.Column(db.String())
    community_level = db.Column(db.Integer())

    attributes = db.Column(JSONB, nullable=True, default=None)
    description = db.Column(db.String(), nullable=True, default=None)
    description_embedding = db.Column(JSONB, nullable=True, default=None)
    document_ids = db.Column(JSONB, nullable=True, default=None)
    id = db.Column(db.String())
    short_id = db.Column(db.String(), nullable=True)
    source = db.Column(db.String())
    target = db.Column(db.String())
    text_unit_ids = db.Column(JSONB, nullable=True, default=None)
    weight = db.Column(db.Float(), nullable=True, default=1.0)

    datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())
    datetime_updated = db.Column(db.DateTime(), default=datetime.utcnow())

    _idx1 = db.Index(
        "relationships_idx_experiment_community", "experiment_id", "community_level"
    )
    _idx2 = db.Index("relationships_idx_source", "source")
    _idx3 = db.Index("relationships_idx_target", "target")
    _idx4 = db.Index(
        "relationships_idx_experiment_community_source_target",
        "experiment_id",
        "community_level",
        "source",
        "target",
    )
    _idx5 = db.Index(
        "relationships_idx_attributes_gin", "attributes", postgresql_using="gin"
    )


class ExperimentCovariates(db.Model):
    __tablename__ = "covariates"

    global_id = db.Column(
        db.String(), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    experiment_id = db.Column(db.String())
    community_level = db.Column(db.Integer())

    attributes = db.Column(JSONB, nullable=True, default=None)
    covariate_type = db.Column(db.String(), nullable=False, default="claim")
    document_ids = db.Column(JSONB, nullable=True, default=None)
    id = db.Column(db.String())
    short_id = db.Column(db.String(), nullable=True)
    subject_id = db.Column(db.String())
    subject_type = db.Column(db.String(), nullable=False, default="entity")
    text_unit_ids = db.Column(JSONB, nullable=True, default=None)

    datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())
    datetime_updated = db.Column(db.DateTime(), default=datetime.utcnow())

    _idx1 = db.Index(
        "covariates_idx_experiment_community_type",
        "experiment_id",
        "community_level",
        "covariate_type",
    )
    _idx2 = db.Index("covariates_idx_subject_id_type", "subject_id", "subject_type")
    _idx3 = db.Index(
        "covariates_idx_attributes_gin", "attributes", postgresql_using="gin"
    )


class ExperimentTextUnits(db.Model):
    __tablename__ = "text_units"

    global_id = db.Column(
        db.String(), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    experiment_id = db.Column(db.String())
    community_level = db.Column(db.Integer())

    attributes = db.Column(JSONB, nullable=True, default=None)
    covariate_dis = db.Column(JSONB, nullable=True, default=None)
    document_ids = db.Column(JSONB, nullable=True, default=None)
    entity_ids = db.Column(JSONB, nullable=True, default=None)
    id = db.Column(db.String())
    n_tokens = db.Column(db.Integer(), nullable=True, default=None)
    relationship_ids = db.Column(JSONB, nullable=True, default=None)
    short_id = db.Column(db.String(), nullable=True)
    text = db.Column(db.String())
    text_embedding = db.Column(JSONB, nullable=True, default=None)

    datetime_created = db.Column(db.DateTime(), default=datetime.utcnow())
    datetime_updated = db.Column(db.DateTime(), default=datetime.utcnow())

    _idx1 = db.Index(
        "text_units_idx_experiment_community", "experiment_id", "community_level"
    )
    # _idx2 = db.Index("text_units_idx_text_gin", "text", postgresql_using="gin")
    # _idx3 = db.Index(
    #     "text_units_idx_entity_ids_gin", "entity_ids", postgresql_using="gin"
    # )
    _idx4 = db.Index(
        "text_units_idx_relationship_ids_gin",
        "relationship_ids",
        postgresql_using="gin",
    )
    # _idx5 = db.Index(
    #     "text_units_idx_document_ids_gin", "document_ids", postgresql_using="gin"
    # )


POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")

postgresql_db_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
logger.info(
    f"Postgresql connection:\n"
    f" POSTGRES_HOST: {POSTGRES_HOST}\n"
    f" POSTGRES_PORT: {POSTGRES_PORT}\n"
    f" POSTGRES_DB: {POSTGRES_DB}"
)
