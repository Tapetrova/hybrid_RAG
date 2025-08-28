import asyncio
import json
import math
import traceback
import uuid
from dataclasses import asdict
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from qdrant_openapi_client.models import models

from apps.knowledge_manager.src.graph_rag_manager.utils import (
    _get_progress_reporter,
    S3Helper,
)
from apps.knowledge_manager.src.graphrag.index import (
    create_pipeline_config,
    run_pipeline_with_config,
    PipelineConfig,
)
from apps.knowledge_manager.src.graphrag.index.emit import TableEmitterType
from libs.python.databases.database import (
    db,
    postgresql_db_url,
    ExperimentReports,
    ExperimentRelationships,
    ExperimentEntities,
    ExperimentCovariates,
    ExperimentTextUnits,
)
from libs.python.schemas.configuration import Config
from libs.python.utils.cacher import hash_string
from libs.python.utils.logger import logger
from .celery_app import celery_app
from ..graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_reports,
    read_indexer_relationships,
    read_indexer_covariates,
    read_indexer_text_units,
)
from ..graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from ..graphrag.vector_stores import QdrantVectorStore

# Constants
TEXT_UNITS_TO_RELS = "join_text_units_to_relationship_ids"
TEXT_UNITS_TO_ENTS = "join_text_units_to_entity_ids"
FINAL_DOCS = "create_final_documents"
FINAL_COMMS = "create_final_communities"

COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"


class DataProcessor:
    """Handles data processing utilities"""

    @staticmethod
    def unify_array_to_list(arr: np.ndarray | List) -> List[str]:
        """Convert array or list to unified list of strings"""
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()

        result = set()
        for item in arr:
            result.update(set(item.split(",")))

        return list(result)

    @staticmethod
    def reverse_map_from_list_to_id(
        key_list_of_values: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Create reverse mapping from list values to keys"""
        reversed_map = {}
        for key, values in key_list_of_values.items():
            for value in values:
                if value in reversed_map:
                    reversed_map[value].append(key)
                else:
                    reversed_map[value] = [key]
        return reversed_map

    @staticmethod
    def is_json(value: Any) -> bool:
        """Check if value is valid JSON"""
        try:
            if isinstance(value, str):
                if value.isnumeric():
                    return False
                json.JSONDecoder().decode(value)
                return True
            return False
        except (ValueError, json.JSONDecodeError):
            return False

    @staticmethod
    def serialize_value(value: Any) -> Any:
        """Serialize numpy arrays and JSON strings"""
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, str) and DataProcessor.is_json(value):
            return json.loads(value)
        return value


class DatabaseOperations:
    """Handles database operations"""

    @staticmethod
    async def calculate_batch_size(num_columns: int, max_arguments: int = 32767) -> int:
        """Calculate maximum batch size based on number of columns"""
        return max_arguments // num_columns

    @staticmethod
    async def insert_data_to_table_processed(
        table: db.Model, data: List[Dict], experiment_id: str, **kwargs
    ) -> Dict[str, Any]:
        """Insert processed data into database table with batching"""
        total_records = len(data)
        if total_records == 0:
            message = f"table_name: {table.__tablename__}; experiment_id: {experiment_id}; total_records == 0"
            logger.exception(message)
            return {"status": False, "message": message}

        community_level = kwargs.get("community_level")

        # Check and handle existing data
        exists_data = await table.query.where(
            (table.experiment_id == experiment_id)
            & (table.community_level == community_level)
        ).gino.first()

        if exists_data is not None:
            status_deleting = await table.delete.where(
                (table.experiment_id == experiment_id)
                & (table.community_level == community_level)
            ).gino.status()
            logger.warning(
                f"table_name: {table.__tablename__}; experiment_id[{experiment_id}] and "
                f"community_level[{community_level}] already exists! REMOVE EXIST DATA. "
                f"status_deleting: {status_deleting}"
            )

        # Calculate batch size and insert data
        num_columns = len(data[0])
        batch_size = await DatabaseOperations.calculate_batch_size(num_columns)

        try:
            num_batches = math.ceil(total_records / batch_size)

            for i in range(num_batches):
                batch = data[i * batch_size : (i + 1) * batch_size]

                # Add metadata to each record
                for record in batch:
                    record["experiment_id"] = experiment_id
                    record["community_level"] = community_level

                if batch:
                    async with db.transaction():
                        await table.insert().gino.all(batch)

            message = f"Data inserted successfully into table '{table.__tablename__}'."
            logger.info(message)
            return {"status": True, "data_insertion": message}

        except Exception as e:
            exc = traceback.format_exc()
            message = f"Failed to insert data into table '{table.__tablename__}': {str(e)}::{exc}"
            logger.exception(message)
            return {"status": False, "message": message}


class S3Operations:
    """Handles S3 operations"""

    @staticmethod
    def get_df_from_s3(
        s3_table_name: str,
        s3_helper: S3Helper,
        s3_bucket_name: str,
        s3_folder_path: str,
        timestamp_folder: str,
    ) -> Optional[pd.DataFrame]:
        """Get DataFrame from S3"""
        pre_df_path = f"output/{timestamp_folder}/artifacts/{s3_table_name}.parquet"
        df_path = f"{s3_folder_path}/{pre_df_path}"

        if s3_helper.check_file_existence(
            bucket_name=s3_bucket_name, s3_path=pre_df_path
        ):
            return pd.read_parquet(df_path)
        else:
            logger.error(
                f"DOES NOT EXIST!! bucket_name={s3_bucket_name}, s3_path={pre_df_path}"
            )
            return None


class GraphRAGProcessor:
    """Main processor for GraphRAG operations"""

    def __init__(
        self,
        config: Config,
        s3_helper: S3Helper,
        s3_bucket_name: str,
        s3_folder_path: str,
        timestamp_folder: str,
    ):
        self.config = config
        self.s3_helper = s3_helper
        self.s3_bucket_name = s3_bucket_name
        self.s3_folder_path = s3_folder_path
        self.timestamp_folder = timestamp_folder
        self.community_level = (
            config.config_knowledge_manager.config_retrieval_mode.community_level
        )
        self.embedding_model = (
            config.config_knowledge_manager.config_retrieval_mode.embeddings.llm.model
        )

    async def process_reports(
        self, entity_df: pd.DataFrame, report_df: pd.DataFrame
    ) -> None:
        """Process and upload reports"""
        if entity_df is None or report_df is None:
            logger.error(f"entity_df: {entity_df} or report_df: {report_df} IS NONE")
            return

        reports = read_indexer_reports(report_df, entity_df, self.community_level)
        await DatabaseOperations.insert_data_to_table_processed(
            table=ExperimentReports,
            data=[asdict(r) for r in reports],
            experiment_id=self.timestamp_folder,
            community_level=self.community_level,
        )

    async def process_entities(
        self, entity_df: pd.DataFrame, entity_embedding_df: pd.DataFrame
    ) -> None:
        """Process and upload entities with embeddings"""
        if entity_df is None or entity_embedding_df is None:
            logger.error(
                f"entity_df: {entity_df} or entity_embedding_df: {entity_embedding_df} IS NONE"
            )
            return

        entities = read_indexer_entities(
            entity_df, entity_embedding_df, self.community_level
        )

        # Upload to PostgreSQL
        await DatabaseOperations.insert_data_to_table_processed(
            table=ExperimentEntities,
            data=[asdict(e) for e in entities],
            experiment_id=self.timestamp_folder,
            community_level=self.community_level,
        )

        # Upload to VectorStore
        await self._upload_to_vectorstore(entities)

    async def _upload_to_vectorstore(self, entities: List) -> None:
        """Upload entities to Qdrant vector store"""
        collection_name = (
            f"entity_description_embeddings_{self.timestamp_folder}_"
            f"{self.embedding_model}_community_level_{self.community_level}"
        )

        logger.info(
            "[GraphRAGQueryManager] Start init connection to `QdrantVectorStore`"
        )
        description_embedding_store = QdrantVectorStore(collection_name=collection_name)
        description_embedding_store.connect()
        logger.info(
            "[GraphRAGQueryManager] Finish init connection to `QdrantVectorStore`"
        )

        if description_embedding_store.db_connection.collection_exists(collection_name):
            description_embedding_store.db_connection.recreate_collection(
                collection_name,
                vectors_config=models.VectorParams(
                    size=len(entities[0].description_embedding),
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(
                f"[GraphRAGQueryManager] Collection Name: {collection_name} WAS RECREATED!"
            )
        else:
            description_embedding_store.db_connection.create_collection(
                collection_name,
                vectors_config=models.VectorParams(
                    size=len(entities[0].description_embedding),
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(
                f"[GraphRAGQueryManager] Collection Name: {collection_name} WAS CREATED!"
            )

        store_entity_semantic_embeddings(
            entities=entities, vectorstore=description_embedding_store
        )

    async def process_relationships(self, relationship_df: pd.DataFrame) -> None:
        """Process and upload relationships"""
        if relationship_df is None:
            logger.error("relationship_df IS NONE")
            return

        relationships = read_indexer_relationships(relationship_df)
        await DatabaseOperations.insert_data_to_table_processed(
            table=ExperimentRelationships,
            data=[asdict(r) for r in relationships],
            experiment_id=self.timestamp_folder,
            community_level=self.community_level,
        )

    async def process_covariates(self, covariate_df: pd.DataFrame) -> None:
        """Process and upload covariates"""
        if covariate_df is None:
            logger.error("covariate_df IS NONE")
            return

        covariates = read_indexer_covariates(covariate_df)
        await DatabaseOperations.insert_data_to_table_processed(
            table=ExperimentCovariates,
            data=[asdict(c) for c in covariates],
            experiment_id=self.timestamp_folder,
            community_level=self.community_level,
        )

    async def process_text_units(self, text_unit_df: pd.DataFrame) -> None:
        """Process and upload text units"""
        if text_unit_df is None:
            logger.error("text_unit_df IS NONE")
            return

        text_units = read_indexer_text_units(text_unit_df)
        await DatabaseOperations.insert_data_to_table_processed(
            table=ExperimentTextUnits,
            data=[asdict(tu) for tu in text_units],
            experiment_id=self.timestamp_folder,
            community_level=self.community_level,
        )


async def from_s3_processed_to_sql_table_data_uploading(
    config: Config,
    s3_helper: S3Helper,
    s3_bucket_name: str,
    s3_folder_path: str,
    timestamp_folder: str,
) -> None:
    """Main function to process and upload data from S3 to SQL and vector stores"""
    processor = GraphRAGProcessor(
        config, s3_helper, s3_bucket_name, s3_folder_path, timestamp_folder
    )

    # Get dataframes from S3
    entity_df = S3Operations.get_df_from_s3(
        "create_final_nodes",
        s3_helper,
        s3_bucket_name,
        s3_folder_path,
        timestamp_folder,
    )
    report_df = S3Operations.get_df_from_s3(
        "create_final_community_reports",
        s3_helper,
        s3_bucket_name,
        s3_folder_path,
        timestamp_folder,
    )
    entity_embedding_df = S3Operations.get_df_from_s3(
        "create_final_entities",
        s3_helper,
        s3_bucket_name,
        s3_folder_path,
        timestamp_folder,
    )
    relationship_df = S3Operations.get_df_from_s3(
        "create_final_relationships",
        s3_helper,
        s3_bucket_name,
        s3_folder_path,
        timestamp_folder,
    )
    covariate_df = S3Operations.get_df_from_s3(
        "create_final_covariates",
        s3_helper,
        s3_bucket_name,
        s3_folder_path,
        timestamp_folder,
    )
    text_unit_df = S3Operations.get_df_from_s3(
        "create_final_text_units",
        s3_helper,
        s3_bucket_name,
        s3_folder_path,
        timestamp_folder,
    )

    # Process all data types
    await processor.process_reports(entity_df, report_df)
    await processor.process_entities(entity_df, entity_embedding_df)
    await processor.process_relationships(relationship_df)
    await processor.process_covariates(covariate_df)
    await processor.process_text_units(text_unit_df)


async def build_knowledge_maps(
    s3_helper: S3Helper,
    s3_bucket_name: str,
    s3_folder_path: str,
    timestamp_folder: str,
) -> Dict[str, Any]:
    """Build knowledge maps from S3 data"""
    all_maps = {}
    data_processor = DataProcessor()

    # Process text units to relationships
    text_units_to_rels_df = S3Operations.get_df_from_s3(
        TEXT_UNITS_TO_RELS, s3_helper, s3_bucket_name, s3_folder_path, timestamp_folder
    )
    if text_units_to_rels_df is not None:
        all_maps["text_units_to_rels"] = {
            row.id: data_processor.unify_array_to_list(row.relationship_ids)
            for _, row in text_units_to_rels_df.iterrows()
        }
        all_maps["rels_to_text_units"] = data_processor.reverse_map_from_list_to_id(
            all_maps["text_units_to_rels"]
        )

    # Process text units to entities
    text_units_to_ents_df = S3Operations.get_df_from_s3(
        TEXT_UNITS_TO_ENTS, s3_helper, s3_bucket_name, s3_folder_path, timestamp_folder
    )
    if text_units_to_ents_df is not None:
        all_maps["text_units_to_ents"] = {
            row.text_unit_ids: data_processor.unify_array_to_list(row.entity_ids)
            for _, row in text_units_to_ents_df.iterrows()
        }
        all_maps["ents_to_text_units"] = data_processor.reverse_map_from_list_to_id(
            all_maps["text_units_to_ents"]
        )

    # Process final documents
    final_docs_df = S3Operations.get_df_from_s3(
        FINAL_DOCS, s3_helper, s3_bucket_name, s3_folder_path, timestamp_folder
    )
    if final_docs_df is not None:
        all_maps["docs_to_text_units"] = {
            row.id: data_processor.unify_array_to_list(row.text_unit_ids)
            for _, row in final_docs_df.iterrows()
        }
        all_maps["docs_to_source"] = {
            row.id: row.source for _, row in final_docs_df.iterrows()
        }
        all_maps["text_units_to_docs"] = data_processor.reverse_map_from_list_to_id(
            all_maps["docs_to_text_units"]
        )

    # Process final communities
    final_comms_df = S3Operations.get_df_from_s3(
        FINAL_COMMS, s3_helper, s3_bucket_name, s3_folder_path, timestamp_folder
    )
    if final_comms_df is not None:
        all_maps["comms_to_text_units"] = {
            row.id: data_processor.unify_array_to_list(row.text_unit_ids)
            for _, row in final_comms_df.iterrows()
        }
        all_maps["text_units_to_comms"] = data_processor.reverse_map_from_list_to_id(
            all_maps["comms_to_text_units"]
        )

    return all_maps


async def arun_workflow_graph_building(
    knowledge_content: List[Dict[str, str]],
    config: Dict[str, Any],
    timestamp_folder: str,
) -> Dict[str, Any]:
    """Main async function to run graph building workflow"""
    await db.set_bind(postgresql_db_url)
    logger.info("!!!CONNECT TO DB WAS SUCCESSFUL!!!")

    try:
        config = Config(**config)

        # Extract content data
        srcs = [v.get("src") for v in knowledge_content]
        contents = [v.get("text") for v in knowledge_content]
        titles = [v.get("title") for v in knowledge_content]

        logger.info(
            f"start process: run_workflow_graph_building; SRCS: {json.dumps(srcs, indent=2)}"
        )

        # Prepare data
        hash_data = hash_string("".join(sorted(contents)))
        ids = [str(uuid.uuid4()) for _ in range(len(knowledge_content))]

        input_data = pd.DataFrame(
            {
                "id": ids,
                "url": srcs,
                "content": contents,
                "title": [
                    (
                        f"id_{ids[i].split('-')[0]}_hash_data_{hash_data}_src_{srcs[i]}"
                        if titles[i] is None
                        else titles[i]
                    )
                    for i in range(len(ids))
                ],
            }
        )

        # Configure settings
        settings = config.config_knowledge_manager.config_retrieval_mode
        fn = f"hash_data_{hash_data}"
        settings.input.file_pattern = ".*\\.csv$"
        settings.input.file_filter = {f"{fn}.csv": True}

        # Upload to S3
        s3_helper = S3Helper(region_name=settings.input.region_name)
        status_upload, exc = s3_helper.upload_dataframe_to_s3(
            df=input_data,
            bucket_name=settings.input.bucket_name,
            s3_path=f"{settings.input.base_dir}/{fn}.csv",
        )

        if not status_upload:
            return {"status": False, "result": None, "error": exc}

        # Configure storage paths
        settings.reporting.base_dir = f"output/{timestamp_folder}/reports"
        settings.storage.base_dir = f"output/{timestamp_folder}/artifacts"
        settings.cache.base_dir = f"cache_{hash_data}"
        settings.cache.base_name = ""

        # Create and run pipeline
        pipeline_cfg: PipelineConfig = create_pipeline_config(settings=settings)
        progress_reporter = _get_progress_reporter(reporter_type="rich")

        outputs = []
        async for output in run_pipeline_with_config(
            pipeline_cfg,
            run_id=timestamp_folder,
            memory_profile=False,
            progress_reporter=progress_reporter,
            emit=[TableEmitterType.Parquet],
            is_resume_run=False,
        ):
            outputs.append(output)

        # Build knowledge maps
        s3_bucket_name = (
            config.config_knowledge_manager.config_retrieval_mode.storage.bucket_name
        )
        s3_folder_path = f"s3://{s3_bucket_name}"

        all_maps = await build_knowledge_maps(
            s3_helper, s3_bucket_name, s3_folder_path, timestamp_folder
        )

        # Upload maps to S3
        all_maps_path = f"output/{timestamp_folder}/artifacts/all_maps.json"
        s3_helper.upload_json_to_s3(
            json_data=all_maps, bucket_name=s3_bucket_name, s3_path=all_maps_path
        )

        # Process and upload data to storages
        await from_s3_processed_to_sql_table_data_uploading(
            config=config,
            s3_helper=s3_helper,
            s3_bucket_name=s3_bucket_name,
            s3_folder_path=s3_folder_path,
            timestamp_folder=timestamp_folder,
        )

        await db.pop_bind().close()
        logger.info("!!!CONNECTION TO DB WAS CLOSED!!!")
        logger.info("!!!Finish process: run_workflow_graph_building!!!")

        return {"status": True, "result": timestamp_folder}

    except Exception as e:
        exc = traceback.format_exc()
        await db.pop_bind().close()
        logger.info("!!!CONNECTION TO DB WAS CLOSED!!!")
        return {"status": False, "result": None, "error": f"ERROR:{e}:{exc}"}


# Celery tasks
@celery_app.task()
def test_from_s3_processed_to_sql_table_data_uploading(config: Dict[str, Any]):
    """Test task for S3 to SQL data uploading"""

    async def _async_test():
        await db.set_bind(postgresql_db_url)
        logger.info("!!!CONNECT TO DB WAS SUCCESSFUL!!!")
        logger.info("!!!test_sql START!!!")

        config_obj = Config(**config)
        timestamp_folder = (
            config_obj.config_knowledge_manager.config_retrieval_mode.checkpoint_s3_folder_name
        )
        s3_bucket_name = (
            config_obj.config_knowledge_manager.config_retrieval_mode.storage.bucket_name
        )
        s3_folder_path = f"s3://{s3_bucket_name}"
        s3_region_name = (
            config_obj.config_knowledge_manager.config_retrieval_mode.storage.region_name
        )
        s3_helper = S3Helper(region_name=s3_region_name)

        await from_s3_processed_to_sql_table_data_uploading(
            config=config_obj,
            s3_helper=s3_helper,
            s3_bucket_name=s3_bucket_name,
            s3_folder_path=s3_folder_path,
            timestamp_folder=timestamp_folder,
        )

        await db.pop_bind().close()
        logger.info("!!!CONNECTION TO DB WAS CLOSED!!!")
        logger.info("!!!test_sql FINISH!!!")

    return asyncio.run(_async_test())


@celery_app.task()
def run_workflow_graph_building(
    knowledge_content: List[Dict[str, str]],
    config: Dict[str, Any],
    timestamp_folder: str,
):
    """Celery task to run workflow graph building"""
    return asyncio.run(
        arun_workflow_graph_building(
            knowledge_content=knowledge_content,
            config=config,
            timestamp_folder=timestamp_folder,
        )
    )
