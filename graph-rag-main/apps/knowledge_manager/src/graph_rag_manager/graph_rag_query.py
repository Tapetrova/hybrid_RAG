import asyncio
import json
import time
from typing import Any

import tiktoken

# Global search
from apps.knowledge_manager.src.graph_rag_manager.utils import S3Helper
from apps.knowledge_manager.src.graphrag.model import (
    CommunityReport,
    Entity,
    Relationship,
    Covariate,
)
from apps.knowledge_manager.src.graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)

# Local Search
from apps.knowledge_manager.src.graphrag.query.context_builder.entity_extraction import (
    EntityVectorStoreKey,
)
from apps.knowledge_manager.src.graphrag.query.llm.oai.chat_openai import ChatOpenAI
from apps.knowledge_manager.src.graphrag.query.llm.oai.embedding import OpenAIEmbedding
from apps.knowledge_manager.src.graphrag.query.llm.oai.typing import (
    OpenaiApiType,
)
from apps.knowledge_manager.src.graphrag.query.structured_search.base import (
    SearchResult,
)
from apps.knowledge_manager.src.graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from apps.knowledge_manager.src.graphrag.query.structured_search.global_search.search import (
    GlobalSearch,
    GlobalSearchResult,
)
from apps.knowledge_manager.src.graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from apps.knowledge_manager.src.graphrag.query.structured_search.local_search.search import (
    LocalSearch,
)
from apps.knowledge_manager.src.graphrag.query.utils import (
    get_all_reports,
    get_all_entities,
    get_all_covariates_by_type,
    get_all_relationships,
)
from apps.knowledge_manager.src.graphrag.vector_stores import QdrantVectorStore
from libs.python.schemas.configuration import Config
from libs.python.schemas.graphrag.graph_rag_config import SearchMode
from libs.python.utils.graph_rag_utils.format_output_resources import (
    get_resources_from_column_context_data_result,
    add_resources_to_response,
    ContextDataColumn,
)
from libs.python.utils.logger import logger

# from apps.knowledge_manager.src.graphrag.query.question_gen.local_gen import (
#     LocalQuestionGen,
# )

COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
ALL_MAPS = "all_maps"

RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"


class GlobalSearchGraphRAGQueryManager:
    def __init__(
        self,
    ):
        # main engines
        self.global_search_engine = None
        self.global_context_builder = None
        self.local_search_engine = None
        self.local_context_builder = None

        # General Attr
        self.community_level: int | None = None
        self.experiment_id: str | None = None
        self.embedding_model: str | None = None

        # General Additional
        self.checkpoint_s3_folder_name: str | None = None
        self.llm = None
        self.token_encoder = None
        self.text_embedder = None
        self.s3_bucket_name = None
        self.s3_region_name = None
        self.s3_folder_path = None
        self.all_maps = None

        # General Data
        # Global and Local Data
        self.community_reports = None
        self.entities = None

        # Local Data
        self.collection_name = None
        self.description_embedding_store = None

    async def setup_graphrag(self, config: Config):
        st_time_main_super = time.time()
        logger.info(f"START INIT `setup_graphrag`")
        search_mode = config.config_knowledge_manager.config_retrieval_mode.search_mode

        _is_need_update_graphrag = False
        community_level_new = (
            config.config_knowledge_manager.config_retrieval_mode.community_level
        )
        experiment_id_new = (
            config.config_knowledge_manager.config_retrieval_mode.checkpoint_s3_folder_name
        )
        embedding_model_new = (
            config.config_knowledge_manager.config_retrieval_mode.embeddings.llm.model
        )

        if search_mode == SearchMode.local_mode:
            if (self.embedding_model is None) or (
                self.embedding_model != embedding_model_new
            ):
                logger.info(
                    f"NEW embedding_model FROM `{self.embedding_model}` TO {embedding_model_new}"
                )
                _is_need_update_graphrag = True
                self.embedding_model = embedding_model_new

        if (self.community_level is None) or (
            self.community_level != community_level_new
        ):
            logger.info(
                f"NEW community_level: FROM `{self.community_level}` TO {community_level_new}"
            )
            _is_need_update_graphrag = True
            self.community_level = community_level_new

        if (self.experiment_id is None) or (self.experiment_id != experiment_id_new):
            logger.info(
                f"NEW experiment_id: FROM `{self.experiment_id}` TO {experiment_id_new}"
            )
            _is_need_update_graphrag = True
            self.experiment_id = experiment_id_new
            self.checkpoint_s3_folder_name = self.experiment_id

        logger.info(f"_is_need_update_graphrag: {_is_need_update_graphrag}")
        if _is_need_update_graphrag:
            st_time_main = time.time()
            logger.info(f"START: update_graphrag!!!")

            llm_parameters = (
                config.config_knowledge_manager.config_retrieval_mode.global_search.llm_model.llm
            )
            self.llm = ChatOpenAI(
                model=llm_parameters.model,
                api_type=OpenaiApiType.OpenAI,
                max_retries=llm_parameters.max_retries,
                api_base=llm_parameters.api_base,
                organization=llm_parameters.organization,
                request_timeout=llm_parameters.request_timeout,
                api_version=llm_parameters.api_version,
                deployment_name=llm_parameters.deployment_name,
            )
            self.token_encoder = tiktoken.encoding_for_model(
                model_name=(
                    llm_parameters.model
                    if isinstance(llm_parameters.model, str)
                    or (llm_parameters.model is None)
                    else llm_parameters.model.value
                )
            )

            community_reports = await get_all_reports(
                experiment_id=self.experiment_id,
                community_level=self.community_level,
                as_dict=True,
            )
            entities = await get_all_entities(
                experiment_id=self.experiment_id,
                community_level=self.community_level,
                as_dict=True,
            )

            self.s3_bucket_name = (
                config.config_knowledge_manager.config_retrieval_mode.storage.bucket_name
            )
            self.s3_region_name = (
                config.config_knowledge_manager.config_retrieval_mode.storage.region_name
            )
            self.s3_folder_path = f"s3://{self.s3_bucket_name}/output/{self.checkpoint_s3_folder_name}/artifacts"

            s3_helper = S3Helper(region_name=self.s3_region_name)
            self.all_maps = s3_helper.read_json(
                bucket_name=self.s3_bucket_name,
                s3_path=f"output/{self.checkpoint_s3_folder_name}/artifacts/{ALL_MAPS}.json",
            )

            if search_mode == SearchMode.global_mode:
                st = time.time()
                logger.info(f"[GraphRAGQueryManager] Start init Global Search")
                await self._init_global_search(
                    config=config,
                    community_reports=list(community_reports.values()),
                    entities=list(entities.values()),
                )
                logger.info(
                    f"[GraphRAGQueryManager] Finish init Global Search: {time.time() - st}"
                )
            elif search_mode == SearchMode.local_mode:

                logger.info(
                    f"[GraphRAGQueryManager] Start init connection to `QdrantVectorStore`"
                )

                self.collection_name = f"entity_description_embeddings_{self.checkpoint_s3_folder_name}_{self.embedding_model}_community_level_{self.community_level}"
                self.description_embedding_store = QdrantVectorStore(
                    collection_name=self.collection_name,
                )
                self.description_embedding_store.connect()
                logger.info(
                    f"[GraphRAGQueryManager] Finish init connection to `QdrantVectorStore`"
                )
                self.text_embedder = OpenAIEmbedding(
                    model=self.embedding_model,
                    deployment_name=self.embedding_model,
                    max_retries=20,
                )

                relationships = await get_all_relationships(
                    community_level=self.community_level,
                    experiment_id=self.experiment_id,
                    as_dict=True,
                )
                covariates = await get_all_covariates_by_type(
                    community_level=self.community_level,
                    experiment_id=self.experiment_id,
                    covariates_type=["claim"],
                )
                st = time.time()
                logger.info(f"[GraphRAGQueryManager] Start init Local Search")
                await self._init_local_search(
                    config=config,
                    community_reports=community_reports,
                    entities=entities,
                    relationships=relationships,
                    covariates=covariates,
                )
                logger.info(
                    f"[GraphRAGQueryManager] Finish init Local Search: {time.time() - st}"
                )
            logger.info(
                f"FINISH: update_graphrag!!!: time: {time.time() - st_time_main}"
            )
        logger.info(f"FINISH INIT `setup_graphrag`: {time.time() - st_time_main_super}")

    async def _init_local_search(
        self,
        config: Config,
        entities: dict[str, Entity],
        community_reports: dict[str, CommunityReport] | None = None,
        relationships: dict[str, Relationship] | None = None,
        covariates: dict[str, list[Covariate]] | None = None,
    ):
        # TODO: add these params to `LocalSearch` in `Config`
        local_context_params = {
            "community_level": self.community_level,
            "experiment_id": self.checkpoint_s3_folder_name,
            "text_unit_prop": config.config_knowledge_manager.config_retrieval_mode.local_search.text_unit_prop,
            "community_prop": config.config_knowledge_manager.config_retrieval_mode.local_search.community_prop,
            "conversation_history_max_turns": config.config_knowledge_manager.config_retrieval_mode.local_search.conversation_history_max_turns,
            "conversation_history_user_turns_only": config.config_knowledge_manager.config_retrieval_mode.local_search.conversation_history_max_turns,
            "top_k_mapped_entities": config.config_knowledge_manager.config_retrieval_mode.local_search.top_k_entities,
            "top_k_relationships": config.config_knowledge_manager.config_retrieval_mode.local_search.top_k_relationships,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,
            # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
            "max_tokens": 90_000,
            # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        }

        llm_params = {
            "max_tokens": 1_500,
            # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
            "temperature": 0.0,
        }

        self.local_context_builder = LocalSearchMixedContext(
            community_reports=community_reports,
            # text_units=self.text_units,
            entities=entities,
            relationships=relationships,
            covariates=covariates,
            # community_level=self.community_level,
            entity_text_embeddings=self.description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
            text_embedder=self.text_embedder,
            token_encoder=self.token_encoder,
        )
        self.local_search_engine = LocalSearch(
            llm=self.llm,
            context_builder=self.local_context_builder,
            token_encoder=self.token_encoder,
            llm_params=llm_params,
            context_builder_params=local_context_params,
            response_type="multiple paragraphs",
            # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        )

    async def _init_global_search(
        self,
        config: Config,
        community_reports: list[CommunityReport],
        entities: list[Entity] | None = None,
    ):

        # GLOBAL SEARCH
        max_data_tokens = (
            config.config_knowledge_manager.config_retrieval_mode.global_search.max_data_tokens
        )
        map_llm_params = json.loads(
            str(
                config.config_knowledge_manager.config_retrieval_mode.global_search.map_llm_params
            )
        )
        reduce_llm_params = json.loads(
            str(
                config.config_knowledge_manager.config_retrieval_mode.global_search.reduce_llm_params
            )
        )
        context_builder_params = json.loads(
            str(
                config.config_knowledge_manager.config_retrieval_mode.global_search.context_builder_params
            )
        )
        allow_general_knowledge = (
            config.config_knowledge_manager.config_retrieval_mode.global_search.allow_general_knowledge
        )
        json_mode = (
            config.config_knowledge_manager.config_retrieval_mode.global_search.json_mode
        )
        concurrent_coroutines = (
            config.config_knowledge_manager.config_retrieval_mode.global_search.concurrency
        )
        response_type = (
            config.config_knowledge_manager.config_retrieval_mode.global_search.response_type
        )

        # Prompts
        map_system_prompt = (
            config.config_knowledge_manager.config_retrieval_mode.global_search.map_system_prompt
        )
        reduce_system_prompt = (
            config.config_knowledge_manager.config_retrieval_mode.global_search.reduce_system_prompt
        )
        general_knowledge_inclusion_prompt = (
            config.config_knowledge_manager.config_retrieval_mode.global_search.general_knowledge_inclusion_prompt
        )

        context_builder_params["experiment_id"] = (
            config.config_knowledge_manager.config_retrieval_mode.checkpoint_s3_folder_name
        )
        context_builder_params["community_level"] = self.community_level

        self.global_context_builder = GlobalCommunityContext(
            token_encoder=self.token_encoder,
            community_reports=community_reports,
            entities=entities,
        )
        self.global_search_engine = GlobalSearch(
            map_system_prompt=map_system_prompt,
            reduce_system_prompt=reduce_system_prompt,
            general_knowledge_inclusion_prompt=general_knowledge_inclusion_prompt,
            llm=self.llm,
            context_builder=self.global_context_builder,
            token_encoder=self.token_encoder,
            max_data_tokens=max_data_tokens,
            map_llm_params=map_llm_params,
            reduce_llm_params=reduce_llm_params,
            allow_general_knowledge=allow_general_knowledge,
            # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
            json_mode=json_mode,  # set this to False if your LLM model does not support JSON mode.
            context_builder_params=context_builder_params,
            concurrent_coroutines=concurrent_coroutines,
            response_type=response_type,
            # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        )

    def search_global(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        run_reduce_response: bool = True,
        **kwargs: Any,
    ) -> GlobalSearchResult:
        """Perform a global search synchronously."""
        result = asyncio.run(
            self.global_search_engine.asearch(
                query=query,
                conversation_history=conversation_history,
                run_reduce_response=run_reduce_response,
                kwargs=kwargs,
            )
        )
        return result

    async def asearch_global(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        run_reduce_response: bool = True,
        **kwargs: Any,
    ) -> GlobalSearchResult:
        result = await self.global_search_engine.asearch(
            query=query,
            conversation_history=conversation_history,
            run_reduce_response=run_reduce_response,
            kwargs=kwargs,
        )
        logger.info(
            f"[Query=<{query}>] Content Data Reports Percentage: "
            f"{result.context_data['reports'].shape[0]} / {len(self.all_maps['comms_to_text_units'])} = "
            f"{round(result.context_data['reports'].shape[0] / len(self.all_maps['comms_to_text_units']), ndigits=3)}"
        )
        resources = await get_resources_from_column_context_data_result(
            result,
            all_maps=self.all_maps,
            based_on_column_=ContextDataColumn.REPORTS,
        )
        result.resources = (
            {ContextDataColumn.REPORTS.value: resources}
            if resources is not None
            else {ContextDataColumn.REPORTS.value: dict()}
        )
        if run_reduce_response:
            if resources is not None:
                result.response = await add_resources_to_response(
                    response_text=result.response, resources=resources
                )
            else:
                logger.warning(
                    f'!!"[resources is None] reports" NOT in result.context_data!!'
                )

        return result

    def search_local(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        run_reduce_response: bool = True,
        **kwargs: Any,
    ) -> SearchResult:
        """Perform a global search synchronously."""
        result = asyncio.run(
            self.local_search_engine.asearch(
                query=query,
                conversation_history=conversation_history,
                run_reduce_response=run_reduce_response,
                kwargs=kwargs,
            )
        )
        return result

    async def asearch_local(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        run_reduce_response: bool = True,
        **kwargs: Any,
    ) -> SearchResult:
        result = await self.local_search_engine.asearch(
            query=query,
            conversation_history=conversation_history,
            run_reduce_response=run_reduce_response,
            kwargs=kwargs,
        )
        resources_all = {
            ContextDataColumn.REPORTS.value: dict(),
            ContextDataColumn.ENTITIES.value: dict(),
            ContextDataColumn.RELATIONSHIPS.value: dict(),
        }
        for resource_column_data_name in resources_all.keys():
            resources = await get_resources_from_column_context_data_result(
                result,
                all_maps=self.all_maps,
                based_on_column_=ContextDataColumn(resource_column_data_name),
            )
            if resources is not None:
                resources_all[resource_column_data_name].update(resources)

        result.resources = resources_all
        if run_reduce_response:
            if not all((len(rakv) == 0 for rakv in resources_all.values())):
                result.response = await add_resources_to_response(
                    response_text=result.response, resources=resources_all
                )
            else:
                logger.warning(
                    f'!!"[resources is None] reports" NOT in result.context_data!!'
                )

        return result
