import json
from enum import Enum
from copy import deepcopy
from libs.python.schemas.configuration import (
    Config,
    ConfigAgentFlow,
    ConfigKnowledgeManager,
    ConfigContentScraper,
    KnowledgeManagerMode,
    Country,
    Locale,
    ConfigVectorDB,
    GoogleSearchType,
    TypeParsing,
    VerifyType,
    EmbedderModel,
    ConfigContentScraperMode,
    ConfigRetrievalMode,
)
from libs.python.schemas.basic_models import LLMModel, BaseEnum
from libs.python.schemas.graphrag import (
    GlobalSearchConfig,
    StorageConfig,
    LocalSearchConfig,
)
from libs.python.schemas.graphrag.graph_rag_config import GraphRagConfig, SearchMode


class ConfigPreSet(str, BaseEnum):
    BASE_LLM_GPT4O = "base_llm_gpt4o"
    BASE_LLM_SYS_PROMPT_GPT4O = "base_llm_sys_prompt_gpt4o"
    GRAPHRAG_TEST_GPT4O = "graphrag_test_gpt4o"

    VDB_INET_SCRAP_GPT4O = "vdb_inet_scrap_gpt4o"
    VDB_GPT4O = "vdb_gpt4o"
    VDB_GPT4O_FIXED_CONTENT_BASED_ON_DIALOGS = (
        "vdb_gpt4o_fixed_content_based_on_dialogs"
    )
    VDB_USE_VERIF_GPT4O = "vdb_use_verif_gpt4o"
    VDB_INET_SCRAP_USE_VERIF_GPT4O = "vdb_inet_scrap_use_verif_gpt4o"

    BASE_LLM_GPT4O_MINI = "base_llm_gpt4o_mini"
    BASE_LLM_SYS_PROMPT_GPT4O_MINI = "base_llm_sys_prompt_gpt4o_mini"
    GRAPHRAG_TEST_GPT4O_MINI = "graphrag_test_gpt4o_mini"

    VDB_INET_SCRAP_GPT4O_MINI = "vdb_inet_scrap_gpt4o_mini"
    VDB_GPT4O_MINI = "vdb_gpt4o_mini"
    VDB_GPT4O_MINI_FIXED_CONTENT_BASED_ON_DIALOGS = (
        "vdb_gpt4o_mini_fixed_content_based_on_dialogs"
    )
    VDB_USE_VERIF_GPT4O_MINI = "vdb_use_verif_gpt4o_mini"
    VDB_INET_SCRAP_USE_VERIF_GPT4O_MINI = "vdb_inet_scrap_use_verif_gpt4o_mini"


preset_configs = {
    ConfigPreSet.GRAPHRAG_TEST_GPT4O: Config(
        config_agent_flow=ConfigAgentFlow(
            llm_model=LLMModel.GPT_4O,
            window_k_dialog_messages=2400,
            llm_model_temperature=0.0,
        ),
        config_knowledge_manager=ConfigKnowledgeManager(
            threshold_go_to_content_scraper=2,
            mode=KnowledgeManagerMode.GRAPHRAG,
            config_retrieval_mode=GraphRagConfig(
                search_mode=SearchMode.local_mode,
                # search_mode=SearchMode.global_mode,
                storage=StorageConfig(bucket_name="dev-1"),
                # checkpoint_s3_folder_name="25da564b-4e4c-4880-9c29-6de17cb80d38_20240805-205611",
                # checkpoint_s3_folder_name="be453ff8-eba0-4269-a7a9-77822c258472_20240809-031156",
                # checkpoint_s3_folder_name="7d849eb6-cd2b-4e20-aefd-af08ea0906b7_20240819-141608",
                # checkpoint_s3_folder_name="76b7124e-1573-40ef-bdb7-53d64c2b7c57_20240828-235943",  # added qdrant and postgres tables
                # checkpoint_s3_folder_name="eba4cc04-8065-4263-ba0c-2a03f6ef8236_20240829-175300",  # fixed bug
                checkpoint_s3_folder_name="4089a296-f4ce-4b21-830e-ca56b8419732_20240829-222856",  # on prod
                global_search=GlobalSearchConfig(),
                local_search=LocalSearchConfig(),
            ),
        ),
        config_content_scraper=ConfigContentScraper(
            mode=ConfigContentScraperMode.OFF,
            llm_model_verifier=LLMModel.GPT_4O_MINI,
            top_k_url=7,
            country=Country.GB,
            locale=Locale.EN,
            domain_filter=[],
            use_output_from_verif_as_content=True,
            google_search_type=GoogleSearchType.SERPER,
            type_parsing=TypeParsing.CHROMIUM,
            verify_type=VerifyType.CHAIN,
        ),
    ),
    ConfigPreSet.BASE_LLM_SYS_PROMPT_GPT4O: Config(
        config_agent_flow=ConfigAgentFlow(
            llm_model=LLMModel.GPT_4O,
            window_k_dialog_messages=2400,
            llm_model_temperature=0.0,
        ),
        config_knowledge_manager=ConfigKnowledgeManager(
            threshold_go_to_content_scraper=2,
            mode=KnowledgeManagerMode.OFF,
            config_retrieval_mode=ConfigRetrievalMode(metadata={}),
        ),
        config_content_scraper=ConfigContentScraper(
            mode=ConfigContentScraperMode.OFF,
            llm_model_verifier=LLMModel.GPT_4O_MINI,
            top_k_url=7,
            country=Country.GB,
            locale=Locale.EN,
            domain_filter=[],
            use_output_from_verif_as_content=True,
            google_search_type=GoogleSearchType.SERPER,
            type_parsing=TypeParsing.CHROMIUM,
            verify_type=VerifyType.CHAIN,
        ),
    ),
    ConfigPreSet.BASE_LLM_GPT4O: Config(
        config_agent_flow=ConfigAgentFlow(
            llm_model=LLMModel.GPT_4O,
            window_k_dialog_messages=2400,
            llm_model_temperature=0.0,
            sys_prompt="",
        ),
        config_knowledge_manager=ConfigKnowledgeManager(
            threshold_go_to_content_scraper=2,
            mode=KnowledgeManagerMode.OFF,
            config_retrieval_mode=ConfigRetrievalMode(metadata={}),
        ),
        config_content_scraper=ConfigContentScraper(
            mode=ConfigContentScraperMode.OFF,
            llm_model_verifier=LLMModel.GPT_4O_MINI,
            top_k_url=7,
            country=Country.GB,
            locale=Locale.EN,
            domain_filter=[],
            use_output_from_verif_as_content=True,
            google_search_type=GoogleSearchType.SERPER,
            type_parsing=TypeParsing.CHROMIUM,
            verify_type=VerifyType.CHAIN,
        ),
    ),
    ConfigPreSet.VDB_GPT4O: Config(
        config_agent_flow=ConfigAgentFlow(
            llm_model=LLMModel.GPT_4O,
            window_k_dialog_messages=2400,
            llm_model_temperature=0.0,
        ),
        config_knowledge_manager=ConfigKnowledgeManager(
            threshold_go_to_content_scraper=2,
            mode=KnowledgeManagerMode.VECTOR_DB,
            config_retrieval_mode=ConfigVectorDB(
                score_threshold=0.88,
                top_k_retrieval=5,
                metadata={},
                embedder_model=EmbedderModel.TEXT_EMBEDDING_ADA_002,
                chunk_size=2000,
                collection_name="car-db",
            ),
        ),
        config_content_scraper=ConfigContentScraper(
            mode=ConfigContentScraperMode.OFF,
            llm_model_verifier=LLMModel.GPT_4O_MINI,
            top_k_url=7,
            country=Country.GB,
            locale=Locale.EN,
            domain_filter=[],
            use_output_from_verif_as_content=False,
            google_search_type=GoogleSearchType.SERPER,
            type_parsing=TypeParsing.CHROMIUM,
            verify_type=VerifyType.CHAIN,
        ),
    ),
    ConfigPreSet.VDB_GPT4O_FIXED_CONTENT_BASED_ON_DIALOGS: Config(
        config_agent_flow=ConfigAgentFlow(
            llm_model=LLMModel.GPT_4O,
            window_k_dialog_messages=2400,
            llm_model_temperature=0.0,
        ),
        config_knowledge_manager=ConfigKnowledgeManager(
            threshold_go_to_content_scraper=2,
            mode=KnowledgeManagerMode.VECTOR_DB,
            config_retrieval_mode=ConfigVectorDB(
                score_threshold=0.5,
                top_k_retrieval=20,
                metadata={},
                embedder_model=EmbedderModel.TEXT_EMBEDDING_ADA_002,
                chunk_size=50,
                collection_name="car_db_reviews_content_merged_unified_sampled_based_on_dialogs",
            ),
        ),
        config_content_scraper=ConfigContentScraper(
            mode=ConfigContentScraperMode.OFF,
            llm_model_verifier=LLMModel.GPT_4O_MINI,
            top_k_url=7,
            country=Country.GB,
            locale=Locale.EN,
            domain_filter=[],
            use_output_from_verif_as_content=False,
            google_search_type=GoogleSearchType.SERPER,
            type_parsing=TypeParsing.CHROMIUM,
            verify_type=VerifyType.CHAIN,
        ),
    ),
    ConfigPreSet.VDB_USE_VERIF_GPT4O: Config(
        config_agent_flow=ConfigAgentFlow(
            llm_model=LLMModel.GPT_4O,
            window_k_dialog_messages=2400,
            llm_model_temperature=0.0,
        ),
        config_knowledge_manager=ConfigKnowledgeManager(
            threshold_go_to_content_scraper=2,
            mode=KnowledgeManagerMode.VECTOR_DB,
            config_retrieval_mode=ConfigVectorDB(
                score_threshold=0.88,
                top_k_retrieval=5,
                metadata={},
                embedder_model=EmbedderModel.TEXT_EMBEDDING_ADA_002,
                chunk_size=2000,
                collection_name="car-db",
            ),
        ),
        config_content_scraper=ConfigContentScraper(
            mode=ConfigContentScraperMode.OFF,
            llm_model_verifier=LLMModel.GPT_4O_MINI,
            top_k_url=7,
            country=Country.GB,
            locale=Locale.EN,
            domain_filter=[],
            use_output_from_verif_as_content=True,
            google_search_type=GoogleSearchType.SERPER,
            type_parsing=TypeParsing.CHROMIUM,
            verify_type=VerifyType.CHAIN,
        ),
    ),
    ConfigPreSet.VDB_INET_SCRAP_GPT4O: Config(
        config_agent_flow=ConfigAgentFlow(
            llm_model=LLMModel.GPT_4O,
            window_k_dialog_messages=2400,
            llm_model_temperature=0.0,
        ),
        config_knowledge_manager=ConfigKnowledgeManager(
            threshold_go_to_content_scraper=2,
            mode=KnowledgeManagerMode.VECTOR_DB,
            config_retrieval_mode=ConfigVectorDB(
                score_threshold=0.88,
                top_k_retrieval=5,
                metadata={},
                embedder_model=EmbedderModel.TEXT_EMBEDDING_ADA_002,
                chunk_size=2000,
                collection_name="car-db",
            ),
        ),
        config_content_scraper=ConfigContentScraper(
            mode=ConfigContentScraperMode.ON,
            llm_model_verifier=LLMModel.GPT_4O_MINI,
            top_k_url=7,
            country=Country.GB,
            locale=Locale.EN,
            domain_filter=[],
            use_output_from_verif_as_content=False,
            google_search_type=GoogleSearchType.SERPER,
            type_parsing=TypeParsing.CHROMIUM,
            verify_type=VerifyType.CHAIN,
        ),
    ),
    ConfigPreSet.VDB_INET_SCRAP_USE_VERIF_GPT4O: Config(
        config_agent_flow=ConfigAgentFlow(
            llm_model=LLMModel.GPT_4O,
            window_k_dialog_messages=2400,
            llm_model_temperature=0.0,
        ),
        config_knowledge_manager=ConfigKnowledgeManager(
            threshold_go_to_content_scraper=2,
            mode=KnowledgeManagerMode.VECTOR_DB,
            config_retrieval_mode=ConfigVectorDB(
                score_threshold=0.88,
                top_k_retrieval=5,
                metadata={},
                embedder_model=EmbedderModel.TEXT_EMBEDDING_ADA_002,
                chunk_size=2000,
                collection_name="car-db",
            ),
        ),
        config_content_scraper=ConfigContentScraper(
            mode=ConfigContentScraperMode.ON,
            llm_model_verifier=LLMModel.GPT_4O_MINI,
            top_k_url=7,
            country=Country.GB,
            locale=Locale.EN,
            domain_filter=[],
            use_output_from_verif_as_content=True,
            google_search_type=GoogleSearchType.SERPER,
            type_parsing=TypeParsing.CHROMIUM,
            verify_type=VerifyType.CHAIN,
        ),
    ),
}

for name_config_value_src, name_config in zip(
    (
        ConfigPreSet.BASE_LLM_GPT4O,
        ConfigPreSet.GRAPHRAG_TEST_GPT4O,
        ConfigPreSet.BASE_LLM_SYS_PROMPT_GPT4O,
        ConfigPreSet.VDB_INET_SCRAP_USE_VERIF_GPT4O,
        ConfigPreSet.VDB_GPT4O,
        ConfigPreSet.VDB_GPT4O_FIXED_CONTENT_BASED_ON_DIALOGS,
        ConfigPreSet.VDB_INET_SCRAP_GPT4O,
        ConfigPreSet.VDB_USE_VERIF_GPT4O,
    ),
    (
        ConfigPreSet.BASE_LLM_GPT4O_MINI,
        ConfigPreSet.GRAPHRAG_TEST_GPT4O_MINI,
        ConfigPreSet.BASE_LLM_SYS_PROMPT_GPT4O_MINI,
        ConfigPreSet.VDB_INET_SCRAP_USE_VERIF_GPT4O_MINI,
        ConfigPreSet.VDB_GPT4O_MINI,
        ConfigPreSet.VDB_GPT4O_MINI_FIXED_CONTENT_BASED_ON_DIALOGS,
        ConfigPreSet.VDB_INET_SCRAP_GPT4O_MINI,
        ConfigPreSet.VDB_USE_VERIF_GPT4O_MINI,
    ),
):
    preset_configs[name_config] = deepcopy(preset_configs.get(name_config_value_src))
    preset_configs[name_config].config_agent_flow.llm_model = LLMModel.GPT_4O_MINI


def get_config(preset: ConfigPreSet) -> Config:
    return preset_configs.get(preset)
