import json
from enum import Enum
from typing import List, Optional, Union, Dict

from pydantic import Field, model_validator, ValidationError

from libs.python.schemas.basic_models import LLMModel, BaseModelUpd, BaseEnum
from libs.python.schemas.graphrag.graph_rag_config import GraphRagConfig


# Content Scraper Schemas
class Country(str, BaseEnum):
    GB = "gb"


class Locale(str, BaseEnum):
    EN = "en"


class TypeParsing(str, BaseEnum):
    CHROMIUM = "chromium"
    REDDIT = "reddit"
    HAND_MAKE = "hand_make"


class VerifyType(str, BaseEnum):
    CHAIN = "chain"
    OFF = "off"


class GoogleSearchType(str, BaseEnum):
    SERPER = "serper"


class ConfigContentScraperMode(str, BaseEnum):
    OFF = "off"
    ON = "on"


class ConfigContentScraper(BaseModelUpd):
    mode: ConfigContentScraperMode = ConfigContentScraperMode.ON
    llm_model_verifier: LLMModel = LLMModel.GPT_4O_MINI
    top_k_url: int = 10
    country: Country = Country.GB
    locale: Locale = Locale.EN
    domain_filter: List[str] = []
    use_output_from_verif_as_content: bool = True
    google_search_type: GoogleSearchType = GoogleSearchType.SERPER
    type_parsing: TypeParsing = TypeParsing.CHROMIUM
    verify_type: VerifyType = VerifyType.CHAIN


# Knowledge Manager Schemas


class BehaviorGraphDB(str, BaseEnum):
    EXTRACT_GRAPH = "extract_graph"
    EXTRACT_SCHEME = "extract_scheme"
    EXTRACT_SCHEME_EXTRACT_GRAPH = "extract_scheme_extract_graph"


class EmbedderModel(str, BaseEnum):
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    TEXT_EMBEDDING_ADA_003 = "text-embedding-ada-003"


class ConfigRetrievalMode(BaseModelUpd):
    metadata: Dict = {}


class ConfigVectorDB(ConfigRetrievalMode):
    score_threshold: float = 0.88
    top_k_retrieval: int = 5
    chunk_size: int = 2000
    embedder_model: EmbedderModel = EmbedderModel.TEXT_EMBEDDING_ADA_002
    collection_name: str = "car-db"


# class ConfigGraphDB(ConfigRetrievalMode):
#     space_name: str = "car_kg_parsed_content_v0"
#     behavior_graph_db: BehaviorGraphDB = BehaviorGraphDB.EXTRACT_GRAPH
#     is_need_upd_scheme_during_extract_graph: bool = False
#     model_name_extract_scheme: LLMModel = LLMModel.GPT_4_0613
#     model_name_extract_graph: LLMModel = LLMModel.GPT_4_0613
#
#
# class ConfigVectorDBAndGraphDB(ConfigVectorDB, ConfigGraphDB):
#     pass


class KnowledgeManagerMode(str, BaseEnum):
    OFF = "off"
    VECTOR_DB = "vector_db"
    GRAPHRAG = "graphrag"
    # GRAPH_DB = "graph_db"
    # VECTOR_DB_AND_GRAPH_DB = "vector_db_and_graph_db"


class ConfigKnowledgeManager(BaseModelUpd):
    threshold_go_to_content_scraper: int = Field(
        2,
        description="Count of `Content(text: str, src: str)`, "
        "think about it as count of `chunk` from `Retrieval Engine`",
    )
    mode: KnowledgeManagerMode = KnowledgeManagerMode.GRAPHRAG
    config_retrieval_mode: Optional[
        Union[
            ConfigRetrievalMode, ConfigVectorDB, GraphRagConfig
        ]  # ConfigGraphDB, ConfigVectorDBAndGraphDB
    ] = GraphRagConfig()

    @model_validator(mode="before")
    def validate_config_retrieval_mode(cls, values):

        if (
            (values.get("mode") is None)
            and (values.get("config_retrieval_mode") is not None)
        ) or (
            (values.get("mode") is not None)
            and (values.get("config_retrieval_mode") is None)
        ):
            raise ValueError(
                f"mode and config_retrieval_mode have to be set both at the same time!"
            )

        if values.get("mode") and values.get("config_retrieval_mode"):

            mode = values.get("mode")
            config_values = values.get("config_retrieval_mode")
            if isinstance(mode, str):
                mode = KnowledgeManagerMode(mode)
            if not isinstance(config_values, dict):
                config_values = json.loads(config_values.json())

            if mode == KnowledgeManagerMode.VECTOR_DB:
                try:
                    values["config_retrieval_mode"] = ConfigVectorDB(**config_values)
                except ValidationError as e:
                    raise ValueError(f"Invalid parameters for {mode}: {e}")

            elif mode == KnowledgeManagerMode.GRAPHRAG:
                try:
                    values["config_retrieval_mode"] = GraphRagConfig(**config_values)
                except ValidationError as e:
                    raise ValueError(f"Invalid parameters for {mode}: {e}")

            # elif mode == KnowledgeManagerMode.GRAPH_DB:
            #     try:
            #         values["config_retrieval_mode"] = ConfigGraphDB(**config_values)
            #     except ValidationError as e:
            #         raise ValueError(f"Invalid parameters for {mode}: {e}")
            #
            # elif mode == KnowledgeManagerMode.VECTOR_DB_AND_GRAPH_DB:
            #     try:
            #         values["config_retrieval_mode"] = ConfigVectorDBAndGraphDB(
            #             **config_values
            #         )
            #     except ValidationError as e:
            #         raise ValueError(f"Invalid parameters for {mode}: {e}")

            elif mode == KnowledgeManagerMode.OFF:
                try:
                    values["config_retrieval_mode"] = ConfigRetrievalMode(
                        **config_values
                    )
                except ValidationError as e:
                    raise ValueError(f"Invalid parameters for {mode}: {e}")

            else:
                raise ValueError(f"Invalid knowledge manager mode: {mode}")

        return values


class ConfigNextGenUserSteps(BaseModelUpd):
    available: bool = True
    users_potential_count_of_variants_of_the_next_step: int = 4
    llm_model: LLMModel = LLMModel.GPT_4O_MINI
    llm_model_temperature: float = 0.0
    sys_prompt: Optional[str] = None


class ConfigSummarizerFinalAnswerSetup(BaseModelUpd):
    available: bool = True
    llm_model: LLMModel = LLMModel.GPT_4O_MINI
    llm_model_temperature: float = 0.0
    chars_threshold: int = 350
    sys_prompt: Optional[str] = None


# Agent Flow Schemas
class ConfigAgentFlow(BaseModelUpd):
    llm_model: LLMModel = LLMModel.GPT_4O_MINI
    llm_model_temperature: float = 0.0
    sys_prompt: Optional[str] = None
    next_gen_user_steps: ConfigNextGenUserSteps = ConfigNextGenUserSteps()
    summarizer_final_answer_setup: ConfigSummarizerFinalAnswerSetup = (
        ConfigSummarizerFinalAnswerSetup()
    )
    window_k_dialog_messages: int = Field(
        2400,
        description="Count of messages that goes to agent, includes any type of Message (AI, HUMAN)",
    )


class Config(BaseModelUpd):
    config_agent_flow: ConfigAgentFlow
    config_knowledge_manager: ConfigKnowledgeManager
    config_content_scraper: Optional[ConfigContentScraper]
