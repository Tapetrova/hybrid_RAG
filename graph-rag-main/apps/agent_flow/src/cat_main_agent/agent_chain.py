import json
import os
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent,
    create_openai_functions_agent,
)
from langchain.globals import set_llm_cache
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
from redis import Redis

from apps.agent_flow.src.cat_main_agent.agent_tools import (
    aget_information_from_internet,
    get_information_from_internet,
    aget_information_from_internet_graphrag,
    get_information_from_internet_graphrag,
)
from apps.agent_flow.src.cat_main_agent.openai_helper_strict_schema_output import (
    StructureChainOpenai,
)
from apps.agent_flow.src.cat_main_agent.prompts import (
    SYS_STR_PROMPT_CAR_ASSISTANT_AGENT_TOOL,
    SYS_STR_PROMPT_CAR_ASSISTANT_AGENT,
    SYS_STR_PROMPT_GET_INFO_FROM_INET_TOOL_DESCRIPTION,
    GET_INFORMATION_FROM_INTERNET_NATURAL_QUERY_FIELD_DESCRIPTION,
)
from libs.python.schemas.basic_models import LLMModel
from libs.python.schemas.configuration import KnowledgeManagerMode, Config
from libs.python.schemas.graphrag.graph_rag_config import SearchMode
from libs.python.schemas.metrics import (
    LLMProcessMetrics,
    InputLLMProcessMetrics,
    OutputLLMProcessMetrics,
    TotalLLMProcessMetrics,
)
from libs.python.utils.cacher import aget_cached_data, hash_string
from libs.python.utils.cacher_langchain import RedisCache
from libs.python.utils.callbacks import MAP_USAGE_CALLBACK
from libs.python.utils.graph_rag_utils.format_output_resources import (
    add_resources_to_response,
    ContextDataColumn,
)
from libs.python.utils.logger import logger

# Initialize Redis cache
set_llm_cache(
    RedisCache(
        redis_=Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=os.getenv("REDIS_PORT", 6379),
            db=os.getenv("REDIS_DB_CONV", 7),
            password=os.getenv("REDIS_PASSWORD_LLM", "redispassword"),
        )
    )
)


class GetInformationFromInternetToolSchema(BaseModel):
    """Schema for the internet information retrieval tool."""

    natural_query: str = Field(
        description=GET_INFORMATION_FROM_INTERNET_NATURAL_QUERY_FIELD_DESCRIPTION
    )


class PromptManager:
    """Manages prompt templates for different knowledge manager modes."""

    def __init__(
        self,
        knowledge_manager_mode: KnowledgeManagerMode,
        sys_prompt: Optional[str] = None,
    ):
        self.knowledge_manager_mode = knowledge_manager_mode
        self.sys_prompt = self._get_system_prompt(sys_prompt, knowledge_manager_mode)
        self.prompt = self._build_prompt()
        self.prompt_only_chat_history_and_function_output_holder = (
            self._build_function_output_prompt()
        )

    def _get_system_prompt(
        self, sys_prompt: Optional[str], mode: KnowledgeManagerMode
    ) -> str:
        """Get the appropriate system prompt based on mode."""
        if sys_prompt is not None:
            return sys_prompt

        if mode == KnowledgeManagerMode.OFF:
            return SYS_STR_PROMPT_CAR_ASSISTANT_AGENT
        return SYS_STR_PROMPT_CAR_ASSISTANT_AGENT_TOOL

    def _build_prompt(self) -> ChatPromptTemplate:
        """Build the main prompt template."""
        messages = self._create_base_messages()

        if self.knowledge_manager_mode != KnowledgeManagerMode.OFF:
            messages.append(MessagesPlaceholder("agent_scratchpad"))
            input_vars = ["chat_history", "input", "agent_scratchpad"]
        else:
            input_vars = ["chat_history", "input"]

        return ChatPromptTemplate(messages=messages, input_variables=input_vars)

    def _build_function_output_prompt(self) -> ChatPromptTemplate:
        """Build the function output prompt template."""
        messages = self._create_base_messages()
        messages.append(
            SystemMessagePromptTemplate.from_template("TOOL OUTPUT:\n {tool_output}")
        )

        return ChatPromptTemplate(
            messages=messages, input_variables=["chat_history", "input", "tool_output"]
        )

    def _create_base_messages(self) -> List:
        """Create base message templates."""
        messages = []
        if self.sys_prompt:
            messages.append(SystemMessagePromptTemplate.from_template(self.sys_prompt))
        messages.extend(
            [
                MessagesPlaceholder("chat_history", optional=True),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        return messages

    async def format_prompt(
        self,
        input_message: str,
        chat_history: List[BaseMessage],
    ) -> str:
        """Format the prompt with input and chat history."""
        format_args = {"input": input_message, "chat_history": chat_history}

        if self.knowledge_manager_mode != KnowledgeManagerMode.OFF:
            format_args["agent_scratchpad"] = []

        return await self.prompt.aformat(**format_args)

    def get_prompt(self) -> ChatPromptTemplate:
        """Get the main prompt template."""
        return self.prompt

    def get_prompt_only_chat_history_and_function_output_holder(
        self,
    ) -> ChatPromptTemplate:
        """Get the function output prompt template."""
        return self.prompt_only_chat_history_and_function_output_holder


class ToolManager:
    """Manages tools for the agent."""

    def __init__(
        self,
        config: Config,
        user_id: str,
        pubsub_channel_name: str,
        prompt_manager: PromptManager,
        chain_llm_with_structure: StructureChainOpenai,
        chat_history: List[BaseMessage],
        input_message: str,
    ):
        self.config = config
        self.user_id = user_id
        self.pubsub_channel_name = pubsub_channel_name
        self.prompt_manager = prompt_manager
        self.chain_llm_with_structure = chain_llm_with_structure
        self.chat_history = chat_history
        self.input_message = input_message
        self._run_reduce_response = self._should_run_reduce_response()

    def _should_run_reduce_response(self) -> bool:
        """Determine if response reduction should be run."""
        if self.config.config_knowledge_manager.mode != KnowledgeManagerMode.GRAPHRAG:
            return False

        retrieval_mode = self.config.config_knowledge_manager.config_retrieval_mode
        search_mode = retrieval_mode.search_mode

        if search_mode == SearchMode.global_mode:
            return retrieval_mode.global_search.run_reduce_response
        elif search_mode == SearchMode.local_mode:
            return retrieval_mode.local_search.run_reduce_response
        return False

    async def _aget_information_from_internet(self, natural_query: str) -> str:
        """Async function to get information from internet."""
        if self.config.config_knowledge_manager.mode == KnowledgeManagerMode.VECTOR_DB:
            return await aget_information_from_internet(
                natural_query=natural_query,
                user_id=self.user_id,
                config=self.config,
                pubsub_channel_name=self.pubsub_channel_name,
            )
        else:
            context_data = await aget_information_from_internet_graphrag(
                natural_query=natural_query,
                user_id=self.user_id,
                config=self.config,
                pubsub_channel_name=self.pubsub_channel_name,
                run_reduce_response=self._run_reduce_response,
            )

            if self._run_reduce_response:
                return context_data

            output = await self.chain_llm_with_structure.ainvoke(
                system_prompt=self.prompt_manager.sys_prompt,
                input_message=self.input_message,
                chat_history=self.chat_history,
                tool_output=context_data,
                temperature=self.config.config_agent_flow.llm_model_temperature,
            )
            output.context_data_from_tool = context_data
            return output.json()

    def _get_information_from_internet(self, natural_query: str) -> str:
        """Sync function to get information from internet."""
        if self.config.config_knowledge_manager.mode == KnowledgeManagerMode.VECTOR_DB:
            return get_information_from_internet(
                natural_query=natural_query,
                user_id=self.user_id,
                config=self.config,
                pubsub_channel_name=self.pubsub_channel_name,
            )
        else:
            context_data = get_information_from_internet_graphrag(
                natural_query=natural_query,
                user_id=self.user_id,
                config=self.config,
                pubsub_channel_name=self.pubsub_channel_name,
                run_reduce_response=self._run_reduce_response,
            )

            if self._run_reduce_response:
                return context_data

            output = self.chain_llm_with_structure.invoke(
                system_prompt=self.prompt_manager.sys_prompt,
                input_message=self.input_message,
                chat_history=self.chat_history,
                tool_output=context_data,
                temperature=self.config.config_agent_flow.llm_model_temperature,
            )
            output.context_data_from_tool = context_data
            return output.json()

    def create_tools(self) -> List[StructuredTool]:
        """Create the tools for the agent."""
        return_direct = (
            self.config.config_knowledge_manager.mode != KnowledgeManagerMode.VECTOR_DB
        )

        return [
            StructuredTool.from_function(
                name="get_information_from_internet",
                func=self._get_information_from_internet,
                coroutine=self._aget_information_from_internet,
                description=SYS_STR_PROMPT_GET_INFO_FROM_INET_TOOL_DESCRIPTION,
                args_schema=GetInformationFromInternetToolSchema,
                return_direct=return_direct,
            ),
        ]


class MetricsHandler:
    """Handles metrics collection and processing."""

    @staticmethod
    def create_initial_metrics(
        count_input_tokens_exceeded: int = 0,
    ) -> LLMProcessMetrics:
        """Create initial metrics object."""
        return LLMProcessMetrics(
            name="main_agent",
            input_metrics=InputLLMProcessMetrics(input_tokens=0, input_cost=0),
            output_metrics=OutputLLMProcessMetrics(output_tokens=0, output_cost=0),
            total_metrics=TotalLLMProcessMetrics(total_tokens=0, total_cost=0),
            count_input_tokens_exceeded=count_input_tokens_exceeded,
            time_exe=0.0,
        )

    @staticmethod
    def update_metrics_from_callback(
        metrics: LLMProcessMetrics,
        callback: Any,
        time_exe: float,
        count_input_tokens_exceeded: int = 0,
    ) -> LLMProcessMetrics:
        """Update metrics from callback."""
        return LLMProcessMetrics(
            name="main_agent",
            input_metrics=InputLLMProcessMetrics(
                input_tokens=callback.input_tokens,
                input_cost=callback.input_cost,
            ),
            output_metrics=OutputLLMProcessMetrics(
                output_tokens=callback.output_tokens,
                output_cost=callback.output_cost,
            ),
            total_metrics=TotalLLMProcessMetrics(
                total_tokens=callback.total_tokens,
                total_cost=callback.total_cost,
            ),
            count_input_tokens_exceeded=count_input_tokens_exceeded,
            time_exe=time_exe,
        )


class GraphRAGResponseProcessor:
    """Processes GraphRAG responses."""

    @staticmethod
    async def process_graphrag_response(
        output: Dict[str, Any],
        output_structure_string: str,
        pubsub_channel_name: str,
        run_reduce_response: bool,
    ) -> Tuple[str, Dict[str, Any], Optional[str], Optional[LLMProcessMetrics]]:
        """Process GraphRAG response and extract relevant data."""
        output_structure_dict = json.loads(output_structure_string)
        context_data_from_tool = output_structure_dict.get("context_data_from_tool")

        hash_output_tool = hash_string(input_string=context_data_from_tool)

        references = output_structure_dict.get("output", {}).get("references")
        output["output"] = output_structure_dict.get("output", {}).get("agent_answer")

        # Update metrics
        update_usage_tool_info = output_structure_dict.get("process_metrics")
        if update_usage_tool_info:
            update_usage_metrics = LLMProcessMetrics(**update_usage_tool_info)
        else:
            update_usage_metrics = None

        return (
            hash_output_tool,
            references,
            context_data_from_tool,
            update_usage_metrics,
        )

    @staticmethod
    async def add_resources_to_output(
        output: Dict[str, Any],
        resources: Dict[str, Any],
        references: Dict[str, Any],
        run_reduce_response: bool,
    ) -> str:
        """Add resources to the output response."""
        if not resources or all(len(v) == 0 for v in resources.values()):
            logger.warning("Resources not found or empty")
            return output["output"]

        extracted_numbers_map = None
        if not run_reduce_response:
            extracted_numbers_map = {
                ContextDataColumn.REPORTS: references.get("ids_reports"),
                ContextDataColumn.RELATIONSHIPS: references.get("ids_relationships"),
                ContextDataColumn.ENTITIES: references.get("ids_entities"),
            }

        return await add_resources_to_response(
            response_text=output["output"],
            resources=resources,
            extracted_numbers_map=extracted_numbers_map,
        )

    @staticmethod
    def process_context_data(
        graph_res_data: Dict[str, Any],
        references: Dict[str, Any],
        resources: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process context data from GraphRAG response."""
        output_tool = {}

        if "context_data" not in graph_res_data:
            return output_tool

        for type_resource in ("reports", "entities", "relationships"):
            if type_resource not in graph_res_data["context_data"]:
                continue

            type_resource_values = pd.DataFrame(
                graph_res_data["context_data"][type_resource]
            )

            for idx, row in type_resource_values.iterrows():
                if int(row.id) not in references.get(f"ids_{type_resource}", []):
                    continue

                content = row.content if type_resource == "reports" else row.description

                for source in row.sources:
                    if source in resources.get(type_resource, {}):
                        output_tool[f"source_data_{row.id}_{type_resource}"] = {
                            "Content": content,
                            "Source": source,
                        }

        return output_tool


class CarAgentMain:
    """Main agent class for car assistant."""

    def __init__(
        self,
        llm_model: LLMModel,
        tool_get_info_config: Config,
        user_id: str,
        pubsub_channel_name: str,
        started_datetime: datetime,
        prompt_manager: PromptManager,
        chat_history: List[BaseMessage],
        input_message: str,
    ):
        self.llm_model = llm_model
        self.tool_get_info_config = tool_get_info_config
        self.user_id = user_id
        self.pubsub_channel_name = pubsub_channel_name
        self.started_datetime = started_datetime
        self.prompt_manager = prompt_manager
        self.chat_history = chat_history
        self.input_message = input_message

        # Initialize components
        self._initialize_llm()
        self._initialize_chain()
        self._initialize_agent()
        self._log_configuration()

    def _initialize_llm(self):
        """Initialize the LLM."""
        self.llm = ChatOpenAI(
            model_name=self.llm_model.value,
            temperature=self.tool_get_info_config.config_agent_flow.llm_model_temperature,
            request_timeout=120,
            verbose=True,
        )

    def _initialize_chain(self):
        """Initialize the structure chain."""
        self.chain_llm_with_structure = StructureChainOpenai(llm_model=self.llm_model)

    def _initialize_agent(self):
        """Initialize the agent and tools."""
        self.prompt = self.prompt_manager.get_prompt()

        if (
            self.tool_get_info_config.config_knowledge_manager.mode
            == KnowledgeManagerMode.OFF
        ):
            self.tools = None
            self.agent_executor = self.prompt | self.llm
        else:
            # Create tool manager and tools
            tool_manager = ToolManager(
                config=self.tool_get_info_config,
                user_id=self.user_id,
                pubsub_channel_name=self.pubsub_channel_name,
                prompt_manager=self.prompt_manager,
                chain_llm_with_structure=self.chain_llm_with_structure,
                chat_history=self.chat_history,
                input_message=self.input_message,
            )

            self.tools = tool_manager.create_tools()
            self._run_reduce_response = tool_manager._run_reduce_response

            # Create agent based on mode
            if (
                self.tool_get_info_config.config_knowledge_manager.mode
                == KnowledgeManagerMode.VECTOR_DB
            ):
                self.agent = create_openai_tools_agent(
                    self.llm, self.tools, self.prompt
                )
            else:
                self.agent = create_openai_functions_agent(
                    self.llm, self.tools, self.prompt
                )

            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
            )

    def _log_configuration(self):
        """Log the agent configuration."""
        logger.info(
            f"KnowledgeManagerMode: {self.tool_get_info_config.config_knowledge_manager.mode}"
        )
        logger.info(
            f"[CarAgentMain] LLMModel name: {self.tool_get_info_config.config_agent_flow.llm_model.value}"
        )
        logger.info(
            f"[CarAgentMain] LLMModel temperature: {self.tool_get_info_config.config_agent_flow.llm_model_temperature}"
        )
        logger.info(
            f"[CarAgentMain] config_knowledge_manager.mode: {self.tool_get_info_config.config_knowledge_manager.mode.value}"
        )
        logger.info(
            f"[CarAgentMain] config_retrieval_mode: {self.tool_get_info_config.config_knowledge_manager.config_retrieval_mode.json()}"
        )

    async def inference(
        self, input: str, chat_history: List[BaseMessage], *args, **kwargs
    ) -> Tuple[str, Optional[LLMProcessMetrics], Dict, str]:
        """Perform inference with the agent."""
        input = input.strip()

        # Initialize metrics
        count_input_tokens_exceeded = kwargs.get("count_agent_input_tokens_exceeded", 0)
        usage_model_metrics_info = MetricsHandler.create_initial_metrics(
            count_input_tokens_exceeded
        )

        # Process with callback
        usage_callback = MAP_USAGE_CALLBACK[self.llm_model]

        with usage_callback(
            model_name=self.llm_model.value,
            pubsub_channel=self.pubsub_channel_name,
            started_datetime=self.started_datetime,
        ) as cb:
            start_time = time.time()
            output = await self.agent_executor.ainvoke(
                {"input": input, "chat_history": chat_history}
            )

            # Update metrics
            usage_model_metrics_info = MetricsHandler.update_metrics_from_callback(
                usage_model_metrics_info,
                cb,
                time.time() - start_time,
                count_input_tokens_exceeded,
            )

            # Process output
            final_output, output_tool, context_data_from_tool = (
                await self._process_output(output, usage_model_metrics_info)
            )

            logger.info(f"FINAL ANSWER:\n\n{final_output}\n\n")

            return (
                final_output,
                usage_model_metrics_info,
                output_tool,
                context_data_from_tool,
            )

    async def _process_output(
        self, output: Dict[str, Any], usage_model_metrics_info: LLMProcessMetrics
    ) -> Tuple[str, Dict[str, Any], str]:
        """Process the output from the agent."""
        output_tool = {}
        context_data_from_tool = ""

        if self.tools is None:
            logger.info(f"[CarAgentMain] [inference] EXECUTE WITHOUT TOOLS")
            return output.content, output_tool, context_data_from_tool

        logger.info(f"[CarAgentMain] [inference] EXECUTE WITH TOOLS")

        # Process intermediate steps if available
        if "intermediate_steps" in output and len(output["intermediate_steps"]) == 1:
            output_tool, context_data_from_tool = (
                await self._process_intermediate_steps(output, usage_model_metrics_info)
            )

        return output.get("output"), output_tool, context_data_from_tool

    async def _process_intermediate_steps(
        self, output: Dict[str, Any], usage_model_metrics_info: LLMProcessMetrics
    ) -> Tuple[Dict[str, Any], str]:
        """Process intermediate steps from tool execution."""
        output_tool = {}
        context_data_from_tool = ""

        intermediate_step = output["intermediate_steps"]
        if not isinstance(intermediate_step[0], tuple):
            return output_tool, context_data_from_tool

        output_tool_to_agent_final = intermediate_step[0][-1]

        # Handle GraphRAG mode
        if (
            self.tool_get_info_config.config_knowledge_manager.mode
            == KnowledgeManagerMode.GRAPHRAG
        ):
            if not self._run_reduce_response:
                # Process structured output
                hash_output_tool, references, context_data_from_tool, update_metrics = (
                    await GraphRAGResponseProcessor.process_graphrag_response(
                        output,
                        output["output"],
                        self.pubsub_channel_name,
                        self._run_reduce_response,
                    )
                )

                # Update metrics if available
                if update_metrics:
                    usage_model_metrics_info.input_metrics = (
                        update_metrics.input_metrics
                    )
                    usage_model_metrics_info.output_metrics = (
                        update_metrics.output_metrics
                    )
                    usage_model_metrics_info.total_metrics = (
                        update_metrics.total_metrics
                    )
                    usage_model_metrics_info.time_exe = update_metrics.time_exe
            else:
                hash_output_tool = hash_string(output_tool_to_agent_final)
                references = None
        else:
            hash_output_tool = hash_string(output_tool_to_agent_final)
            context_data_from_tool = intermediate_step[0]
            references = None

        # Get cached data
        cache_key = f"cache_helper_{self.pubsub_channel_name}_{hash_output_tool}"
        cached_data = await aget_cached_data(cache_key)

        if cached_data:
            usage_model_metrics_info.additional_metrics = {
                k: v for k, v in cached_data.items() if k != "graph_rag_answer_response"
            }

            # Process GraphRAG response if available
            if (
                self.tool_get_info_config.config_knowledge_manager.mode
                == KnowledgeManagerMode.GRAPHRAG
            ):
                graph_res_data = cached_data.get("graph_rag_answer_response")
                if graph_res_data:
                    output_tool, context_data_from_tool = (
                        await self._process_graph_response(
                            graph_res_data, output, output_tool, references
                        )
                    )

        return output_tool, context_data_from_tool

    async def _process_graph_response(
        self,
        graph_res_data: Dict[str, Any],
        output: Dict[str, Any],
        output_tool: Dict[str, Any],
        references: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], str]:
        """Process GraphRAG response data."""
        context_data_from_tool = graph_res_data.get("context_text", "")

        # Process resources
        resources = graph_res_data.get("resources", {})
        if (
            resources
            and self.tools
            and not all(len(v) == 0 for v in resources.values())
        ):
            output["output"] = await GraphRAGResponseProcessor.add_resources_to_output(
                output, resources, references, self._run_reduce_response
            )

        # Process context data
        if references:
            context_tool_data = GraphRAGResponseProcessor.process_context_data(
                graph_res_data, references, resources
            )
            output_tool.update(context_tool_data)

        return output_tool, context_data_from_tool
