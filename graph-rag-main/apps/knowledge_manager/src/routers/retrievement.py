import json
import os
import time
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import APIRouter, HTTPException, status
from qdrant_client.conversions import common_types as qd_types

from apps.knowledge_manager.src.qdrant_engine import AsyncNeuralSearcher
from apps.knowledge_manager.src.schemas.basic_models import Content
from apps.knowledge_manager.src.schemas.default_rag_schemas import (
    ResponseVectorRetrieve,
    ResponseGraphRAGRetrieve,
    RequestDefaultRAGRetrieve,
    RequestGraphRAGRetrieve,
)
from apps.knowledge_manager.src.search_engine_helper.search_engine_helper import (
    search_engine,
)
from apps.knowledge_manager.src.utils import (
    aconvert_graphrag_res_from_pd_to_json,
)
from libs.python.databases.database import (
    ParserVerificationResults,
    ParserResults,
    GoogleSearchResults,
)
from libs.python.schemas.configuration import ConfigContentScraperMode
from libs.python.schemas.events import KnowledgeManagerEvent
from libs.python.schemas.graphrag.graph_rag_config import SearchMode
from libs.python.utils.logger import logger
from libs.python.utils.request_utils import asend_request

router = APIRouter()


class RetrievalConfig:
    """Configuration constants for retrieval operations"""

    TIMEOUT = 120.0
    TIME_SLEEP_CHECK_SEC = 5.0
    MAX_ITERATIONS = 360
    CONTENT_SCRAPER_ENDPOINT = os.getenv(
        "ENDPOINT_CONTENT_SCRAPER",
        "http://localhost:8099/search_content",
    )


class ParseStatus(Enum):
    """Parsing status enumeration"""

    PENDING = "pending"
    PARSING = "parsing"
    COMPLETED = "completed"
    FAILED = "failed"


class ContentScraperService:
    """Service for handling content scraping operations"""

    def __init__(self):
        self.config = RetrievalConfig()

    async def wait_for_parser_results(
        self, google_search_result: GoogleSearchResults
    ) -> Optional[List[ParserResults]]:
        """Wait for parser results to be available"""
        if not google_search_result:
            logger.error("[/vector_retrieve] google_search_result is NONE")
            return None

        for iteration in range(self.config.MAX_ITERATIONS):
            parser_results = await ParserResults.query.where(
                google_search_result.id == ParserResults.google_search_results_id
            ).gino.all()

            if parser_results and len(parser_results) > 0:
                logger.info(
                    f"[/vector_retrieve] SUCCESS! Found {len(parser_results)} parser results "
                    f"after {iteration + 1} iterations"
                )
                return parser_results

            logger.info(
                f"[/vector_retrieve] Waiting... iteration {iteration + 1}, "
                f"sleeping for {self.config.TIME_SLEEP_CHECK_SEC}s"
            )
            time.sleep(self.config.TIME_SLEEP_CHECK_SEC)

        return None

    async def wait_for_parsing_completion(
        self, parser_results: List[ParserResults], threshold_ratio: float = 0.75
    ) -> bool:
        """Wait for parsing to complete for a threshold of results"""
        for iteration in range(self.config.MAX_ITERATIONS):
            parsed_count = sum(1 for result in parser_results if result.is_parsed)
            threshold = int(threshold_ratio * len(parser_results))

            if parsed_count >= threshold:
                logger.info(
                    f"[/vector_retrieve] Parsing complete! {parsed_count}/{len(parser_results)} "
                    f"parsed (threshold: {threshold})"
                )
                return True

            logger.info(
                f"[/vector_retrieve] Waiting for parsing... {parsed_count}/{len(parser_results)} "
                f"parsed (need {threshold})"
            )
            time.sleep(self.config.TIME_SLEEP_CHECK_SEC)

        return False

    async def wait_for_verification_results(
        self, parser_results: List[ParserResults]
    ) -> bool:
        """Wait for verification results to be available"""
        parsed_results = [r for r in parser_results if r.is_parsed]

        for iteration in range(self.config.MAX_ITERATIONS):
            verified_count = 0

            for parser_result in parsed_results:
                verif_result = await ParserVerificationResults.query.where(
                    parser_result.id == ParserVerificationResults.parser_results_id
                ).gino.first()

                if (
                    verif_result
                    and verif_result.verified_content_summarization is not None
                    and verif_result.is_provided_answer_useful is not None
                ):
                    verified_count += 1

            if verified_count == len(parsed_results):
                logger.info(
                    f"[/vector_retrieve] Verification complete! "
                    f"{verified_count}/{len(parsed_results)} verified"
                )
                return True

            logger.info(
                f"[/vector_retrieve] Waiting for verification... "
                f"{verified_count}/{len(parsed_results)} verified"
            )
            time.sleep(self.config.TIME_SLEEP_CHECK_SEC)

        return False

    async def trigger_content_scraping(
        self, request: RequestDefaultRAGRetrieve
    ) -> Optional[GoogleSearchResults]:
        """Trigger content scraping for the given request"""
        data = {
            "user_id": request.user_id,
            "session_id": request.session_id,
            "dialog_id": request.dialog_id,
            "query": request.natural_query,
            "config": json.loads(request.config.json()),
        }

        try:
            await asend_request(data, endpoint=self.config.CONTENT_SCRAPER_ENDPOINT)
            logger.info(
                f"[/vector_retrieve] Sent request to {self.config.CONTENT_SCRAPER_ENDPOINT}"
            )

            # Query for Google search results
            domain_filter = self._get_domain_filter(request.config)
            return await self._query_google_results(request, domain_filter)

        except Exception as e:
            logger.error(f"Error triggering content scraping: {str(e)}")
            return None

    def _get_domain_filter(self, config) -> List[str]:
        """Extract domain filter from configuration"""
        if isinstance(config.config_content_scraper.domain_filter, list):
            return config.config_content_scraper.domain_filter
        return []

    async def _query_google_results(
        self, request: RequestDefaultRAGRetrieve, domain_filter: List[str]
    ) -> Optional[GoogleSearchResults]:
        """Query for Google search results"""
        config = request.config.config_content_scraper

        return await GoogleSearchResults.query.where(
            (GoogleSearchResults.google_query == request.natural_query)
            & (GoogleSearchResults.type == config.google_search_type.value)
            & (GoogleSearchResults.domain_filter == domain_filter)
            & (GoogleSearchResults.country == config.country.value)
            & (GoogleSearchResults.locale == config.locale.value)
            & (GoogleSearchResults.top_k_url == config.top_k_url)
        ).gino.first()


class VectorRetrievalService:
    """Service for vector-based content retrieval"""

    def __init__(self):
        self.content_scraper = ContentScraperService()

    async def retrieve_content(
        self, request: RequestDefaultRAGRetrieve
    ) -> ResponseVectorRetrieve:
        """Retrieve content based on vector similarity"""
        # Configure neural searcher
        config = request.config.config_knowledge_manager.config_retrieval_mode
        collection_name = self._format_collection_name(request)

        logger.info(
            f"[user_id={request.user_id}] Query: <{request.natural_query}>\n"
            f"Collection: {collection_name}, Top-K: {config.top_k_retrieval}, "
            f"Threshold: {config.score_threshold}"
        )

        # Initialize searcher and perform search
        neural_searcher = AsyncNeuralSearcher(
            collection_name=collection_name, embedder_model=config.embedder_model
        )

        retrieved_results = await neural_searcher.search(
            query=request.natural_query,
            with_payload=["text", "src"],
            top_k=config.top_k_retrieval,
            score_threshold=config.score_threshold,
        )

        # Convert results to content
        output = self._convert_results_to_content(retrieved_results)
        logger.info(f"Initial retrieval returned {len(output)} results")

        # Check if content scraping is needed
        events = []
        if await self._should_trigger_scraping(request, output):
            events.append(KnowledgeManagerEvent.GOTO_CONTENT_SCRAPER)
            output = await self._retrieve_with_scraping(request, neural_searcher)

        return ResponseVectorRetrieve(
            knowledge_content=output, knowledge_manager_events=events
        )

    def _format_collection_name(self, request: RequestDefaultRAGRetrieve) -> str:
        """Format collection name based on configuration"""
        config = request.config.config_knowledge_manager.config_retrieval_mode
        use_verif = (
            request.config.config_content_scraper.use_output_from_verif_as_content
        )

        return (
            f"{config.collection_name}--{config.chunk_size}--"
            f"{config.embedder_model.name}---output_verif_as_content-{use_verif}"
        )

    def _convert_results_to_content(
        self, results: List[qd_types.ScoredPoint]
    ) -> List[Content]:
        """Convert scored points to content objects"""
        return [
            Content(text=result.payload.get("text"), src=result.payload.get("src"))
            for result in results
        ]

    async def _should_trigger_scraping(
        self, request: RequestDefaultRAGRetrieve, output: List[Content]
    ) -> bool:
        """Determine if content scraping should be triggered"""
        if request.config.config_content_scraper.mode != ConfigContentScraperMode.ON:
            logger.info("Content scraper mode is OFF")
            return False

        threshold = (
            request.config.config_knowledge_manager.threshold_go_to_content_scraper
        )
        if len(output) >= threshold:
            logger.info(f"Sufficient content found: {len(output)} >= {threshold}")
            return False

        logger.info(
            f"Insufficient content: {len(output)} < {threshold}, triggering scraper"
        )
        return True

    async def _retrieve_with_scraping(
        self, request: RequestDefaultRAGRetrieve, neural_searcher: AsyncNeuralSearcher
    ) -> List[Content]:
        """Retrieve content with content scraping"""
        # Check for existing results
        domain_filter = self.content_scraper._get_domain_filter(request.config)
        google_result = await self.content_scraper._query_google_results(
            request, domain_filter
        )

        if not google_result:
            logger.info("No existing Google results, triggering new search")
            google_result = await self.content_scraper.trigger_content_scraping(request)

        if google_result:
            # Wait for scraping to complete
            await self._wait_for_scraping_completion(request, google_result)

            # Retry search after scraping
            time.sleep(RetrievalConfig.TIME_SLEEP_CHECK_SEC)
            retrieved_results = await neural_searcher.search(
                query=request.natural_query,
                with_payload=["text", "src"],
                top_k=request.config.config_knowledge_manager.config_retrieval_mode.top_k_retrieval,
                score_threshold=request.config.config_knowledge_manager.config_retrieval_mode.score_threshold,
            )

            output = self._convert_results_to_content(retrieved_results)
            logger.info(f"After scraping: retrieved {len(output)} results")
            return output

        return []

    async def _wait_for_scraping_completion(
        self, request: RequestDefaultRAGRetrieve, google_result: GoogleSearchResults
    ) -> None:
        """Wait for content scraping to complete"""
        # Wait for parser results
        parser_results = await self.content_scraper.wait_for_parser_results(
            google_result
        )

        if parser_results:
            # Wait for parsing to complete
            await self.content_scraper.wait_for_parsing_completion(parser_results)

            # Wait for verification if configured
            if request.config.config_content_scraper.use_output_from_verif_as_content:
                await self.content_scraper.wait_for_verification_results(parser_results)


class GraphRAGRetrievalService:
    """Service for GraphRAG-based content retrieval"""

    async def retrieve(
        self, request: RequestGraphRAGRetrieve
    ) -> ResponseGraphRAGRetrieve:
        """Retrieve content using GraphRAG"""
        logger.info(
            f"[user_id={request.user_id}] GraphRAG retrieval for query: "
            f"<{request.natural_query}>"
        )

        search_mode = (
            request.config.config_knowledge_manager.config_retrieval_mode.search_mode
        )

        # Setup GraphRAG
        await search_engine.setup_graphrag(config=request.config)

        # Perform search based on mode
        if search_mode == SearchMode.global_mode:
            result = await self._global_search(request)
        elif search_mode == SearchMode.local_mode:
            result = await self._local_search(request)
        elif search_mode == SearchMode.question_generation_mode:
            # TODO: Implement question generation mode
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Question generation mode not implemented",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid search mode: {search_mode}",
            )

        return ResponseGraphRAGRetrieve(
            graph_rag_answer=result, knowledge_manager_events=[]
        )

    async def _global_search(self, request: RequestGraphRAGRetrieve) -> Dict[str, Any]:
        """Perform global search"""
        st_time = time.time()
        config = request.config.config_knowledge_manager.config_retrieval_mode

        result = await search_engine.asearch_global(
            query=request.natural_query,
            conversation_history=None,  # TODO: Support conversation history
            run_reduce_response=config.global_search.run_reduce_response,
        )

        result = await aconvert_graphrag_res_from_pd_to_json(result)
        logger.info(f"Global search completed in {time.time() - st_time:.2f}s")
        return result

    async def _local_search(self, request: RequestGraphRAGRetrieve) -> Dict[str, Any]:
        """Perform local search"""
        st_time = time.time()
        config = request.config.config_knowledge_manager.config_retrieval_mode

        result = await search_engine.asearch_local(
            query=request.natural_query,
            conversation_history=None,  # TODO: Support conversation history
            run_reduce_response=config.local_search.run_reduce_response,
        )

        result = await aconvert_graphrag_res_from_pd_to_json(result)
        logger.info(f"Local search completed in {time.time() - st_time:.2f}s")
        return result


# Route handlers
@router.post(
    "/vector_retrieve",
    response_model=ResponseVectorRetrieve,
    summary="Retrieve content using vector similarity search",
    status_code=status.HTTP_200_OK,
)
async def vector_retrieve(request: RequestDefaultRAGRetrieve):
    """
    Retrieve content using vector similarity search.

    Args:
        request: Request containing query and configuration

    Returns:
        ResponseVectorRetrieve: Retrieved content and events

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        service = VectorRetrievalService()
        return await service.retrieve_content(request)
    except Exception as e:
        logger.exception(f"Error in vector_retrieve: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}",
        )


@router.post(
    "/graphrag/retrieve",
    response_model=ResponseGraphRAGRetrieve,
    summary="Retrieve content using GraphRAG",
    status_code=status.HTTP_200_OK,
)
async def graphrag_retrieve(request: RequestGraphRAGRetrieve):
    """
    Retrieve content using GraphRAG search.

    Args:
        request: Request containing query and configuration

    Returns:
        ResponseGraphRAGRetrieve: GraphRAG search results

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        service = GraphRAGRetrievalService()
        return await service.retrieve(request)
    except Exception as e:
        logger.exception(f"Error in graphrag_retrieve: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GraphRAG retrieval failed: {str(e)}",
        )
