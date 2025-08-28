"""Business logic services for content scraper."""

import uuid
from typing import List, Optional, Tuple
from datetime import datetime, timezone

from apps.content_scraper.models import RequestTaskData
from apps.content_scraper.scrapper import ScraperUrls
from apps.content_scraper.celery_worker import parse_content
from libs.python.databases.database import (
    GoogleSearchResults,
    ParserResults,
)
from libs.python.utils.logger import logger


class SearchService:
    """Service for handling search operations."""

    @staticmethod
    async def get_or_create_search_results(
        query: str,
        request: RequestTaskData,
        scraper: ScraperUrls,
    ) -> Tuple[str, List, bool]:
        """Get existing search results or create new ones.

        Args:
            query: Search query
            request: Request data
            scraper: Scraper instance

        Returns:
            Tuple of (result_id, results, is_new)
        """
        config = request.config.config_content_scraper

        # Build query conditions
        conditions = (
            (GoogleSearchResults.google_query == query)
            & (GoogleSearchResults.type == config.google_search_type.value)
            & (GoogleSearchResults.domain_filter == config.domain_filter)
            & (GoogleSearchResults.country == config.country.value)
            & (GoogleSearchResults.locale == config.locale.value)
            & (GoogleSearchResults.top_k_url == config.top_k_url)
        )

        # Check for existing query
        existing_query = await GoogleSearchResults.query.where(conditions).gino.first()

        if existing_query:
            logger.info("Found existing search results for query: %s", query)
            return existing_query.id, existing_query.results, False

        # Create new search results
        logger.info("Creating new search results for query: %s", query)
        results = await scraper.google_loader.aresults(query=query)
        result_id = str(uuid.uuid4())

        await GoogleSearchResults.create(
            id=result_id,
            user_id=request.user_id,
            session_id=request.session_id,
            dialog_id=request.dialog_id,
            domain_filter=config.domain_filter,
            type=config.google_search_type.value,
            google_query=query,
            results=results,
            top_k_url=config.top_k_url,
            country=config.country.value,
            locale=config.locale.value,
            datetime_created=datetime.now(timezone.utc),
        )

        return result_id, results, True


class ParserService:
    """Service for handling parsing operations."""

    @staticmethod
    async def get_or_create_parser_result(
        url: Tuple[str, str],
        type_parsing: str,
        parser_res_id: str,
        batch_id: str,
        batch_input_count: int,
        google_search_results_id: str,
    ) -> Optional[dict]:
        """Get existing parser result or create placeholder for new parsing.

        Args:
            url: Tuple of (title, url)
            type_parsing: Type of parsing to perform
            parser_res_id: Parser result ID
            batch_id: Batch ID
            batch_input_count: Number of items in batch
            google_search_results_id: Google search results ID

        Returns:
            Existing parser result dict or None if new
        """
        # Check for existing parsed result
        conditions = (
            (ParserResults.url == url[1])
            & (ParserResults.type == type_parsing)
            & (ParserResults.is_parsed == True)
        )

        parser_result = await ParserResults.query.where(conditions).gino.first()

        if parser_result:
            logger.info("Found existing parser result for URL: %s", url[1])
            return parser_result.to_dict()

        # Create placeholder for new parsing
        logger.info("Creating parser result placeholder for URL: %s", url[1])

        await ParserResults.create(
            id=parser_res_id,
            batch_id=batch_id,
            batch_input_count=batch_input_count,
            title=url[0],
            url=url[1],
            type=type_parsing,
            html_content=None,
            content=None,
            is_parsed=False,
            google_search_results_id=google_search_results_id,
            datetime_created=datetime.now(timezone.utc),
            datetime_updated=datetime.now(timezone.utc),
        )

        return None

    @staticmethod
    def dispatch_parsing_task(
        request: RequestTaskData,
        url: str,
        parser_res_id: str,
        search_id: str,
        batch_id: str,
        batch_input_count: int,
        query: str,
        parser_result_dict: Optional[dict] = None,
    ) -> None:
        """Dispatch a parsing task to Celery.

        Args:
            request: Request data
            url: URL to parse
            parser_res_id: Parser result ID
            search_id: Search results ID
            batch_id: Batch ID
            batch_input_count: Number of items in batch
            query: Original search query
            parser_result_dict: Existing parser result if available
        """
        parse_content.delay(
            user_id=request.user_id,
            session_id=request.session_id,
            dialog_id=request.dialog_id,
            id_prs=parser_res_id,
            pars_verif_res_id=None,
            url=url,
            google_search_results_id=search_id,
            batch_id=batch_id,
            query=query,
            batch_input_count=batch_input_count,
            parser_result_from_db=parser_result_dict,
            pars_verif_from_db=None,
            config=request.config.model_dump(),
        )

        logger.debug("Dispatched parsing task for URL: %s", url)
