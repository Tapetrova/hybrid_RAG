"""Content Scraper API - Refactored version."""

import json
import uuid
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException

from apps.content_scraper.models import (
    RequestTaskData,
    ResponseGoogleScrapperData,
    RequestGetData,
    ResponseGoogleScrapperDataFullList,
    ResponseGoogleScrapperDataFull,
)
from apps.content_scraper.scrapper import ScraperUrls
from apps.content_scraper.services import SearchService, ParserService
from apps.content_scraper.routers import data
from libs.python.databases.database import (
    db,
    postgresql_db_url,
    GoogleSearchResults,
    ParserResults,
)
from libs.python.utils import health
from libs.python.utils.logger import logger


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Handle application lifecycle events."""
    # Startup
    try:
        await db.set_bind(postgresql_db_url)
        await db.gino.create_all()
        logger.info("Database connection established successfully")
    except Exception as e:
        logger.error("Failed to connect to PostgreSQL: %s", str(e))
        raise

    yield

    # Shutdown
    try:
        await db.pop_bind().close()
        logger.info("Database connection closed successfully")
    except Exception as e:
        logger.error("Failed to close database connection: %s", str(e))
        raise


app = FastAPI(
    title="Content Scraper API",
    description="API for scraping and parsing web content",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/search_content", response_model=ResponseGoogleScrapperData)
async def search_content(request: RequestTaskData) -> ResponseGoogleScrapperData:
    """Search content from Google and initiate parsing tasks.

    Args:
        request: Request data containing query and configuration

    Returns:
        ResponseGoogleScrapperData: Response with URLs and batch information

    Raises:
        HTTPException: If query is None or processing fails
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    try:
        logger.info(
            "Processing search request for user %s with query: %s",
            request.user_id,
            request.query,
        )

        # Extract configuration
        config = request.config.config_content_scraper
        metadata = (
            request.config.config_knowledge_manager.config_retrieval_mode.metadata
        )
        metadata.update(
            {
                "query": request.query,
                "country": config.country.value,
                "locale": config.locale.value,
            }
        )

        # Initialize scraper
        domain_filter = (
            config.domain_filter if isinstance(config.domain_filter, list) else []
        )
        scraper = ScraperUrls(
            k=config.top_k_url,
            gl=config.country.value,
            hl=config.locale.value,
            domain_filter=domain_filter,
        )

        # Get or create search results
        search_id, results, _ = await SearchService.get_or_create_search_results(
            request.query, request, scraper
        )

        # Extract URLs from search results
        urls = await scraper.google_loader.async_get_urls_from_search_results(
            results=results
        )

        if not urls:
            logger.warning("No URLs found for query: %s", request.query)
            return ResponseGoogleScrapperData(
                id=search_id,
                urls=[],
                batch_id=str(uuid.uuid4()),
                batch_input_count=0,
            )

        # Create batch for processing
        batch_id = str(uuid.uuid4())
        batch_input_count = len(urls)

        # Process each URL
        for idx, url in enumerate(urls):
            await _process_url(
                url=url,
                idx=idx,
                total_urls=len(urls),
                request=request,
                search_id=search_id,
                batch_id=batch_id,
                batch_input_count=batch_input_count,
                metadata=metadata,
            )

        return ResponseGoogleScrapperData(
            id=search_id,
            urls=[list(u) for u in urls],
            batch_id=batch_id,
            batch_input_count=batch_input_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing search request: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


async def _process_url(
    url: tuple[str, str],
    idx: int,
    total_urls: int,
    request: RequestTaskData,
    search_id: str,
    batch_id: str,
    batch_input_count: int,
    metadata: dict,
) -> None:
    """Process a single URL for content parsing."""
    logger.info("Processing URL %d/%d: %s", idx + 1, total_urls, url[1])

    # Determine parsing type
    type_parsing = request.config.config_content_scraper.type_parsing.value
    parser_res_id = str(uuid.uuid4())

    # Check for existing parser result
    parser_result_dict = await ParserService.get_or_create_parser_result(
        url=url,
        type_parsing=type_parsing,
        parser_res_id=parser_res_id,
        batch_id=batch_id,
        batch_input_count=batch_input_count,
        google_search_results_id=search_id,
    )

    if parser_result_dict:
        parser_res_id = parser_result_dict["id"]

    # Update metadata and dispatch parsing task
    metadata["parser_res_id"] = parser_res_id

    ParserService.dispatch_parsing_task(
        request=request,
        url=url[1],
        parser_res_id=parser_res_id,
        search_id=search_id,
        batch_id=batch_id,
        batch_input_count=batch_input_count,
        query=request.query,
        parser_result_dict=parser_result_dict,
    )


@app.post("/get_content", response_model=ResponseGoogleScrapperDataFullList)
async def get_content(request: RequestGetData) -> ResponseGoogleScrapperDataFullList:
    """Retrieve parsed content for a given query.

    Args:
        request: Request data containing the query

    Returns:
        ResponseGoogleScrapperDataFullList: List of parsed results

    Raises:
        HTTPException: If query is None or not found
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    try:
        # Find search results
        search_result = await GoogleSearchResults.query.where(
            GoogleSearchResults.google_query == request.query
        ).gino.first()

        if not search_result:
            raise HTTPException(status_code=404, detail="Query not found")

        # Get all parser results for this search
        parser_results = await ParserResults.query.where(
            ParserResults.google_search_results_id == search_result.id
        ).gino.all()

        # Convert to response model
        results = [
            ResponseGoogleScrapperDataFull(
                id=r.id,
                title=r.title,
                url=r.url,
                type=r.type,
                html_content=r.html_content,
                content=r.content,
            )
            for r in parser_results
        ]

        return ResponseGoogleScrapperDataFullList(results=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error retrieving content: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(data.router, prefix="/data", tags=["data"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8099)
