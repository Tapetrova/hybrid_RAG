import time
from fastapi import APIRouter, HTTPException

from apps.content_scraper.models import (
    ResponseGoogleScrapperDataFullList,
    ResponseGoogleScrapperDataFull,
    RequestGetContentSourceByUrls,
)
from libs.python.databases.database import ParserResults
from libs.python.utils.logger import logger

router = APIRouter()


@router.post("/get_content_source_by_urls")
async def get_content_source_by_urls(
    urls: RequestGetContentSourceByUrls,
) -> ResponseGoogleScrapperDataFullList:
    """Get content source data for given URLs.

    Args:
        urls: Request containing list of URLs

    Returns:
        ResponseGoogleScrapperDataFullList: List of content data
    """
    logger.info(f"[retrieve urls={urls.urls}]")
    main_start_process = time.time()

    if not urls.urls:
        raise HTTPException(status_code=400, detail="URLs list cannot be empty")
    parser_results = await ParserResults.query.where(
        ParserResults.url.in_(urls.urls)
    ).gino.all()

    results = []
    if parser_results:
        logger.info(f"Found {len(parser_results)} results")

        for r in parser_results:
            # Log only essential information
            logger.debug(
                f"Processing result - id: {r.id}, url: {r.url}, "
                f"has_content: {bool(r.content)}, has_html: {bool(r.html_content)}"
            )

            results.append(
                ResponseGoogleScrapperDataFull(
                    id=r.id,
                    title=r.title,
                    url=r.url,
                    type=r.type,
                    html_content=r.html_content,
                    content=r.content,
                )
            )
    else:
        logger.info("No results found for provided URLs")

    response = ResponseGoogleScrapperDataFullList(results=results)

    logger.info(
        f"[get_content_source_by_urls] execution time: {time.time() - main_start_process:.2f}s"
    )

    return response
