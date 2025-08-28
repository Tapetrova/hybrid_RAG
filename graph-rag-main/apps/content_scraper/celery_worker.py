"""Celery worker for asynchronous content processing - Refactored version."""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from urllib.parse import quote

from celery import Celery
from celery.exceptions import MaxRetriesExceededError
import html2text
from playwright.sync_api import sync_playwright

from apps.content_scraper.models import ResponseParseContentData
from libs.python.schemas.configuration import Config
from libs.python.utils.cacher import cache_data, get_cached_data, hash_string
from libs.python.utils.logger import logger


# Configuration
class CeleryConfig:
    """Centralized configuration for Celery."""

    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")
    REDIS_PASSWORD = quote(os.getenv("REDIS_PASSWORD_LLM", ""))
    CELERY_BROKER_DB = os.getenv("CELERY_BROKER_CONTENT_SCRAPER_DB", "0")
    CELERY_BACKEND_DB = os.getenv("CELERY_BACKEND_CONTENT_SCRAPER_DB", "0")

    PLAYWRIGHT_TIMEOUT_MS = int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", "180000"))

    @classmethod
    def get_broker_url(cls) -> str:
        """Get Celery broker URL."""
        return (
            f"redis://default:{cls.REDIS_PASSWORD}@{cls.REDIS_HOST}:"
            f"{cls.REDIS_PORT}/{cls.CELERY_BROKER_DB}"
        )

    @classmethod
    def get_backend_url(cls) -> str:
        """Get Celery backend URL."""
        return (
            f"redis://default:{cls.REDIS_PASSWORD}@{cls.REDIS_HOST}:"
            f"{cls.REDIS_PORT}/{cls.CELERY_BACKEND_DB}"
        )


# Initialize Celery
celery_app = Celery(
    "tasks",
    broker=CeleryConfig.get_broker_url(),
    backend=CeleryConfig.get_backend_url(),
)


class ContentScraper:
    """Handle content scraping operations."""

    @staticmethod
    def scrape_with_playwright(url: str, timeout_ms: int = None) -> str:
        """Scrape content from URL using Playwright.

        Args:
            url: URL to scrape
            timeout_ms: Timeout in milliseconds

        Returns:
            Scraped HTML content or error message
        """
        timeout_ms = timeout_ms or CeleryConfig.PLAYWRIGHT_TIMEOUT_MS
        logger.info("Starting Playwright scraping for URL: %s", url)

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                try:
                    page = browser.new_page()
                    page.set_default_timeout(timeout_ms)
                    page.goto(url)
                    content = page.content()
                    logger.info("Successfully scraped content from: %s", url)
                    return content
                finally:
                    browser.close()
        except Exception as e:
            error_msg = f"Error scraping {url}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    @staticmethod
    def html_to_text(
        html_content: str, ignore_links: bool = True, ignore_images: bool = True
    ) -> str:
        """Convert HTML content to plain text.

        Args:
            html_content: HTML content to convert
            ignore_links: Whether to ignore links
            ignore_images: Whether to ignore images

        Returns:
            Plain text content
        """
        h = html2text.HTML2Text()
        h.ignore_links = ignore_links
        h.ignore_images = ignore_images
        return h.handle(html_content)


class ContentCache:
    """Handle content caching operations."""

    @staticmethod
    def get_cache_key(
        url: str, type_parsing: str, ignore_links: bool, ignore_images: bool
    ) -> str:
        """Generate cache key for content.

        Args:
            url: URL being processed
            type_parsing: Type of parsing
            ignore_links: Whether links are ignored
            ignore_images: Whether images are ignored

        Returns:
            Cache key string
        """
        cache_data_dict = {
            "url": url,
            "type_parsing": type_parsing,
            "ignore_links": ignore_links,
            "ignore_images": ignore_images,
        }
        return f"pars_{hash_string(json.dumps(cache_data_dict))}"

    @staticmethod
    def get_cached_content(cache_key: str) -> Optional[Dict[str, str]]:
        """Get cached content if available.

        Args:
            cache_key: Cache key

        Returns:
            Cached content dict or None
        """
        cached = get_cached_data(cache_key)
        if cached:
            logger.info("Retrieved content from cache: %s", cache_key)
        return cached

    @staticmethod
    def cache_content(cache_key: str, content: str, html_content: str) -> None:
        """Cache content for future use.

        Args:
            cache_key: Cache key
            content: Plain text content
            html_content: HTML content
        """
        cache_data(
            cache_key,
            {
                "content": content,
                "html_content": html_content,
            },
        )
        logger.info("Cached content: %s", cache_key)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def parse_content(
    self,
    user_id: str,
    session_id: str,
    dialog_id: str,
    batch_id: str,
    batch_input_count: int,
    query: str,
    id_prs: str,
    pars_verif_res_id: Optional[str],
    url: str,
    google_search_results_id: str,
    config: Dict,
    ignore_links: bool = True,
    ignore_images: bool = True,
    parser_result_from_db: Optional[Dict[str, Any]] = None,
    pars_verif_from_db: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Parse content from URL asynchronously.

    This task handles the actual content scraping and parsing,
    with caching and error handling.

    Args:
        self: Celery task instance
        user_id: User identifier
        session_id: Session identifier
        dialog_id: Dialog identifier
        batch_id: Batch processing identifier
        batch_input_count: Number of items in batch
        query: Original search query
        id_prs: Parser result ID
        pars_verif_res_id: Parser verification result ID (unused)
        url: URL to parse
        google_search_results_id: Google search results ID
        config: Configuration dictionary
        ignore_links: Whether to ignore links in content
        ignore_images: Whether to ignore images in content
        parser_result_from_db: Existing parser result if available
        pars_verif_from_db: Existing verification result (unused)

    Returns:
        Dictionary with parsed content data
    """
    try:
        logger.info("Starting content parsing task for URL: %s", url)

        # Parse configuration
        config_obj = Config(**config)
        type_parsing = config_obj.config_content_scraper.type_parsing.value

        # Check if content already exists
        if parser_result_from_db:
            logger.info("Using existing parser result from database for URL: %s", url)
            content = parser_result_from_db.get("content", "")
            html_content = parser_result_from_db.get("html_content", "")
            is_already_parsing_in_db = True
        else:
            # Check cache
            cache_key = ContentCache.get_cache_key(
                url, type_parsing, ignore_links, ignore_images
            )
            cached_content = ContentCache.get_cached_content(cache_key)

            if cached_content:
                content = cached_content.get("content", "")
                html_content = cached_content.get("html_content", "")
            else:
                # Scrape new content
                start_time = datetime.now()

                # Scrape HTML
                html_content = ContentScraper.scrape_with_playwright(url)

                # Convert to text
                content = ContentScraper.html_to_text(
                    html_content, ignore_links, ignore_images
                )

                # Cache the result
                ContentCache.cache_content(cache_key, content, html_content)

                duration = (datetime.now() - start_time).total_seconds()
                logger.info("Parsed URL in %.2f seconds: %s", duration, url)

            is_already_parsing_in_db = False

        # Build response data
        response_data = ResponseParseContentData(
            id=id_prs,
            html_content=html_content,
            content=content,
            user_id=user_id,
            session_id=session_id,
            query=query,
            dialog_id=dialog_id,
            batch_id=batch_id,
            batch_input_count=batch_input_count,
            google_search_results_id=google_search_results_id,
            config=config_obj,
            content_verification_response=None,
            is_already_parsing_in_db=is_already_parsing_in_db,
            is_already_verification_parsing_in_db=False,
        )

        return json.loads(response_data.model_dump_json())

    except Exception as e:
        logger.exception("Task failed with error: %s", str(e))

        # Retry if possible
        try:
            logger.info(
                "Retrying task, attempt %d/%d",
                self.request.retries + 1,
                self.max_retries,
            )
            return self.retry(exc=e)
        except MaxRetriesExceededError:
            logger.error("Max retries exceeded for URL: %s", url)
            return {"error": f"Max retries exceeded: {str(e)}", "url": url}
