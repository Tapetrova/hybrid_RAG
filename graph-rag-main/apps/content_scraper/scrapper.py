from typing import Sequence, Optional, Tuple, List

from apps.content_scraper.utils.serper import GoogleSerperAPIWrapper


class ScraperUrls:
    """Scraper for URLs from Google search results."""

    def __init__(
        self,
        k: Optional[int] = None,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        domain_filter: Optional[List[str]] = None,
    ):
        """Initialize the ScraperUrls.

        Args:
            k: Number of results to return
            gl: Country code for Google search
            hl: Language code for Google search
            domain_filter: List of domains to filter results
        """
        self.google_loader = GoogleSerperAPIWrapper(
            k=k, gl=gl, hl=hl, domain_filter=domain_filter
        )

    async def scrape(self, query: str) -> Sequence[Tuple[str, str]]:
        """Scrape URLs from Google search results.

        Args:
            query: Search query string

        Returns:
            Sequence of tuples containing (title, url)
        """
        results = await self.google_loader.aresults(query=query)
        urls = await self.google_loader.async_get_urls_from_search_results(
            results=results
        )
        return urls
