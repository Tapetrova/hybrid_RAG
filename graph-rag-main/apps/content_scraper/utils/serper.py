"""Google Serper API wrapper - Refactored version."""

import os
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import aiohttp
import requests
from typing_extensions import Literal

from libs.python.utils.logger import logger


class SearchType(str, Enum):
    """Supported search types."""

    NEWS = "news"
    SEARCH = "search"
    PLACES = "places"
    IMAGES = "images"


class GoogleSerperAPIWrapper:
    """Wrapper around the Serper.dev Google Search API.

    This wrapper provides both synchronous and asynchronous methods
    for searching Google via the Serper API.

    You can create a free API key at https://serper.dev.
    Set the environment variable SERPER_API_KEY with your API key.
    """

    # API configuration
    BASE_URL = "https://google.serper.dev"
    DEFAULT_NUM_RESULTS = 10
    DEFAULT_COUNTRY = "us"
    DEFAULT_LANGUAGE = "en"

    # Result keys mapping
    RESULT_KEY_MAPPING = {
        SearchType.NEWS: "news",
        SearchType.PLACES: "places",
        SearchType.IMAGES: "images",
        SearchType.SEARCH: "organic",
    }

    def __init__(
        self,
        k: Optional[int] = None,
        gl: Optional[str] = None,
        hl: Optional[str] = None,
        domain_filter: Optional[List[str]] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the Google Serper API wrapper.

        Args:
            k: Number of results to return
            gl: Country code for Google search (e.g., 'us', 'uk')
            hl: Language code for Google search (e.g., 'en', 'fr')
            domain_filter: List of domains to filter results
            api_key: Serper API key (defaults to env var SERPER_API_KEY)

        Raises:
            ValueError: If API key is not provided
        """
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Serper API key not found. "
                "Please set SERPER_API_KEY environment variable or pass api_key parameter."
            )

        self.k = k or self.DEFAULT_NUM_RESULTS
        self.gl = gl or self.DEFAULT_COUNTRY
        self.hl = hl or self.DEFAULT_LANGUAGE
        self.domain_filter = domain_filter or []
        self.type: SearchType = SearchType.SEARCH
        self.tbs: Optional[str] = None
        self.aiosession: Optional[aiohttp.ClientSession] = None

        logger.info(
            "Initialized GoogleSerperAPIWrapper with k=%d, gl=%s, hl=%s, domains=%s",
            self.k,
            self.gl,
            self.hl,
            self.domain_filter,
        )

    def _create_headers(self) -> Dict[str, str]:
        """Create request headers."""
        return {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

    def _create_search_query(self, query: str) -> str:
        """Create search query with domain filters if specified.

        Args:
            query: Base search query

        Returns:
            Modified query with domain filters
        """
        if not self.domain_filter:
            return query

        domain_queries = [f"site:{domain}" for domain in self.domain_filter]
        return f"{query} {' OR '.join(domain_queries)}"

    def _build_params(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Build request parameters.

        Args:
            query: Search query
            **kwargs: Additional parameters

        Returns:
            Dictionary of parameters
        """
        params = {
            "q": query,
            "gl": kwargs.get("gl", self.gl),
            "hl": kwargs.get("hl", self.hl),
            "num": kwargs.get("num", self.k),
        }

        if self.tbs:
            params["tbs"] = self.tbs

        # Add any additional non-None parameters
        params.update({k: v for k, v in kwargs.items() if v is not None})

        return params

    def results(self, query: str, **kwargs: Any) -> Dict:
        """Run synchronous query through Google Search.

        Args:
            query: Search query
            **kwargs: Additional search parameters

        Returns:
            Raw search results dictionary
        """
        modified_query = self._create_search_query(query)
        params = self._build_params(modified_query, **kwargs)

        url = f"{self.BASE_URL}/{self.type.value}"
        response = requests.post(url, headers=self._create_headers(), params=params)
        response.raise_for_status()

        results = response.json()
        logger.debug(
            "Retrieved %d results for query: %s",
            len(results.get(self.RESULT_KEY_MAPPING[self.type], [])),
            query,
        )

        return results

    async def aresults(self, query: str, **kwargs: Any) -> Dict:
        """Run asynchronous query through Google Search.

        Args:
            query: Search query
            **kwargs: Additional search parameters

        Returns:
            Raw search results dictionary
        """
        modified_query = self._create_search_query(query)
        params = self._build_params(modified_query, **kwargs)

        url = f"{self.BASE_URL}/{self.type.value}"

        # Use provided session or create temporary one
        if self.aiosession:
            async with self.aiosession.post(
                url,
                params=params,
                headers=self._create_headers(),
                raise_for_status=True,
            ) as response:
                results = await response.json()
        else:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    params=params,
                    headers=self._create_headers(),
                    raise_for_status=True,
                ) as response:
                    results = await response.json()

        logger.debug(
            "Retrieved %d async results for query: %s",
            len(results.get(self.RESULT_KEY_MAPPING[self.type], [])),
            query,
        )

        return results

    def run(self, query: str, **kwargs: Any) -> str:
        """Run query and parse results into a string.

        Args:
            query: Search query
            **kwargs: Additional search parameters

        Returns:
            Parsed results as a string
        """
        results = self.results(query, **kwargs)
        return self._parse_results(results)

    async def arun(self, query: str, **kwargs: Any) -> str:
        """Run async query and parse results into a string.

        Args:
            query: Search query
            **kwargs: Additional search parameters

        Returns:
            Parsed results as a string
        """
        results = await self.aresults(query, **kwargs)
        return self._parse_results(results)

    def _parse_results(self, results: dict) -> str:
        """Parse search results into a readable string.

        Args:
            results: Raw search results

        Returns:
            Formatted string of results
        """
        snippets = self._parse_snippets(results)
        return " ".join(snippets)

    def _parse_snippets(self, results: dict) -> List[str]:
        """Extract snippets from search results.

        Args:
            results: Raw search results

        Returns:
            List of text snippets
        """
        snippets = []

        # Check for answer box
        if answer_box := results.get("answerBox"):
            if answer := answer_box.get("answer"):
                return [answer]
            elif snippet := answer_box.get("snippet"):
                return [snippet.replace("\n", " ")]
            elif highlighted := answer_box.get("snippetHighlighted"):
                return highlighted

        # Check for knowledge graph
        if kg := results.get("knowledgeGraph"):
            snippets.extend(self._parse_knowledge_graph(kg))

        # Parse regular results
        result_key = self.RESULT_KEY_MAPPING[self.type]
        for result in results.get(result_key, [])[: self.k]:
            if snippet := result.get("snippet"):
                snippets.append(snippet)

            # Add attributes if present
            for attr, value in result.get("attributes", {}).items():
                snippets.append(f"{attr}: {value}.")

        return snippets if snippets else ["No good Google Search Result was found"]

    def _parse_knowledge_graph(self, kg: dict) -> List[str]:
        """Parse knowledge graph data.

        Args:
            kg: Knowledge graph dictionary

        Returns:
            List of formatted strings
        """
        snippets = []
        title = kg.get("title", "")

        if entity_type := kg.get("type"):
            snippets.append(f"{title}: {entity_type}.")

        if description := kg.get("description"):
            snippets.append(description)

        for attribute, value in kg.get("attributes", {}).items():
            snippets.append(f"{title} {attribute}: {value}.")

        return snippets

    @classmethod
    async def async_get_urls_from_search_results(
        cls, results: Optional[Dict] = None
    ) -> List[Tuple[str, str]]:
        """Extract URLs from search results.

        Args:
            results: Search results dictionary from Serper API

        Returns:
            List of tuples containing (title, url)
        """
        urls = []

        if not results:
            return urls

        # Extract URLs from organic results
        for page in results.get("organic", []):
            if "link" in page and "title" in page:
                urls.append((page["title"], page["link"]))

        logger.debug("Extracted %d URLs from search results", len(urls))
        return urls


def create_google_search_domain_filter(query: str, domains: List[str]) -> str:
    """Generate a Google search query with domain filters.

    Args:
        query: The search query string
        domains: List of domain strings to filter

    Returns:
        Modified query with domain filters

    Example:
        >>> create_google_search_domain_filter("python", ["github.com", "stackoverflow.com"])
        'python site:github.com OR site:stackoverflow.com'
    """
    if not domains:
        return query

    domain_queries = [f"site:{domain}" for domain in domains]
    return f"{query} {' OR '.join(domain_queries)}"
