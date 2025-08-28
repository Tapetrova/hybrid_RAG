"""Pydantic models for content scraper - Refactored version."""

from typing import List, Optional, Any, Dict
from pydantic import Field, HttpUrl, validator, UUID4, field_validator

from libs.python.schemas.basic_models import BaseModelUpd
from libs.python.schemas.configuration import Config


class ResponseGoogleScrapperData(BaseModelUpd):
    """Response model for Google search scraping results."""

    id: str = Field(
        ...,
        description="Unique identifier for the search results",
        example="550e8400-e29b-41d4-a716-446655440000",
    )
    urls: List[List[str]] = Field(
        ...,
        description="List of [title, url] pairs from search results",
        example=[["Example Title", "https://example.com"]],
    )
    batch_id: Optional[str] = Field(
        None,
        description="Batch processing identifier",
        example="550e8400-e29b-41d4-a716-446655440001",
    )
    batch_input_count: int = Field(
        ..., ge=0, description="Number of items in the batch", example=10
    )

    @field_validator("urls")
    def validate_urls(cls, v):
        """Ensure each URL entry has exactly 2 elements: title and URL."""
        for item in v:
            if len(item) != 2:
                raise ValueError(
                    "Each URL entry must have exactly 2 elements: [title, url]"
                )
            if not isinstance(item[0], str) or not isinstance(item[1], str):
                raise ValueError("Both title and URL must be strings")
        return v


class RequestTaskData(BaseModelUpd):
    """Request model for task data."""

    user_id: str = Field(
        ..., min_length=1, description="Unique user identifier", example="user_123"
    )
    session_id: str = Field(
        ..., min_length=1, description="Session identifier", example="session_456"
    )
    dialog_id: str = Field(
        ..., min_length=1, description="Dialog identifier", example="dialog_789"
    )
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query string",
        example="python web scraping best practices",
    )
    config: Config = Field(..., description="Configuration object for the task")


class ResponseParseContentData(BaseModelUpd):
    """Response model for parsed content data."""

    id: str = Field(
        ...,
        description="Parser result identifier",
        example="550e8400-e29b-41d4-a716-446655440002",
    )
    html_content: str = Field(..., description="Raw HTML content from the URL")
    content: str = Field(..., description="Extracted text content from the HTML")
    user_id: str = Field(..., description="User who initiated the parsing")
    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., description="Original search query")
    dialog_id: str = Field(..., description="Dialog identifier")
    batch_id: str = Field(..., description="Batch processing identifier")
    batch_input_count: int = Field(
        ..., ge=0, description="Number of items in the batch"
    )
    google_search_results_id: str = Field(
        ..., description="ID of the Google search results"
    )
    config: Config = Field(..., description="Configuration used for parsing")
    content_verification_response: Optional[Dict[str, Any]] = Field(
        None, description="Verification response data (if verification is enabled)"
    )
    is_already_parsing_in_db: bool = Field(
        False,
        description="Whether the content was already parsed and retrieved from DB",
    )
    is_already_verification_parsing_in_db: bool = Field(
        False, description="Whether verification was already done and retrieved from DB"
    )


class RequestGetData(BaseModelUpd):
    """Request model for getting data by query."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query to retrieve results for",
        example="python web scraping",
    )


class RequestGetContentSourceByUrls(BaseModelUpd):
    """Request model for getting content by URLs."""

    urls: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of URLs to retrieve content for",
        example=["https://example.com/page1", "https://example.com/page2"],
    )

    @field_validator("urls")
    def validate_urls(cls, v):
        """Basic URL validation."""
        for url in v:
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Invalid URL format: {url}")
        return v


class ResponseGoogleScrapperDataFull(BaseModelUpd):
    """Response model for full Google scraper data."""

    id: str = Field(..., description="Unique identifier for the parsed result")
    url: str = Field(..., description="URL that was scraped")
    type: str = Field(..., description="Type of parsing performed", example="html")
    content: str = Field(..., description="Extracted text content")
    title: Optional[str] = Field(None, description="Page title if available")
    html_content: Optional[str] = Field(
        None, description="Raw HTML content if available"
    )


class ResponseGoogleScrapperDataFullList(BaseModelUpd):
    """Response model for a list of full Google scraper data."""

    results: List[ResponseGoogleScrapperDataFull] = Field(
        ..., description="List of parsed results"
    )

    class Config:
        """Pydantic configuration."""

        schema_extra = {
            "example": {
                "results": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "url": "https://example.com",
                        "type": "html",
                        "content": "Example content",
                        "title": "Example Title",
                        "html_content": "<html>...</html>",
                    }
                ]
            }
        }


# Reddit-specific models (kept for backward compatibility but deprecated)
class RequestInsertRedditResults(BaseModelUpd):
    """Request model for inserting Reddit results.

    DEPRECATED: Reddit processing has been removed from the system.
    This model is kept for backward compatibility only.
    """

    id: str = Field(..., description="Result identifier")
    parser_results_id: str = Field(..., description="Parser results identifier")
    title_post: Optional[str] = Field(None, description="Reddit post title")
    data_type: Optional[str] = Field(None, description="Type of Reddit data")
    community_name: Optional[str] = Field(None, description="Reddit community name")
    body: Optional[str] = Field(None, description="Post body content")
    number_of_comments: Optional[int] = Field(None, ge=0, description="Comment count")
    username: Optional[str] = Field(None, description="Post author username")
    number_of_replies: Optional[int] = Field(None, ge=0, description="Reply count")
    up_votes: Optional[int] = Field(None, description="Upvote count")
    post_datetime_created: Optional[str] = Field(
        None, description="Post creation timestamp"
    )
    body_query_sim_ada_2_cosine: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Cosine similarity score for body"
    )
    title_post_query_sim_ada_2_cosine: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Cosine similarity score for title"
    )

    class Config:
        """Pydantic configuration."""

        deprecated = True
