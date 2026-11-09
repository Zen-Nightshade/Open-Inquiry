from __future__ import annotations

from abc import ABC, abstractmethod
from .models import SearchResponse


class SearchProviderError(Exception):
    """Exception raised when a search provider fails to complete a search."""

    def __init__(self, message: str, query: str, cause: Exception | None = None):
        super().__init__(message)
        self.query = query
        self.cause = cause


class BaseSearchProvider(ABC):
    """Abstract base provider for web search backends.
    
    Allows DuckDuckGo, Brave, Tavily, SerpAPI, or mock providers to be swapped
    transparently without modifying Researcher orchestrators.
    """

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> SearchResponse:
        """Execute a single search query and return a structured SearchResponse.
        
        Must distinguish between:
        - SearchStatus.SUCCESS (results found)
        - SearchStatus.EMPTY (executed successfully, 0 hits found)
        - SearchStatus.FAILED (network error, rate-limit, timeout, provider outage)
        """
        pass
