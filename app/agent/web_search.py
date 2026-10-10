import html
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Environment variable defaults
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_CALLS_PER_REQUEST = 2
CACHE_TTL_SECONDS = 300.0
CACHE_MAX_ENTRIES = 100
MAX_SNIPPET_LENGTH = 500
MAX_TOTAL_WEB_CONTENT_LENGTH = 2000

# HTML tag stripping regex
HTML_TAG_REGEX = re.compile(r"<[^>]+?>")


def redact_api_key(text: str, key: Optional[str] = None) -> str:
    """Redacts any API key occurrences from log strings, errors, or exception messages."""
    if not text:
        return ""
    candidates = []
    if key and key.strip():
        candidates.append(key.strip())
    env_key = os.getenv("TAVILY_API_KEY", "").strip()
    if env_key:
        candidates.append(env_key)

    redacted = text
    for c in candidates:
        if len(c) >= 6 and c in redacted:
            redacted = redacted.replace(c, "[REDACTED]")

    # Generic key pattern redaction
    redacted = re.sub(r"tvly-[a-zA-Z0-9_-]{10,}", "[REDACTED]", redacted)
    redacted = re.sub(r"sk-[a-zA-Z0-9_-]{10,}", "[REDACTED]", redacted)
    return redacted


def strip_html(text: str) -> str:
    """Strips HTML tags and unescapes HTML entities from text."""
    if not text:
        return ""
    no_tags = HTML_TAG_REGEX.sub(" ", text)
    unescaped = html.unescape(no_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def sanitize_snippet(text: str, max_chars: int = MAX_SNIPPET_LENGTH) -> str:
    """Strips HTML and truncates snippet to safe bounded length."""
    clean = strip_html(text)
    if len(clean) > max_chars:
        return clean[:max_chars].rstrip() + "..."
    return clean


class WebSearchCache:
    """In-memory LRU-style cache with TTL and maximum size cap."""

    def __init__(self, ttl_seconds: float = CACHE_TTL_SECONDS, max_entries: int = CACHE_MAX_ENTRIES):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        key = query.strip().lower()
        if key in self._cache:
            ts, val = self._cache[key]
            if time.time() - ts < self.ttl:
                logger.debug("WebSearchCache HIT for '%s'", key)
                return val
            else:
                del self._cache[key]
        return None

    def set(self, query: str, val: Dict[str, Any]):
        key = query.strip().lower()
        if len(self._cache) >= self.max_entries:
            # Evict oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        self._cache[key] = (time.time(), val)

    def clear(self):
        self._cache.clear()


class WebSearchClient(ABC):
    """Abstract interface for external web research providers."""

    @abstractmethod
    def search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Executes a web search.
        Returns a dict:
        {
            "success": bool,
            "query": str,
            "results": List[{"title": str, "url": str, "snippet": str}],
            "sources": List[{"title": str, "url": str}],
            "error": Optional[str],
            "from_cache": bool,
        }
        """
        pass


class FakeWebSearchClient(WebSearchClient):
    """In-memory mock client for deterministic offline tests."""

    def __init__(
        self,
        canned_results: Optional[List[Dict[str, str]]] = None,
        simulate_error: Optional[str] = None,
        simulate_status: Optional[int] = None,
        simulate_retries: int = 0,
    ):
        self.canned_results = canned_results or [
            {
                "title": "NVIDIA Nemotron 3 Architecture & Hardware Requirements",
                "url": "https://developer.nvidia.com/nemotron-3",
                "snippet": "NVIDIA Nemotron-3 models require 8x H100 SXM GPUs for FP8 inference and Triton Server 24.08.",
            },
            {
                "title": "NVIDIA Token Factory Deployment Guide",
                "url": "https://docs.nvidia.com/token-factory",
                "snippet": "Nemotron Token Factory deployment requires Linux 6.8 kernel and minimum 512GB system RAM.",
            },
        ]
        self.simulate_error = simulate_error
        self.simulate_status = simulate_status
        self.simulate_retries = simulate_retries
        self.calls: List[Dict[str, Any]] = []
        self._cache = WebSearchCache()

    def search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        self.calls.append({"query": query, "max_results": max_results})

        cached = self._cache.get(query)
        if cached is not None:
            res = dict(cached)
            res["from_cache"] = True
            return res

        if self.simulate_error:
            return {
                "success": False,
                "query": query,
                "results": [],
                "sources": [],
                "error": self.simulate_error,
                "from_cache": False,
            }

        # Handle simulated retries (e.g. 429 then success)
        if self.simulate_retries > 0:
            self.simulate_retries -= 1
            return {
                "success": False,
                "query": query,
                "results": [],
                "sources": [],
                "error": "Rate limit 429",
                "from_cache": False,
            }

        # Truncate and sanitize results
        bounded = self.canned_results[:max_results]
        clean_results = []
        clean_sources = []
        for r in bounded:
            snip = sanitize_snippet(r.get("snippet", ""))
            t = strip_html(r.get("title", "Untitled"))
            u = r.get("url", "").strip()
            clean_results.append({"title": t, "url": u, "snippet": snip})
            clean_sources.append({"title": t, "url": u})

        payload = {
            "success": True,
            "query": query,
            "results": clean_results,
            "sources": clean_sources,
            "error": None,
            "from_cache": False,
        }
        self._cache.set(query, payload)
        return payload


class TavilyClient(WebSearchClient):
    """
    HTTP client for Tavily Search API using the standard httpx library.
    Never crashes on missing API key; redacts keys from all logs and errors.
    Ignores Tavily's generated 'answer' field and extracts only titles, snippets, and URLs.
    """

    TAVILY_API_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 2,
    ):
        self._api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.timeout = timeout if timeout is not None else float(
            os.getenv("TAVILY_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
        )
        self.max_retries = max_retries
        self._cache = WebSearchCache()

    @property
    def is_available(self) -> bool:
        k = self._api_key.strip()
        return bool(k and k != "your_tavily_api_key_here")

    def search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        if not self.is_available:
            logger.info("Tavily API key not configured; web research unavailable.")
            return {
                "success": False,
                "query": query,
                "results": [],
                "sources": [],
                "error": "web research unavailable",
                "from_cache": False,
            }

        # Check in-memory cache
        cached = self._cache.get(query)
        if cached is not None:
            res = dict(cached)
            res["from_cache"] = True
            return res

        import httpx

        # Enforce max_results cap (1 to 5)
        n_results = max(1, min(5, max_results))

        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": n_results,
            "search_depth": "basic",
            "include_answer": False,  # Explicitly do NOT use Tavily's generated answer
            "include_raw_content": False,
        }

        headers = {"Content-Type": "application/json"}
        last_error = "Unknown error"

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self.TAVILY_API_URL, json=payload, headers=headers)

                if resp.status_code == 200:
                    data = resp.json()
                    raw_results = data.get("results", [])
                    clean_results = []
                    clean_sources = []
                    total_chars = 0

                    for r in raw_results[:n_results]:
                        title = strip_html(r.get("title", "Untitled"))
                        url = str(r.get("url", "")).strip()
                        raw_snip = r.get("content", "") or ""
                        snip = sanitize_snippet(raw_snip)

                        if total_chars + len(snip) > MAX_TOTAL_WEB_CONTENT_LENGTH:
                            snip = snip[:max(0, MAX_TOTAL_WEB_CONTENT_LENGTH - total_chars)]
                        total_chars += len(snip)

                        clean_results.append({"title": title, "url": url, "snippet": snip})
                        clean_sources.append({"title": title, "url": url})
                        if total_chars >= MAX_TOTAL_WEB_CONTENT_LENGTH:
                            break

                    result_obj = {
                        "success": True,
                        "query": query,
                        "results": clean_results,
                        "sources": clean_sources,
                        "error": None,
                        "from_cache": False,
                    }
                    self._cache.set(query, result_obj)
                    return result_obj

                elif resp.status_code == 429:
                    last_error = f"Tavily rate limit exceeded (HTTP 429)"
                    if attempt < self.max_retries:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                elif 500 <= resp.status_code < 600:
                    last_error = f"Tavily server error (HTTP {resp.status_code})"
                    if attempt < self.max_retries:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                else:
                    err_msg = redact_api_key(resp.text, self._api_key)
                    last_error = f"Tavily search failed (HTTP {resp.status_code}): {err_msg[:200]}"
                    break

            except httpx.TimeoutException:
                last_error = f"Tavily search timed out after {self.timeout}s"
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
            except Exception as e:
                err_str = redact_api_key(str(e), self._api_key)
                last_error = f"Tavily connection error: {err_str}"
                break

        safe_error = redact_api_key(last_error, self._api_key)
        logger.warning("Web search failed for query '%s': %s", query, safe_error)
        return {
            "success": False,
            "query": query,
            "results": [],
            "sources": [],
            "error": safe_error,
            "from_cache": False,
        }


# Global client instance holder
_global_web_search_client: Optional[WebSearchClient] = None


def get_web_search_client() -> WebSearchClient:
    global _global_web_search_client
    if _global_web_search_client is None:
        _global_web_search_client = TavilyClient()
    return _global_web_search_client


def set_default_web_search_client(client: Optional[WebSearchClient]):
    global _global_web_search_client
    _global_web_search_client = client
