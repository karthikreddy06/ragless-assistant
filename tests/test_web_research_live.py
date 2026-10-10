import os
import unittest
import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.agent.web_search import TavilyClient


class TestWebResearchLive(unittest.TestCase):
    """
    Live verification suite for Tavily Search API.
    Auto-skips cleanly when TAVILY_API_KEY is not configured or is a placeholder.
    """

    def setUp(self):
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key or any(
            marker in api_key.lower()
            for marker in ("placeholder", "mock", "your_tavily_api_key_here", "dummy", "test")
        ):
            pytest.skip("TAVILY_API_KEY not configured with live credentials; skipping live web research tests.")
        self.client = TavilyClient(api_key=api_key)

    def test_live_python_release_query(self):
        """Query 1: Latest stable Python release returns valid URL and grounded facts."""
        result = self.client.search("latest stable Python release", max_results=2)
        self.assertTrue(result["success"], f"Tavily search failed: {result.get('error')}")
        self.assertGreaterEqual(len(result["results"]), 1)
        self.assertGreaterEqual(len(result["sources"]), 1)

        first_url = result["results"][0]["url"]
        self.assertTrue(first_url.startswith("http://") or first_url.startswith("https://"))
        self.assertTrue(any("python" in r["title"].lower() or "python" in r["snippet"].lower() for r in result["results"]))

    def test_live_uk_leave_statutory_minimum_query(self):
        """Query 2: 2026 UK statutory annual leave minimum returns valid URL and grounded answer."""
        result = self.client.search("2026 UK statutory annual leave minimum days", max_results=2)
        self.assertTrue(result["success"], f"Tavily search failed: {result.get('error')}")
        self.assertGreaterEqual(len(result["results"]), 1)

        first_url = result["results"][0]["url"]
        self.assertTrue(first_url.startswith("http://") or first_url.startswith("https://"))


if __name__ == "__main__":
    unittest.main()
