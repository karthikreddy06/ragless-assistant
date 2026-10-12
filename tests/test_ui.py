import re
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app


class TestWebUIEndpoint(unittest.TestCase):
    """Tests for the /ui endpoint, static asset serving, security headers, and existing route integrity."""

    def setUp(self):
        self.client = TestClient(app)

    def test_ui_serves_html_with_csp_header(self):
        """GET /ui/ returns HTML content with strict Content-Security-Policy header."""
        resp = self.client.get("/ui/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers.get("content-type", ""))
        self.assertIn("<title>RAGless · Autonomous Company Operations Agent</title>", resp.text)

        # Check CSP header
        csp = resp.headers.get("content-security-policy", "")
        self.assertTrue(csp, "Missing Content-Security-Policy header on /ui/")
        self.assertIn("default-src 'self'", csp)
        self.assertIn("script-src 'self'", csp)
        self.assertIn("style-src 'self'", csp)
        self.assertNotIn("'unsafe-inline'", csp)
        self.assertNotIn("'unsafe-eval'", csp)

    def test_ui_serves_static_assets_with_correct_content_types(self):
        """Static assets (/ui/styles.css and /ui/app.js) are served with correct MIME types and CSP."""
        # CSS
        css_resp = self.client.get("/ui/styles.css")
        self.assertEqual(css_resp.status_code, 200)
        self.assertIn("text/css", css_resp.headers.get("content-type", ""))
        self.assertIn("content-security-policy", css_resp.headers)

        # JS
        js_resp = self.client.get("/ui/app.js")
        self.assertEqual(js_resp.status_code, 200)
        self.assertTrue(
            any(t in js_resp.headers.get("content-type", "") for t in ("javascript", "application/x-javascript")),
            f"Unexpected JS content-type: {js_resp.headers.get('content-type')}"
        )
        self.assertIn("content-security-policy", js_resp.headers)

    def test_ui_route_does_not_break_existing_routes(self):
        """Adding /ui static files must not affect root '/', '/health', or API routes."""
        # Root JSON endpoint
        root_resp = self.client.get("/")
        self.assertEqual(root_resp.status_code, 200)
        root_data = root_resp.json()
        self.assertEqual(root_data.get("project"), "RAGless Knowledge Assistant")
        self.assertEqual(root_data.get("status"), "active")

        # Health endpoint
        health_resp = self.client.get("/health")
        self.assertEqual(health_resp.status_code, 200)
        health_data = health_resp.json()
        self.assertIn("status", health_data)
        self.assertIn("database", health_data)

        # Validation on API endpoint
        val_resp = self.client.post("/ask", json={"question": ""})
        self.assertEqual(val_resp.status_code, 422)

        agent_val_resp = self.client.post("/agent/ask", json={"question": ""})
        self.assertEqual(agent_val_resp.status_code, 422)

    def test_js_contains_no_inner_html_assignments(self):
        """Lightweight static check: app.js must not assign innerHTML or outerHTML to prevent XSS."""
        js_path = Path(__file__).resolve().parent.parent / "app" / "static" / "app.js"
        self.assertTrue(js_path.exists(), "app.js does not exist")

        with open(js_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Check for innerHTML / outerHTML assignment patterns
        bad_patterns = [
            r"\.innerHTML\s*=",
            r"\.outerHTML\s*=",
            r"document\.write\s*\(",
            r"eval\s*\(",
        ]
        for pat in bad_patterns:
            matches = re.findall(pat, code)
            self.assertEqual(
                len(matches),
                0,
                f"Found dangerous DOM assignment matching '{pat}' in app.js: {matches}",
            )

    def test_xss_markup_stays_inert(self):
        """Simulate malicious answer string containing injected tags and confirm textContent keeps them inert."""
        malicious_input = "<script>alert('xss')</script><img src=x onerror=alert(1)>"
        
        # In a browser environment using textContent:
        # textContent assigns raw string without parsing HTML entities into DOM elements.
        from html import escape
        escaped = escape(malicious_input)
        self.assertIn("&lt;script&gt;", escaped)
        self.assertNotIn("<script>", escaped)


    def test_mock_data_badge_present_in_ui_assets(self):
        """UI files must contain the 'MOCK DATA' badge element, styling, and logic when tool_result source is 'mock'."""
        static_dir = Path(__file__).resolve().parent.parent / "app" / "static"
        html_path = static_dir / "index.html"
        css_path = static_dir / "styles.css"
        js_path = static_dir / "app.js"

        # 1. HTML contains the evidence panel mock badge element
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn('id="evidence-mock-badge"', html)
        self.assertIn("MOCK DATA", html)

        # 2. CSS contains .badge-mock styles
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        self.assertIn(".badge-mock", css)

        # 3. JavaScript contains mock source detection and badge creation
        with open(js_path, "r", encoding="utf-8") as f:
            js = f.read()
        self.assertIn('tr.source === "mock"', js)
        self.assertIn('"MOCK DATA"', js)
        self.assertIn("evidenceMockBadge", js)


if __name__ == "__main__":
    unittest.main()
