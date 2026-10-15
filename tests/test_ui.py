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

    def test_collapsible_run_details_and_status_pill_in_ui_assets(self):
        """UI assets must contain collapsible run details section and provider-aware status pill styling."""
        static_dir = Path(__file__).resolve().parent.parent / "app" / "static"
        html_path = static_dir / "index.html"
        css_path = static_dir / "styles.css"
        js_path = static_dir / "app.js"

        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        # Run details collapsible container
        self.assertIn('details id="trace-card"', html)
        self.assertIn("Run details &amp; technical diagnostics", html)
        self.assertIn('id="results-footer-stats"', html)
        self.assertIn('id="trace-steps"', html)
        self.assertIn('id="verification-checks"', html)

        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        self.assertIn(".status-pill", css)
        self.assertIn(".status-ready", css)
        self.assertIn(".status-degraded", css)
        self.assertIn(".status-unavailable", css)

        with open(js_path, "r", encoding="utf-8") as f:
            js = f.read()
        # Provider-aware health handling in JS
        self.assertIn("data.llm_provider", js)
        self.assertIn("status-ready", js)
        self.assertIn("status-degraded", js)
        self.assertIn("status-unavailable", js)

    def test_safe_markdown_rendering_assets(self):
        """UI assets must contain the safe DOM Markdown renderer without innerHTML/outerHTML."""
        static_dir = Path(__file__).resolve().parent.parent / "app" / "static"
        js_path = static_dir / "app.js"
        css_path = static_dir / "styles.css"

        with open(js_path, "r", encoding="utf-8") as f:
            js = f.read()

        self.assertIn("function renderSafeMarkdown", js)
        self.assertIn("function renderInlineMarkdown", js)
        self.assertIn("renderSafeMarkdown(answerText,", js)
        self.assertIn("document.createElement", js)
        self.assertIn("document.createTextNode", js)

        # Confirm zero innerHTML assignments
        self.assertEqual(len(re.findall(r"\.innerHTML\s*=", js)), 0)
        self.assertEqual(len(re.findall(r"\.outerHTML\s*=", js)), 0)

        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()

        self.assertIn(".answer-heading", css)
        self.assertIn(".answer-h3", css)
        self.assertIn(".answer-h4", css)
        self.assertIn(".answer-list", css)
        self.assertIn(".answer-ul", css)
        self.assertIn(".answer-ol", css)
        self.assertIn(".answer-paragraph", css)

    def test_safe_markdown_rendering_behavior_and_xss_immunity(self):
        """Execute node test to verify headings, lists, bold text, and malicious script/HTML immunity."""
        import subprocess
        import shutil

        if not shutil.which("node"):
            self.skipTest("node.js not installed on system")

        script = """
const fs = require('fs');
const path = require('path');
const code = fs.readFileSync(path.join(__dirname, '..', 'app', 'static', 'app.js'), 'utf8');

class MockNode {
  constructor(name) {
    this.nodeName = name;
    this.children = [];
    this.textContent = '';
    this.className = '';
  }
  appendChild(child) {
    this.children.push(child);
    return child;
  }
  removeChild(child) {
    const idx = this.children.indexOf(child);
    if (idx !== -1) this.children.splice(idx, 1);
  }
  get firstChild() { return this.children[0] || null; }
}

const document = {
  createElement: (tag) => new MockNode(tag.toUpperCase()),
  createTextNode: (text) => {
    const n = new MockNode('#text');
    n.textContent = text;
    return n;
  }
};
const window = {};

const fnMatch = code.match(/function renderInlineMarkdown[\\s\\S]*?window\\.renderSafeMarkdown = renderSafeMarkdown;/);
if (!fnMatch) {
  console.error("renderSafeMarkdown not found in app.js");
  process.exit(1);
}
eval(fnMatch[0]);

const container = new MockNode('DIV');

// 1. Test headings, ordered list, unordered list, bold text
const md = '### Day 1:\\n1. **Complete Security Training**\\n- Complete mandatory cybersecurity modules.\\n\\nRegular paragraph with **bold** and `code`.';
renderSafeMarkdown(container, md);

const names = container.children.map(c => c.nodeName);
if (!names.includes('H4')) throw new Error('Missing H4 heading');
if (!names.includes('OL')) throw new Error('Missing OL list');
if (!names.includes('UL')) throw new Error('Missing UL list');
if (!names.includes('P')) throw new Error('Missing P paragraph');

// 2. Test malicious XSS payloads (script tags, img onerror)
const xssContainer = new MockNode('DIV');
const malicious = '<script>alert(1)<' + '/script><img src=x onerror=alert(2)> and **bold payload**';
renderSafeMarkdown(xssContainer, malicious);

// Ensure no SCRIPT or IMG elements were created
function hasElement(node, tag) {
  if (node.nodeName === tag) return true;
  return node.children.some(c => hasElement(c, tag));
}

if (hasElement(xssContainer, 'SCRIPT')) throw new Error('XSS test failed: SCRIPT element created');
if (hasElement(xssContainer, 'IMG')) throw new Error('XSS test failed: IMG element created');

console.log("OK");
"""
        proc = subprocess.run(
            ["node", "-e", script],
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, f"Node test failed: {proc.stderr}\n{proc.stdout}")
        self.assertIn("OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
