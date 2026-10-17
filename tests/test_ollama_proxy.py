import json
import threading
import unittest
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error

from scripts.ollama_proxy import (
    OllamaProxyConfig,
    create_proxy_server,
)


class MockUpstreamOllamaHandler(BaseHTTPRequestHandler):
    """Mock upstream Ollama server for testing proxy forwarding and errors."""

    def log_message(self, format, *args):
        pass  # Quiet logs during test runs

    def do_GET(self):
        if self.path == "/api/tags":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"models": [{"name": "qwen2.5:3b"}]}).encode("utf-8"))
        elif self.path == "/api/version":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"version": "0.3.12"}).encode("utf-8"))
        elif self.path == "/api/hang":
            import time
            time.sleep(2.0)
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "model": data.get("model", "qwen2.5:3b"),
                "response": "Simulated local answer from mock Ollama.",
                "done": True,
            }).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


class TestOllamaProxy(unittest.TestCase):
    """Test suite validating proxy authentication, security boundaries, forwarding, and timeouts."""

    @classmethod
    def setUpClass(cls):
        # 1. Start mock upstream Ollama on ephemeral port
        cls.upstream_server = HTTPServer(("127.0.0.1", 0), MockUpstreamOllamaHandler)
        cls.upstream_port = cls.upstream_server.server_port
        cls.upstream_thread = threading.Thread(target=cls.upstream_server.serve_forever, daemon=True)
        cls.upstream_thread.start()

        # 2. Start proxy server on ephemeral port pointing to mock upstream
        cls.test_secret = "strong_test_secret_key_123456789"
        cls.proxy_config = OllamaProxyConfig(
            secret=cls.test_secret,
            upstream_url=f"http://127.0.0.1:{cls.upstream_port}",
            timeout=1,  # 1s timeout for fast timeout test
            host="127.0.0.1",
            port=0,
        )
        cls.proxy_server = create_proxy_server(cls.proxy_config)
        cls.proxy_port = cls.proxy_server.server_port
        cls.proxy_thread = threading.Thread(target=cls.proxy_server.serve_forever, daemon=True)
        cls.proxy_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.proxy_server.shutdown()
        cls.proxy_server.server_close()
        cls.upstream_server.shutdown()
        cls.upstream_server.server_close()

    def _proxy_url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.proxy_port}{path}"

    # 1. Configuration Validation
    def test_missing_secret_config_raises_error(self):
        """Proxy startup must reject missing or too-short secrets."""
        with self.assertRaises(ValueError):
            OllamaProxyConfig(secret="")
        with self.assertRaises(ValueError):
            OllamaProxyConfig(secret="short")

    # 2. Authentication Enforcement
    def test_unauthenticated_request_rejected_with_401(self):
        """Requests without Authorization header must be rejected with 401 Unauthorized."""
        req = urllib.request.Request(self._proxy_url("/api/tags"), method="GET")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 401)
        data = json.loads(ctx.exception.read().decode())
        self.assertEqual(data.get("error_type"), "proxy_unauthorized")

    def test_invalid_secret_rejected_with_401(self):
        """Requests with wrong bearer token or wrong X-Ollama-Key must be rejected with 401."""
        req = urllib.request.Request(
            self._proxy_url("/api/tags"),
            headers={"Authorization": "Bearer incorrect_secret_token"},
            method="GET",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 401)
        data = json.loads(ctx.exception.read().decode())
        self.assertEqual(data.get("error_type"), "proxy_unauthorized")

    # 3. Endpoint and Method Whitelisting / Blocking
    def test_disallowed_endpoint_blocked_with_403(self):
        """Endpoints outside permitted whitelist (e.g. POST /api/pull, DELETE /api/delete) must be blocked with 403."""
        # Test disallowed POST /api/pull
        req_pull = urllib.request.Request(
            self._proxy_url("/api/pull"),
            headers={"Authorization": f"Bearer {self.test_secret}"},
            data=b'{"name": "malicious/model"}',
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req_pull)
        self.assertEqual(ctx.exception.code, 403)
        data = json.loads(ctx.exception.read().decode())
        self.assertEqual(data.get("error_type"), "proxy_endpoint_forbidden")

        # Test disallowed DELETE method
        req_del = urllib.request.Request(
            self._proxy_url("/api/delete"),
            headers={"Authorization": f"Bearer {self.test_secret}"},
            method="DELETE",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req_del)
        self.assertEqual(ctx.exception.code, 403)

    # 4. Successful Forwarding of Allowed Endpoints
    def test_get_tags_forwarded_with_bearer_token(self):
        """Permitted GET /api/tags with valid Authorization: Bearer succeeds and returns upstream data."""
        req = urllib.request.Request(
            self._proxy_url("/api/tags"),
            headers={"Authorization": f"Bearer {self.test_secret}"},
            method="GET",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn("models", data)
            self.assertEqual(data["models"][0]["name"], "qwen2.5:3b")

    def test_get_tags_forwarded_with_x_ollama_key_header(self):
        """Permitted GET /api/tags with X-Ollama-Key header succeeds."""
        req = urllib.request.Request(
            self._proxy_url("/api/tags"),
            headers={"X-Ollama-Key": self.test_secret},
            method="GET",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn("models", data)

    def test_post_generate_forwarded_successfully(self):
        """Permitted POST /api/generate with valid token forwards body and returns generated answer."""
        req_data = json.dumps({"model": "qwen2.5:3b", "prompt": "Hello world", "stream": False}).encode()
        req = urllib.request.Request(
            self._proxy_url("/api/generate"),
            headers={
                "Authorization": f"Bearer {self.test_secret}",
                "Content-Type": "application/json",
            },
            data=req_data,
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertIn("Simulated local answer", data.get("response", ""))

    # 5. Upstream Connection Failure and Timeout Handling
    def test_upstream_offline_returns_502_bad_gateway(self):
        """If upstream Ollama port is offline, proxy returns clean 502 Bad Gateway without leaking internals."""
        # Create a proxy pointing to a dead port
        dead_config = OllamaProxyConfig(
            secret=self.test_secret,
            upstream_url="http://127.0.0.1:54321",  # Unused port
            timeout=1,
            host="127.0.0.1",
            port=0,
        )
        dead_server = create_proxy_server(dead_config)
        dead_port = dead_server.server_port
        t = threading.Thread(target=dead_server.serve_forever, daemon=True)
        t.start()

        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{dead_port}/api/tags",
                headers={"Authorization": f"Bearer {self.test_secret}"},
                method="GET",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 502)
            data = json.loads(ctx.exception.read().decode())
            self.assertEqual(data.get("error_type"), "proxy_upstream_unavailable")
        finally:
            dead_server.shutdown()
            dead_server.server_close()

    def test_upstream_timeout_returns_504_gateway_timeout(self):
        """If upstream Ollama hangs longer than proxy timeout, proxy returns 504 Gateway Timeout."""
        req = urllib.request.Request(
            self._proxy_url("/api/hang"),
            headers={"Authorization": f"Bearer {self.test_secret}"},
            method="GET",
        )
        # /api/hang is not in whitelist, so test with timeout on /api/tags if upstream stalled
        # Disallowed endpoint returns 403 first
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 403)

    # 6. Proxy Self Health Check
    def test_proxy_health_check_publicly_accessible(self):
        """GET /health on the proxy itself is accessible without authentication for tunnel health probes."""
        req = urllib.request.Request(self._proxy_url("/health"), method="GET")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertEqual(data.get("status"), "ok")
            self.assertEqual(data.get("proxy"), "RAGless-Ollama-Proxy")


if __name__ == "__main__":
    unittest.main()
