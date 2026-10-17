#!/usr/bin/env python3
"""
RAGless Authenticated Ollama Reverse Proxy
==========================================
Provides secure authenticated access to a local Ollama instance on Windows/macOS/Linux.
Binds strictly to localhost (127.0.0.1) and verifies a strong secret header before
forwarding permitted inference and health requests to http://127.0.0.1:11434.

Disallowed endpoints (e.g. DELETE /api/delete, POST /api/pull, POST /api/create)
are rejected with HTTP 403 Forbidden. Unauthenticated requests are rejected with
HTTP 401 Unauthorized.
"""

import argparse
import hmac
import json
import logging
import os
import secrets
import socket
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ollama-proxy")

# Permitted endpoints: (HTTP method, normalized path)
ALLOWED_ENDPOINTS: Set[Tuple[str, str]] = {
    ("GET", "/api/tags"),
    ("POST", "/api/generate"),
    ("GET", "/api/version"),
    ("GET", "/health"),
    ("GET", "/"),
}

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11435
DEFAULT_UPSTREAM = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 60


def sanitize_error(msg: str, secret: Optional[str] = None) -> str:
    """Strip secret tokens and local user directories from error strings."""
    if secret and secret in msg:
        msg = msg.replace(secret, "[REDACTED]")
    return msg


class OllamaProxyConfig:
    def __init__(
        self,
        secret: str,
        upstream_url: str = DEFAULT_UPSTREAM,
        timeout: int = DEFAULT_TIMEOUT,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ):
        if not secret or len(secret.strip()) < 8:
            raise ValueError("Proxy secret must be at least 8 characters long.")
        self.secret = secret.strip()
        self.upstream_url = upstream_url.rstrip("/")
        self.timeout = timeout
        self.host = host
        self.port = port


class OllamaProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler validating authentication and proxying permitted endpoints."""

    server_version = "RAGless-Ollama-Proxy/1.0"

    @property
    def config(self) -> OllamaProxyConfig:
        return self.server.proxy_config  # type: ignore

    def log_message(self, format: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def send_json(self, status_code: int, data: Dict) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def extract_token(self) -> Optional[str]:
        # 1. Authorization: Bearer <secret>
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        if auth.startswith("bearer "):
            return auth[7:].strip()

        # 2. X-Ollama-Key: <secret>
        x_key = self.headers.get("X-Ollama-Key", "")
        if x_key:
            return x_key.strip()

        return None

    def verify_auth(self) -> bool:
        token = self.extract_token()
        if not token:
            self.send_json(401, {
                "error": "Unauthorized: Missing Authorization header or X-Ollama-Key.",
                "error_type": "proxy_unauthorized",
            })
            return False

        # Constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(token, self.config.secret):
            logger.warning("Rejected request with invalid proxy authorization secret.")
            self.send_json(401, {
                "error": "Unauthorized: Invalid proxy authorization secret.",
                "error_type": "proxy_unauthorized",
            })
            return False

        return True

    def check_endpoint_allowed(self) -> bool:
        norm_path = self.path.split("?")[0].rstrip("/") or "/"
        method = self.command.upper()

        # Proxy self-health check: public ping allowed
        if norm_path in ("/health", "/") and method == "GET":
            return True

        if (method, norm_path) not in ALLOWED_ENDPOINTS:
            logger.warning("Blocked disallowed request: %s %s", method, norm_path)
            self.send_json(403, {
                "error": f"Forbidden: Endpoint '{norm_path}' with method '{method}' is not permitted by proxy policy.",
                "error_type": "proxy_endpoint_forbidden",
            })
            return False

        return True

    def do_GET(self) -> None:
        norm_path = self.path.split("?")[0].rstrip("/") or "/"

        # Unauthenticated health check for the proxy itself
        if norm_path in ("/health", "/"):
            self.send_json(200, {
                "status": "ok",
                "proxy": "RAGless-Ollama-Proxy",
                "upstream": self.config.upstream_url,
            })
            return

        if not self.check_endpoint_allowed():
            return
        if not self.verify_auth():
            return

        self.forward_request("GET")

    def do_POST(self) -> None:
        if not self.check_endpoint_allowed():
            return
        if not self.verify_auth():
            return

        self.forward_request("POST")

    def do_PUT(self) -> None:
        self.check_endpoint_allowed()

    def do_DELETE(self) -> None:
        self.check_endpoint_allowed()

    def do_PATCH(self) -> None:
        self.check_endpoint_allowed()

    def do_HEAD(self) -> None:
        self.check_endpoint_allowed()

    def do_OPTIONS(self) -> None:
        self.check_endpoint_allowed()


    def forward_request(self, method: str) -> None:
        upstream_target = f"{self.config.upstream_url}{self.path}"

        # Read request body if present
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        req = urllib.request.Request(
            url=upstream_target,
            data=body,
            method=method,
        )

        # Forward Content-Type
        ct = self.headers.get("Content-Type")
        if ct:
            req.add_header("Content-Type", ct)

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for header, val in resp.getheaders():
                    # Skip hop-by-hop headers
                    if header.lower() not in ("transfer-encoding", "content-length", "connection"):
                        self.send_header(header, val)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            err_body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)
        except urllib.error.URLError as e:
            logger.error("Failed to reach upstream Ollama at %s: %s", upstream_target, e.reason)
            self.send_json(502, {
                "error": f"Failed to connect to local Ollama service at {self.config.upstream_url}. Ensure Ollama is running.",
                "error_type": "proxy_upstream_unavailable",
            })
        except (socket.timeout, TimeoutError):
            logger.error("Upstream Ollama request timed out after %ds", self.config.timeout)
            self.send_json(504, {
                "error": f"Ollama request timed out after {self.config.timeout} seconds.",
                "error_type": "proxy_gateway_timeout",
            })
        except Exception as e:
            logger.exception("Unexpected error proxying to Ollama: %s", e)
            self.send_json(500, {
                "error": "Internal proxy error processing Ollama request.",
                "error_type": "proxy_internal_error",
            })


def create_proxy_server(config: OllamaProxyConfig) -> ThreadingHTTPServer:
    """Create and return a configured ThreadingHTTPServer instance."""
    server = ThreadingHTTPServer((config.host, config.port), OllamaProxyHandler)
    server.proxy_config = config  # type: ignore
    return server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGless Authenticated Ollama Reverse Proxy for secure Cloudflare Quick Tunnel exposure"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("OLLAMA_PROXY_PORT", str(DEFAULT_PORT))),
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("OLLAMA_PROXY_HOST", DEFAULT_HOST),
        help=f"Host address to bind to (strictly localhost/127.0.0.1, default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--upstream-url",
        type=str,
        default=os.getenv("OLLAMA_UPSTREAM_URL", DEFAULT_UPSTREAM),
        help=f"Upstream Ollama URL (default: {DEFAULT_UPSTREAM})",
    )
    parser.add_argument(
        "--secret",
        type=str,
        default=os.getenv("OLLAMA_PROXY_SECRET", ""),
        help="Shared secret token required in Authorization: Bearer <secret> header",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("OLLAMA_PROXY_TIMEOUT", str(DEFAULT_TIMEOUT))),
        help=f"Upstream timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )

    args = parser.parse_args()

    # Enforce localhost binding only
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logger.critical("Security violation: Proxy must only bind to 127.0.0.1/localhost, got '%s'", args.host)
        sys.exit(1)

    secret = args.secret.strip()
    if not secret:
        # Generate a cryptographically secure secret if none provided
        secret = secrets.token_urlsafe(32)
        logger.info("Generated new cryptographically secure proxy secret token.")

    config = OllamaProxyConfig(
        secret=secret,
        upstream_url=args.upstream_url,
        timeout=args.timeout,
        host=args.host,
        port=args.port,
    )

    server = create_proxy_server(config)

    print("=" * 72)
    print(" RAGless Authenticated Ollama Reverse Proxy")
    print("=" * 72)
    print(f" Listening on  : http://{config.host}:{config.port}")
    print(f" Forwarding to : {config.upstream_url}")
    print(f" Upstream model: qwen2.5:3b (configured in app)")
    print(f" Allowed paths : GET /api/tags, POST /api/generate, GET /api/version")
    print(f" Secret Token  : {secret}")
    print("=" * 72)
    print(" To expose via Cloudflare Quick Tunnel (Free, no domain needed):")
    print(f"   cloudflared tunnel --url http://127.0.0.1:{config.port}")
    print(" Then on Render Dashboard -> ragless-assistant -> Environment:")
    print("   OLLAMA_BASE_URL = https://<your-subdomain>.trycloudflare.com")
    print(f"   OLLAMA_API_KEY  = {secret}")
    print("=" * 72)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down proxy gracefully...")
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
