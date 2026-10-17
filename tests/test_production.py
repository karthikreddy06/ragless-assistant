import concurrent.futures
import io
import json
import os
import sqlite3
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.auth import (
    agent_rate_limiter,
    get_client_ip,
    login_rate_limiter,
    register_rate_limiter,
)
from app.database import get_connection, initialize_database
from app.main import app, lifespan


class TestProductionSecurityAndConcurrency(unittest.TestCase):
    """Test suite validating production configuration, security headers, rate limiting, and SQLite concurrency."""

    def setUp(self):
        login_rate_limiter.reset()
        register_rate_limiter.reset()
        agent_rate_limiter.reset()
        self.client = TestClient(app)

    def tearDown(self):
        login_rate_limiter.reset()
        register_rate_limiter.reset()
        agent_rate_limiter.reset()

    # 1. SQLite Concurrency under multi-threaded writes
    def test_sqlite_concurrent_writes_with_wal_and_timeout(self):
        """Verify that multiple concurrent threads can write to SQLite simultaneously without locking errors."""
        initialize_database()

        def worker_write(thread_id: int):
            conn = get_connection(timeout=10.0)
            try:
                cursor = conn.cursor()
                # Insert a unique test user task
                cursor.execute(
                    """
                    INSERT INTO tasks (user_id, title, source, status)
                    VALUES (?, ?, 'user', 'open')
                    """,
                    (f"concurrent-user-{thread_id}", f"Task from thread {thread_id}"),
                )
                conn.commit()
                # Verify immediate readability
                cursor.execute(
                    "SELECT COUNT(*) FROM tasks WHERE user_id = ?",
                    (f"concurrent-user-{thread_id}",),
                )
                count = cursor.fetchone()[0]
                return count == 1
            finally:
                conn.close()

        # Run 10 parallel worker threads concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_write, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        self.assertEqual(len(results), 10)
        self.assertTrue(all(results), "All concurrent writes must succeed without database locks")

    # 2. Production startup secret validation
    def test_production_startup_fails_without_admin_token(self):
        """In production mode, missing or weak ADMIN_TOKEN must fail startup immediately."""
        import asyncio

        async def run_lifespan(env_vars):
            with patch.dict(os.environ, env_vars, clear=False):
                async with lifespan(app):
                    pass

        # 1. Missing ADMIN_TOKEN in production
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(run_lifespan({"ENVIRONMENT": "production", "ADMIN_TOKEN": ""}))
        self.assertIn("ADMIN_TOKEN must be set", str(ctx.exception))

        # 2. Weak placeholder ADMIN_TOKEN in production
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(run_lifespan({"ENVIRONMENT": "production", "ADMIN_TOKEN": "your_admin_token_here"}))
        self.assertIn("ADMIN_TOKEN must be set to a secure, random secret", str(ctx.exception))

        # 3. Missing NEBIUS_API_KEY when LLM_PROVIDER=nebius in production
        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(run_lifespan({
                "ENVIRONMENT": "production",
                "ADMIN_TOKEN": "strong_secret_token_1234567890",
                "LLM_PROVIDER": "nebius",
                "NEBIUS_API_KEY": "",
            }))
        self.assertIn("NEBIUS_API_KEY must be set", str(ctx.exception))

    # 3. Global Security Headers
    def test_global_security_headers_present_on_all_endpoints(self):
        """All HTTP responses must carry standard baseline security headers."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

        # Baseline security headers
        self.assertEqual(resp.headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(resp.headers.get("x-frame-options"), "DENY")
        self.assertEqual(resp.headers.get("referrer-policy"), "strict-origin-when-cross-origin")
        self.assertIn("camera=()", resp.headers.get("permissions-policy", ""))

        # UI endpoint must also have strict CSP
        resp_ui = self.client.get("/ui/")
        self.assertIn("default-src 'self'", resp_ui.headers.get("content-security-policy", ""))

    def test_hsts_header_present_under_https(self):
        """HSTS header must be set when HTTPS or COOKIE_SECURE is active."""
        resp = self.client.get("/health", headers={"X-Forwarded-Proto": "https"})
        self.assertIn("max-age=31536000", resp.headers.get("strict-transport-security", ""))

    # 4. Client IP handling and spoofing resistance
    def test_client_ip_extracts_rightmost_ip_from_proxy_chain(self):
        """When behind a proxy, get_client_ip must extract the rightmost entry added by the proxy."""
        class DummyRequest:
            headers = {"x-forwarded-for": "198.51.100.1, 203.0.113.19"}
            client = type("Client", (), {"host": "10.0.0.1"})()

        with patch.dict(os.environ, {"TRUST_PROXY_HEADERS": "true"}):
            ip = get_client_ip(DummyRequest())
            # Must return rightmost proxy-verified IP, NOT the attacker-spoofed 198.51.100.1
            self.assertEqual(ip, "203.0.113.19")

    def test_client_ip_ignores_forwarded_when_proxy_trust_disabled(self):
        """When TRUST_PROXY_HEADERS=false, X-Forwarded-For is ignored completely."""
        class DummyRequest:
            headers = {"x-forwarded-for": "198.51.100.1"}
            client = type("Client", (), {"host": "127.0.0.1"})()

        with patch.dict(os.environ, {"TRUST_PROXY_HEADERS": "false"}):
            ip = get_client_ip(DummyRequest())
            self.assertEqual(ip, "127.0.0.1")

    # 5. Registration Rate Limiting
    def test_registration_rate_limiting(self):
        """Multiple rapid registrations from the same IP must trigger HTTP 429."""
        headers = {"X-Forwarded-For": "198.51.100.77"}

        # Perform 5 registrations up to the default limit
        for i in range(5):
            payload = {
                "full_name": f"User {i}",
                "email": f"rateuser{i}@example.com",
                "password": "Password123!",
                "confirm_password": "Password123!",
            }
            resp = self.client.post("/auth/register", json=payload, headers=headers)
            self.assertEqual(resp.status_code, 201)

        # 6th registration from the same IP must be rejected with 429
        blocked_payload = {
            "full_name": "Blocked User",
            "email": "blocked@example.com",
            "password": "Password123!",
            "confirm_password": "Password123!",
        }
        resp_blocked = self.client.post("/auth/register", json=blocked_payload, headers=headers)
        self.assertEqual(resp_blocked.status_code, 429)
        self.assertIn("Too many registration attempts", resp_blocked.json().get("detail", ""))

    # 6. Agent Query Rate Limiting
    def test_agent_query_rate_limiting(self):
        """Exceeding the agent query rate limit triggers HTTP 429."""
        headers = {"X-Forwarded-For": "198.51.100.88"}

        # Simulate filling the agent query rate limit (custom limiter with max 3)
        with patch.object(agent_rate_limiter, "max_requests", 3):
            for _ in range(3):
                resp = self.client.post(
                    "/agent/ask",
                    json={"question": "What is our leave policy?"},
                    headers=headers,
                )
                self.assertIn(resp.status_code, (200, 500))  # May fail LLM or succeed, but not 429

            # 4th request must be rate-limited
            resp_limited = self.client.post(
                "/agent/ask",
                json={"question": "What is our leave policy?"},
                headers=headers,
            )
            self.assertEqual(resp_limited.status_code, 429)
            self.assertIn("Rate limit exceeded for agent queries", resp_limited.json().get("detail", ""))

    # 7. Knowledge Import Protection
    def test_knowledge_import_protection_in_production(self):
        """In production, /knowledge/import must reject unauthenticated imports."""
        sample_json = json.dumps({"company": {"name": "TestCorp"}}).encode("utf-8")
        file_payload = {"file": ("knowledge.json", io.BytesIO(sample_json), "application/json")}

        # 1. Blocked when ADMIN_TOKEN is missing in production mode
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "ADMIN_TOKEN": ""}):
            resp = self.client.post("/knowledge/import", files=file_payload)
            self.assertEqual(resp.status_code, 403)

        # 2. Blocked with 401 when ADMIN_TOKEN is set but header is missing/invalid
        with patch.dict(os.environ, {"ADMIN_TOKEN": "valid_secret_admin_token_12345"}):
            resp_no_header = self.client.post("/knowledge/import", files=file_payload)
            self.assertEqual(resp_no_header.status_code, 401)

            resp_wrong_token = self.client.post(
                "/knowledge/import",
                files={"file": ("knowledge.json", io.BytesIO(sample_json), "application/json")},
                headers={"X-Admin-Token": "wrong_token"},
            )
            self.assertEqual(resp_wrong_token.status_code, 401)


if __name__ == "__main__":
    unittest.main()
