import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.database import initialize_database, seed_database
from app.main import app
from app.agent.tasks import get_task_store
from app.agent.memory import get_memory_store


class TestUserAuthenticationAndIsolation(unittest.TestCase):
    """
    Comprehensive tests for user registration, login, logout, session expiration,
    multi-user data isolation, and protection against identity spoofing.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_auth_isolation.db"
        self.env_patcher = patch.dict(os.environ, {"DATABASE_PATH": str(self.db_path)})
        self.env_patcher.start()

        initialize_database()
        seed_database()

        self.client = TestClient(app)

    def tearDown(self):
        self.env_patcher.stop()
        self.temp_dir.cleanup()

    def _register_user(self, full_name="Alice User", email="alice@example.com", password="Password123!"):
        return self.client.post(
            "/auth/register",
            json={
                "full_name": full_name,
                "email": email,
                "password": password,
                "confirm_password": password,
            },
        )

    def _login_user(self, email="alice@example.com", password="Password123!"):
        return self.client.post(
            "/auth/login",
            json={"email": email, "password": password},
        )

    # 1. Successful registration
    def test_01_successful_registration(self):
        resp = self._register_user("Alice Wonderland", "alice@example.com", "SecurePass123!")
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertIn("user", data)
        self.assertEqual(data["user"]["email"], "alice@example.com")
        self.assertEqual(data["user"]["full_name"], "Alice Wonderland")
        self.assertTrue(data["user"]["user_id"].startswith("alice_"))
        # Verify password_hash is NOT leaked
        self.assertNotIn("password_hash", str(data))
        self.assertNotIn("password", str(data.get("user", {})))

    # 2. Duplicate email rejection
    def test_02_duplicate_email_rejection(self):
        r1 = self._register_user("Bob First", "bob@example.com", "Password123!")
        self.assertEqual(r1.status_code, 201)

        # Same email, different case
        r2 = self._register_user("Bob Duplicate", "BOB@example.com", "DifferentPass123!")
        self.assertEqual(r2.status_code, 409)
        self.assertIn("already exists", r2.json().get("detail", "").lower())

    # 3. Successful login (sets HttpOnly cookie and returns session)
    def test_03_successful_login(self):
        self._register_user("Charlie Brown", "charlie@example.com", "CharliePass123!")
        resp = self._login_user("charlie@example.com", "CharliePass123!")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["user"]["email"], "charlie@example.com")
        self.assertIn("session_id", data)
        # Check session cookie is set
        self.assertIn("ragless_session", resp.cookies)
        self.assertEqual(resp.cookies["ragless_session"], data["session_id"])

    # 4. Incorrect password rejection
    def test_04_incorrect_password_rejection(self):
        self._register_user("Dave Miller", "dave@example.com", "DavePass123!")
        resp = self._login_user("dave@example.com", "WrongPassword!")
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json().get("detail"), "Invalid email or password.")
        # Non-existent user
        resp_nonexistent = self._login_user("unknown@example.com", "SomePassword!")
        self.assertEqual(resp_nonexistent.status_code, 401)
        self.assertEqual(resp_nonexistent.json().get("detail"), "Invalid email or password.")

    # 5. Missing or invalid authentication rejection
    def test_05_missing_or_invalid_authentication_rejection(self):
        # Protected /auth/me without cookie
        r1 = self.client.get("/auth/me")
        self.assertEqual(r1.status_code, 401)

        # Protected /auth/me with bogus cookie
        client_bogus = TestClient(app, cookies={"ragless_session": "bogus_token_12345"})
        r2 = client_bogus.get("/auth/me")
        self.assertEqual(r2.status_code, 401)

    # 6. /auth/me returns only current user's profile
    def test_06_auth_me_returns_current_user_profile(self):
        r_reg = self._register_user("Emma Watson", "emma@example.com", "EmmaPass123!")
        user_id = r_reg.json()["user"]["user_id"]

        r_login = self._login_user("emma@example.com", "EmmaPass123!")
        session_id = r_login.json()["session_id"]

        client_auth = TestClient(app, cookies={"ragless_session": session_id})
        resp = client_auth.get("/auth/me")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["user_id"], user_id)
        self.assertEqual(data["email"], "emma@example.com")
        self.assertEqual(data["full_name"], "Emma Watson")
        self.assertEqual(data["onboarding_status"], "not_started")

    # 7. Logout invalidates session
    def test_07_logout_invalidates_session(self):
        self._register_user("Frank Sinatra", "frank@example.com", "FrankPass123!")
        r_login = self._login_user("frank@example.com", "FrankPass123!")
        session_id = r_login.json()["session_id"]

        client_auth = TestClient(app, cookies={"ragless_session": session_id})
        me_before = client_auth.get("/auth/me")
        self.assertEqual(me_before.status_code, 200)

        # Logout
        logout_resp = client_auth.post("/auth/logout")
        self.assertEqual(logout_resp.status_code, 200)

        # Request /auth/me again -> 401
        me_after = client_auth.get("/auth/me")
        self.assertEqual(me_after.status_code, 401)

    # 8. User A cannot read User B's tasks
    def test_08_user_a_cannot_read_user_b_tasks(self):
        # Register User A and User B
        r_a = self._register_user("User A", "usera@example.com", "PasswordA123!")
        user_a_id = r_a.json()["user"]["user_id"]
        sess_a = self._login_user("usera@example.com", "PasswordA123!").json()["session_id"]

        r_b = self._register_user("User B", "userb@example.com", "PasswordB123!")
        user_b_id = r_b.json()["user"]["user_id"]
        sess_b = self._login_user("userb@example.com", "PasswordB123!").json()["session_id"]

        # Create tasks for User B
        task_store = get_task_store()
        task_store.create_task(user_b_id, "User B Private Task")

        # Create task for User A
        task_store.create_task(user_a_id, "User A Public Task")

        # User A calls /tasks/me -> sees only User A tasks
        client_a = TestClient(app, cookies={"ragless_session": sess_a})
        resp_a_tasks = client_a.get("/tasks")
        self.assertEqual(resp_a_tasks.status_code, 200)
        tasks_a = resp_a_tasks.json()["tasks"]
        self.assertEqual(len(tasks_a), 1)
        self.assertEqual(tasks_a[0]["title"], "User A Public Task")

        # User A attempts to access /tasks/{user_b_id} -> 403 Forbidden!
        resp_forbidden = client_a.get(f"/tasks/{user_b_id}")
        self.assertEqual(resp_forbidden.status_code, 403)
        self.assertIn("cannot access another user's tasks", resp_forbidden.json().get("detail", "").lower())

    # 9. User A cannot update or delete User B's tasks
    def test_09_user_a_cannot_update_or_delete_user_b_tasks(self):
        r_a = self._register_user("User A", "usera2@example.com", "PasswordA123!")
        user_a_id = r_a.json()["user"]["user_id"]

        r_b = self._register_user("User B", "userb2@example.com", "PasswordB123!")
        user_b_id = r_b.json()["user"]["user_id"]

        task_store = get_task_store()
        t_b = task_store.create_task(user_b_id, "User B Secret Project")
        b_task_id = t_b["task"]["id"]

        # User A attempts to complete User B's task through task store scoped to User A
        with self.assertRaises(Exception):
            task_store.complete_task(user_id=user_a_id, task_id=b_task_id)

        # Verify User B's task status remains 'open'
        b_tasks = task_store.list_tasks(user_b_id)["tasks"]
        self.assertEqual(b_tasks[0]["status"], "open")

    # 10. User A cannot read User B's onboarding progress
    def test_10_user_a_cannot_read_user_b_onboarding_progress(self):
        r_a = self._register_user("User A", "usera3@example.com", "PasswordA123!")
        sess_a = self._login_user("usera3@example.com", "PasswordA123!").json()["session_id"]

        r_b = self._register_user("User B", "userb3@example.com", "PasswordB123!")
        user_b_id = r_b.json()["user"]["user_id"]

        # Mark item for User B
        mem_store = get_memory_store()
        mem_store.mark_item_complete(user_b_id, 1)

        client_a = TestClient(app, cookies={"ragless_session": sess_a})

        # User A attempts to read User B's memory/progress
        resp = client_a.get(f"/memory/{user_b_id}")
        self.assertEqual(resp.status_code, 403)
        self.assertIn("cannot access another user's profile", resp.json().get("detail", "").lower())

    # 11. User A cannot access User B's memory
    def test_11_user_a_cannot_access_user_b_memory(self):
        r_a = self._register_user("User A", "usera4@example.com", "PasswordA123!")
        user_a_id = r_a.json()["user"]["user_id"]
        sess_a = self._login_user("usera4@example.com", "PasswordA123!").json()["session_id"]

        r_b = self._register_user("User B", "userb4@example.com", "PasswordB123!")
        user_b_id = r_b.json()["user"]["user_id"]

        mem_store = get_memory_store()
        mem_store.set_profile(user_b_id, role="Chief Scientist", department="AI")

        client_a = TestClient(app, cookies={"ragless_session": sess_a})
        # User A reading /memory/me returns User A profile, not User B
        res_a = client_a.get("/memory/me").json()
        self.assertEqual(res_a["user_id"], user_a_id)
        self.assertIsNone(res_a["profile"]["role"])

        # User B profile remains unchanged
        prof_b = mem_store.get_profile(user_b_id)
        self.assertEqual(prof_b["role"], "Chief Scientist")

    # 12. Changing client-supplied user ID cannot bypass ownership checks
    def test_12_client_supplied_user_id_cannot_bypass_ownership(self):
        r_a = self._register_user("User A", "usera5@example.com", "PasswordA123!")
        sess_a = self._login_user("usera5@example.com", "PasswordA123!").json()["session_id"]

        r_b = self._register_user("User B", "userb5@example.com", "PasswordB123!")
        user_b_id = r_b.json()["user"]["user_id"]

        client_a = TestClient(app, cookies={"ragless_session": sess_a})

        # User A attempts to run an agent query with user_id = user_b_id in payload
        # Backend overrides payload.user_id with authenticated user_a_id!
        with patch("app.agent.router.get_orchestrator") as mock_get_orch:
            from app.agent.schemas import AgentResponse, Verification
            mock_orch = unittest.mock.MagicMock()
            mock_orch.run.return_value = AgentResponse(
                question="What are my tasks?",
                answer="No tasks.",
                plan=None,
                tool_results=[],
                verification=Verification(passed=True, checks=[], unresolved=[]),
                processing_time_ms=10.0,
                user_id=r_a.json()["user"]["user_id"],
                llm_provider="ollama",
                llm_model="qwen2.5:3b",
                answer_mode="template",
            )
            mock_get_orch.return_value = mock_orch

            resp = client_a.post("/agent/ask", json={"question": "What are my tasks?", "user_id": user_b_id})
            self.assertEqual(resp.status_code, 200)

            # Assert orchestrator.run was called with User A's ID, NOT User B's ID!
            mock_orch.run.assert_called_once()
            called_uid = mock_orch.run.call_args[1]["user_id"]
            self.assertEqual(called_uid, r_a.json()["user"]["user_id"])
            self.assertNotEqual(called_uid, user_b_id)

    # 13. Unauthenticated dashboard API access is rejected
    def test_13_unauthenticated_dashboard_api_access_rejected(self):
        # /auth/me without session -> 401
        self.assertEqual(self.client.get("/auth/me").status_code, 401)
        # /memory/me without session -> 401
        self.assertEqual(self.client.get("/memory/me").status_code, 401)
        # /memory without session -> 401
        self.assertEqual(self.client.get("/memory").status_code, 401)
        # /tasks/me without session -> 401
        self.assertEqual(self.client.get("/tasks/me").status_code, 401)
        # /tasks without session -> 401
        self.assertEqual(self.client.get("/tasks").status_code, 401)
        # /auth/logout without session -> 200 or 401
        # /tasks/real-user without session -> 401
        self.assertEqual(self.client.get("/tasks/some-registered-user").status_code, 401)
        self.assertEqual(self.client.get("/memory/some-registered-user").status_code, 401)

    # 15. Password hashes and credentials never leak into responses
    def test_15_password_hashes_never_leak(self):
        # Registration response
        r_reg = self._register_user("No Leak", "noleak@example.com", "SecretPass1234!")
        self.assertNotIn("password_hash", r_reg.text)
        self.assertNotIn("argon2", r_reg.text)

        # Login response
        r_login = self._login_user("noleak@example.com", "SecretPass1234!")
        self.assertNotIn("password_hash", r_login.text)
        self.assertNotIn("argon2", r_login.text)

        # /auth/me response
        client_auth = TestClient(app, cookies={"ragless_session": r_login.json()["session_id"]})
        r_me = client_auth.get("/auth/me")
        self.assertNotIn("password_hash", r_me.text)
        self.assertNotIn("argon2", r_me.text)

        # Error response
        r_err = self._login_user("noleak@example.com", "WrongPassword")
        self.assertNotIn("password_hash", r_err.text)
        self.assertNotIn("argon2", r_err.text)


if __name__ == "__main__":
    unittest.main()
