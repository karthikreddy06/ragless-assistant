"""Verification script for testing live Render deployment endpoints.

Tests against https://ragless-assistant.onrender.com:
- Health check
- User registration
- User login & session cookie/token
- Authenticated profile /auth/me (cookie + Bearer header)
- Authenticated tasks /tasks/me
- Agent /agent/ask (first-week deterministic onboarding query & graceful response)
- Logout and session invalidation
"""

import sys
import time
import uuid
import requests

LIVE_URL = "https://ragless-assistant.onrender.com"

def main():
    print(f"Testing live Render deployment at {LIVE_URL}...")
    session = requests.Session()

    # 1. Health check
    print("\n1. Testing GET /health...")
    r = session.get(f"{LIVE_URL}/health", timeout=15)
    print(f"   Status: {r.status_code}, Body: {r.json()}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    health_data = r.json()
    assert health_data.get("status") in ("ok", "ready"), f"Unexpected health status: {health_data}"

    # 2. Root check
    print("\n2. Testing GET / (API)...")
    r = session.get(f"{LIVE_URL}/", headers={"Accept": "application/json"}, timeout=15)
    print(f"   Status: {r.status_code}, Body: {r.json()}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    # 3. Register a test user
    test_id = uuid.uuid4().hex[:8]
    test_email = f"test_{test_id}@example.com"
    test_pass = "TestSecure123!"
    test_name = f"TestUser_{test_id}"

    print(f"\n3. Testing POST /auth/register ({test_email})...")
    reg_payload = {
        "email": test_email,
        "password": test_pass,
        "confirm_password": test_pass,
        "full_name": test_name,
    }
    r = session.post(f"{LIVE_URL}/auth/register", json=reg_payload, timeout=15)
    print(f"   Status: {r.status_code}, Body: {r.json()}")
    assert r.status_code in (200, 201), f"Expected 200/201, got {r.status_code}: {r.text}"
    user_info = r.json().get("user", {})
    assert user_info.get("email") == test_email

    # 4. Login with newly registered credentials
    print(f"\n4. Testing POST /auth/login for {test_email}...")
    login_payload = {
        "email": test_email,
        "password": test_pass,
    }
    r_login = session.post(f"{LIVE_URL}/auth/login", json=login_payload, timeout=15)
    print(f"   Login Status: {r_login.status_code}, Body: {r_login.json()}")
    assert r_login.status_code == 200
    session_id = r_login.json().get("session_id")
    assert session_id, "Expected non-empty session_id from /auth/login"
    print(f"   Received session_id: {session_id}")

    # 5. Auth Me with Session Cookie (from session cookie jar)
    print("\n5. Testing GET /auth/me (with session cookie)...")
    r_me_cookie = session.get(f"{LIVE_URL}/auth/me", timeout=15)
    print(f"   Status: {r_me_cookie.status_code}, Body: {r_me_cookie.json()}")
    assert r_me_cookie.status_code == 200
    assert r_me_cookie.json().get("email") == test_email

    # 6. Auth Me with Authorization Header (Bearer token)
    print("\n6. Testing GET /auth/me (with Authorization: Bearer <session_id> in fresh session)...")
    h_session = requests.Session()
    r_me_header = h_session.get(
        f"{LIVE_URL}/auth/me",
        headers={"Authorization": f"Bearer {session_id}"},
        timeout=15,
    )
    print(f"   Status: {r_me_header.status_code}, Body: {r_me_header.json()}")
    assert r_me_header.status_code == 200
    assert r_me_header.json().get("email") == test_email

    # 7. Tasks
    print("\n7. Testing GET /tasks/me...")
    r_tasks = session.get(f"{LIVE_URL}/tasks/me", timeout=15)
    print(f"   Status: {r_tasks.status_code}, Tasks: {r_tasks.json()}")
    assert r_tasks.status_code == 200
    assert isinstance(r_tasks.json().get("tasks"), list)

    # 8. Chat / Agent Ask (first-week onboarding deterministic fast path)
    print("\n8. Testing POST /agent/ask (first-week deterministic onboarding)...")
    chat_payload = {
        "question": "What tasks do I have scheduled for my first week?",
        "user_id": user_info.get("user_id"),
    }
    r_chat = session.post(f"{LIVE_URL}/agent/ask", json=chat_payload, timeout=30)
    print(f"   Status: {r_chat.status_code}")
    ans = r_chat.json().get("answer", "")
    print(f"   Answer preview: {ans[:200]}...")
    assert r_chat.status_code == 200
    assert "Day" in ans or "Day 1:" in ans or "tasks" in ans.lower()

    # 9. Test general query (when Ollama is unconfigured on Render)
    print("\n9. Testing POST /agent/ask (general query with remote Ollama unconfigured)...")
    r_gen = session.post(f"{LIVE_URL}/agent/ask", json={"question": "Write a python script", "user_id": user_info.get("user_id")}, timeout=30)
    print(f"   Status: {r_gen.status_code}, Answer: {r_gen.json().get('answer')}")
    assert r_gen.status_code == 200
    assert len(r_gen.json().get("answer", "")) > 0

    # 10. Logout and verify session revocation
    print("\n10. Testing POST /auth/logout and session revocation...")
    r_out = session.post(f"{LIVE_URL}/auth/logout", timeout=15)
    assert r_out.status_code == 200

    r_me_anon = session.get(f"{LIVE_URL}/auth/me", timeout=15)
    assert r_me_anon.status_code == 401, f"Expected 401 after logout, got {r_me_anon.status_code}"

    # Also verify that the bearer token is now revoked
    r_me_revoked = h_session.get(
        f"{LIVE_URL}/auth/me",
        headers={"Authorization": f"Bearer {session_id}"},
        timeout=15,
    )
    assert r_me_revoked.status_code == 401, f"Expected 401 for revoked token, got {r_me_revoked.status_code}"

    print("\n ALL LIVE BACKEND VERIFICATION CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
