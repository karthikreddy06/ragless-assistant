import logging
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import sqlite3

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError
from fastapi import Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.database import get_connection

logger = logging.getLogger(__name__)

# Password hasher configuration (Argon2id default in argon2-cffi)
ph = PasswordHasher()

SESSION_COOKIE_NAME = "ragless_session"
SESSION_DURATION_DAYS = 7
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USER_ID_CLEAN_REGEX = re.compile(r"[^a-zA-Z0-9_\-]")


class User(BaseModel):
    """Authenticated user representation (never contains password hash)."""
    model_config = ConfigDict(extra="forbid")

    id: int
    user_id: str
    email: str
    full_name: str
    created_at: Optional[str] = None


class RegisterRequest(BaseModel):
    """Payload for user registration."""
    model_config = ConfigDict(extra="forbid")

    full_name: str = Field(..., min_length=1, max_length=100, description="User full name")
    email: str = Field(..., min_length=3, max_length=120, description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="Password (min 8 chars)")
    confirm_password: str = Field(..., min_length=8, max_length=128, description="Password confirmation")

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Full name cannot be empty.")
        return s

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        s = v.strip().lower()
        if not EMAIL_REGEX.match(s):
            raise ValueError("Invalid email address format.")
        return s


class LoginRequest(BaseModel):
    """Payload for user login."""
    model_config = ConfigDict(extra="forbid")

    email: str = Field(..., min_length=3, max_length=120)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        s = v.strip().lower()
        if not EMAIL_REGEX.match(s):
            raise ValueError("Invalid email address format.")
        return s


class UserPublicResponse(BaseModel):
    """Safe user details (no hashes, secrets, or internal IDs)."""
    model_config = ConfigDict(extra="forbid")

    user_id: str
    email: str
    full_name: str


class AuthMeResponse(BaseModel):
    """Full authenticated profile returned by /auth/me."""
    model_config = ConfigDict(extra="forbid")

    user_id: str
    email: str
    full_name: str
    role: Optional[str] = None
    department: Optional[str] = None
    onboarding_status: str = "not_started"


def generate_unique_user_id(email: str, connection: sqlite3.Connection) -> str:
    """Generate a clean, unique alphanumeric user_id from email prefix + hex suffix."""
    prefix = email.split("@")[0]
    clean_prefix = USER_ID_CLEAN_REGEX.sub("", prefix).lower()[:20]
    if not clean_prefix:
        clean_prefix = "user"

    cursor = connection.cursor()
    for _ in range(10):
        suffix = secrets.token_hex(4)
        candidate = f"{clean_prefix}_{suffix}"
        cursor.execute("SELECT 1 FROM users WHERE user_id = ?", (candidate,))
        if not cursor.fetchone():
            return candidate
    return f"u_{secrets.token_hex(8)}"


def register_user(req: RegisterRequest) -> User:
    """
    Registers a new user:
    - Verifies password confirmation
    - Normalizes email
    - Checks for duplicate email (rejects with 409)
    - Hashes password using Argon2id
    - Atomically creates user and default user_profile records
    """
    if req.password != req.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password and confirm password do not match.",
        )

    norm_email = req.email.strip().lower()
    norm_name = req.full_name.strip()

    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            # 1. Check duplicate email
            cursor.execute("SELECT id FROM users WHERE email = ?", (norm_email,))
            if cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="An account with this email address already exists.",
                )

            # 2. Hash password with Argon2id
            password_hash = ph.hash(req.password)

            # 3. Generate unique user_id
            user_id = generate_unique_user_id(norm_email, conn)

            # 4. Insert into users
            cursor.execute(
                """
                INSERT INTO users (user_id, email, full_name, password_hash)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, norm_email, norm_name, password_hash),
            )
            new_id = cursor.lastrowid

            # 5. Initialize user_profile record
            cursor.execute(
                """
                INSERT OR IGNORE INTO user_profile (user_id, role, department, onboarding_status)
                VALUES (?, NULL, NULL, 'not_started')
                """,
                (user_id,),
            )

            return User(
                id=new_id,
                user_id=user_id,
                email=norm_email,
                full_name=norm_name,
            )
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> User:
    """
    Authenticates user credentials.
    Returns User if valid; raises HTTPException 401 with generic error on failure.
    """
    norm_email = email.strip().lower()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, user_id, email, full_name, password_hash, created_at FROM users WHERE email = ?",
            (norm_email,),
        )
        row = cursor.fetchone()
        if not row:
            # Perform dummy hash verification to mitigate timing attacks
            try:
                ph.verify("$argon2id$v=19$m=65536,t=3,p=4$dummy$dummy", password)
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )

        stored_hash = row["password_hash"]
        try:
            ph.verify(stored_hash, password)
        except (VerifyMismatchError, VerificationError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )

        return User(
            id=row["id"],
            user_id=row["user_id"],
            email=row["email"],
            full_name=row["full_name"],
            created_at=str(row["created_at"]),
        )
    finally:
        conn.close()


def create_user_session(user_id: str, days: int = SESSION_DURATION_DAYS) -> str:
    """Creates a secure cryptographically random session and stores it in user_sessions."""
    session_id = secrets.token_urlsafe(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_sessions (session_id, user_id, expires_at)
                VALUES (?, ?, ?)
                """,
                (session_id, user_id, expires_at),
            )
        return session_id
    finally:
        conn.close()


def invalidate_user_session(session_id: str) -> None:
    """Deletes a session from the database (logout/revocation)."""
    if not session_id:
        return
    conn = get_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_sessions WHERE session_id = ?", (session_id,))
    finally:
        conn.close()


def get_user_from_session(session_id: str) -> Optional[User]:
    """Retrieves the authenticated User associated with an active, unexpired session."""
    if not session_id or not isinstance(session_id, str):
        return None

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT u.id, u.user_id, u.email, u.full_name, u.created_at
            FROM user_sessions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.session_id = ? AND s.expires_at > ?
            """,
            (session_id, now_str),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return User(
            id=row["id"],
            user_id=row["user_id"],
            email=row["email"],
            full_name=row["full_name"],
            created_at=str(row["created_at"]),
        )
    finally:
        conn.close()


def extract_session_token(request: Request) -> Optional[str]:
    """Extract session token from HttpOnly cookie or Authorization Bearer header."""
    # 1. Cookie
    cookie_token = request.cookies.get(SESSION_COOKIE_NAME)
    if cookie_token:
        return cookie_token.strip()

    # 2. Authorization Header (Bearer token fallback for API clients)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token:
            return token

    return None


def get_current_user(request: Request) -> User:
    """FastAPI dependency: Enforces valid user authentication. Raises 401 if missing or invalid."""
    token = extract_session_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
        )

    user = get_user_from_session(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired or is invalid. Please log in again.",
        )
    return user


def get_optional_current_user(request: Request) -> Optional[User]:
    """FastAPI dependency: Returns User if authenticated; None if unauthenticated."""
    token = extract_session_token(request)
    if not token:
        return None
    return get_user_from_session(token)
