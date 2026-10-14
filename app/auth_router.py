import logging
import os
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.auth import (
    AuthMeResponse,
    LoginRequest,
    RegisterRequest,
    SESSION_COOKIE_NAME,
    SESSION_DURATION_DAYS,
    User,
    UserPublicResponse,
    authenticate_user,
    create_user_session,
    extract_session_token,
    get_client_ip,
    get_current_user,
    invalidate_user_session,
    login_rate_limiter,
    register_user,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


def is_secure_cookie_required(request: Request) -> bool:
    """Determine whether the cookie Secure flag must be set (HTTPS or COOKIE_SECURE=true)."""
    is_https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    cookie_secure_env = os.getenv("COOKIE_SECURE", "").strip().lower() in ("true", "1", "yes")
    return is_https or cookie_secure_env


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register New User",
    description="Registers a new user account with Argon2id password hashing and unique email constraint.",
)
async def register(req: RegisterRequest) -> Dict[str, Any]:
    user = register_user(req)
    logger.info("New user registered successfully: %s (%s)", user.user_id, user.email)
    return {
        "message": "Registration successful.",
        "user": {
            "user_id": user.user_id,
            "email": user.email,
            "full_name": user.full_name,
        },
    }


@router.post(
    "/login",
    status_code=status.HTTP_200_OK,
    summary="User Login",
    description="Authenticates user credentials and establishes a secure session via HttpOnly cookie.",
)
async def login(req: LoginRequest, request: Request, response: Response) -> Dict[str, Any]:
    norm_email = req.email.strip().lower()
    client_ip = get_client_ip(request)

    email_key = f"email:{norm_email}"
    ip_key = f"ip:{client_ip}"

    # 1. Rate Limiting Check: per email and per IP
    is_locked, remaining = login_rate_limiter.is_locked_out(email_key)
    if not is_locked:
        is_locked, remaining = login_rate_limiter.is_locked_out(ip_key)

    if is_locked:
        logger.warning(
            "Login rejected due to rate limiting lockout for email '%s' from IP '%s' (%ds remaining)",
            norm_email,
            client_ip,
            remaining,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please try again later.",
        )

    # 2. Attempt authentication
    try:
        user = authenticate_user(norm_email, req.password)
    except HTTPException as e:
        login_rate_limiter.record_failure(email_key)
        login_rate_limiter.record_failure(ip_key)
        raise e

    # 3. Success: clear email failure history
    login_rate_limiter.record_success(email_key)

    session_id = create_user_session(user.user_id)
    secure_cookie = is_secure_cookie_required(request)

    # 4. Issue secure HttpOnly session cookie
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=secure_cookie,
        max_age=SESSION_DURATION_DAYS * 86400,
        path="/",
    )

    logger.info("User logged in successfully: %s", user.user_id)
    return {
        "message": "Login successful.",
        "user": {
            "user_id": user.user_id,
            "email": user.email,
            "full_name": user.full_name,
        },
        "session_id": session_id,
    }


@router.get(
    "/me",
    response_model=AuthMeResponse,
    summary="Get Current User Profile",
    description="Returns the profile and onboarding status of the currently authenticated user.",
)
async def get_me(current_user: User = Depends(get_current_user)) -> AuthMeResponse:
    from app.agent.memory import get_memory_store

    store = get_memory_store()
    prof = store.get_profile(current_user.user_id)

    return AuthMeResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=prof.get("role"),
        department=prof.get("department"),
        onboarding_status=prof.get("onboarding_status") or "not_started",
    )


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
    summary="User Logout",
    description="Invalidates the current session and clears the session cookie.",
)
async def logout(request: Request, response: Response) -> Dict[str, str]:
    token = extract_session_token(request)
    if token:
        invalidate_user_session(token)

    secure_cookie = is_secure_cookie_required(request)
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        secure=secure_cookie,
        httponly=True,
        samesite="lax",
    )
    return {"message": "Logged out successfully."}
