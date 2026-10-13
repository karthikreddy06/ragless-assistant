import logging
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
    get_current_user,
    invalidate_user_session,
    register_user,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


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
async def login(req: LoginRequest, response: Response) -> Dict[str, Any]:
    user = authenticate_user(req.email, req.password)
    session_id = create_user_session(user.user_id)

    # Issue secure HttpOnly session cookie
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=False,  # Suitable for local HTTP development; browsers accept on localhost
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
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")
    return {"message": "Logged out successfully."}
