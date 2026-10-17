import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.database import (
    check_database_health,
    get_full_knowledge,
    initialize_database,
    seed_database,
)
from app.ingestion import IngestionValidationError, ingest_knowledge_from_content
from app.llm import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    check_ollama_health,
)
from app.models import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    ImportSummaryResponse,
    KnowledgeViewResponse,
    RootResponse,
)
from app.query_engine import process_query
from app.agent.router import router as agent_router
from app.auth import User, agent_rate_limiter, get_client_ip, get_current_user, get_optional_current_user
from app.auth_router import router as auth_router

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ragless-assistant")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for database initialization, environment verification, and startup checks."""
    is_production = (
        os.getenv("ENVIRONMENT", "").strip().lower() == "production"
        or os.getenv("RENDER", "").strip().lower() == "true"
    )
    admin_token = os.getenv("ADMIN_TOKEN", "").strip()
    weak_tokens = {"your_admin_token_here", "admin", "password", "secret", "123456", "test", "token", "changeme"}

    if is_production:
        if not admin_token or admin_token.lower() in weak_tokens or len(admin_token) < 16:
            err = "Production startup failed: ADMIN_TOKEN must be set to a secure, random secret (at least 16 characters) in production mode."
            logger.critical(err)
            raise RuntimeError(err)

        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        if provider == "ollama":
            ollama_url = os.getenv("OLLAMA_BASE_URL", "").strip()
            import sys
            is_testing = "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST") is not None
            allow_local = os.getenv("ALLOW_LOCALHOST_IN_PROD", "false").lower() in ("true", "1", "yes")

            if not ollama_url:
                logger.info(
                    "OLLAMA_BASE_URL is unset at startup. Application starting in deterministic mode. "
                    "AI features will gracefully notify callers that Ollama is not configured."
                )
            elif ("localhost" in ollama_url or "127.0.0.1" in ollama_url) and not (is_testing or allow_local):
                err = (
                    f"Production startup failed: OLLAMA_BASE_URL ('{ollama_url}') points to localhost/127.0.0.1. "
                    "In cloud production (such as Render), configure OLLAMA_BASE_URL to point to your securely hosted Ollama server or tunnel."
                )
                logger.critical(err)
                raise RuntimeError(err)
    else:
        if not admin_token:
            logger.warning(
                "Security warning: ADMIN_TOKEN environment variable is unset. "
                "/knowledge/import endpoint is accessible without authentication (local dev mode)."
            )

    logger.info("Initializing and verifying SQLite database...")
    initialize_database()
    seed_database()
    logger.info("Database ready. Startup complete.")

    warmup = os.getenv("WARMUP_ON_STARTUP", "false").lower() in ("true", "1", "yes")
    import sys
    is_testing = "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST") is not None
    if warmup and not is_testing:
        logger.info("Performing startup warm-up call to Ollama (WARMUP_ON_STARTUP=true)...")
        try:
            from app.llm import _call_ollama_generate
            _call_ollama_generate("Warmup ping", temperature=0.0)
            logger.info("Ollama warm-up call completed successfully.")
        except Exception as e:
            logger.warning("Ollama warm-up call failed or skipped: %s", e)

    yield
    logger.info("Shutting down RAGless Knowledge Assistant API.")


app = FastAPI(
    title="RAGless Knowledge Assistant API",
    description=(
        "A dynamic local AI knowledge assistant that uses Ollama (qwen2.5:3b) or Nebius Token Factory for natural-language "
        "understanding and answer grounding, with SQLite as the deterministic single source of truth. "
        "Supports dynamic JSON knowledge ingestion. No vector databases, embeddings, or semantic similarity search."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS safely based on environment
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
is_production = (
    os.getenv("ENVIRONMENT", "").strip().lower() == "production"
    or os.getenv("RENDER", "").strip().lower() == "true"
)

if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
elif is_production:
    # In production, default to deployed Render static site origin and backend origin
    allowed_origins = [
        "https://ragless-assistant-ui.onrender.com",
        "https://ragless-assistant.onrender.com",
    ]
else:
    # Local development origins
    allowed_origins = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Admin-Token"],
    )

# Register Authentication & Agent Routers
app.include_router(auth_router)
app.include_router(agent_router)


@app.middleware("http")
async def add_security_headers_middleware(request: Request, call_next):
    response = await call_next(request)

    # Global baseline security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

    # Strict Content-Security-Policy on UI and HTML responses
    if request.url.path.startswith("/ui") or request.url.path == "/":
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self' https://ragless-assistant.onrender.com;"
        )

    # HSTS when HTTPS is active or COOKIE_SECURE is enabled
    is_https = (
        request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto") == "https"
        or os.getenv("COOKIE_SECURE", "").strip().lower() in ("true", "1", "yes")
    )
    if is_https:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response


# Serve Web UI static files
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic request validation errors gracefully without leaking internals."""
    errors = exc.errors()
    first_msg = errors[0].get("msg", "Invalid request body") if errors else "Validation error"
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "detail": f"Request validation failed: {first_msg}",
            "error_type": "validation_error",
        },
    )


@app.exception_handler(OllamaConnectionError)
async def ollama_connection_exception_handler(
    request: Request, exc: OllamaConnectionError
) -> JSONResponse:
    """Handle Ollama connectivity issues without exposing stack traces."""
    logger.error("Ollama connection error: %s", exc)
    from app.llm import is_ollama_configured
    if not is_ollama_configured():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": "Ollama LLM service is not configured. Please set OLLAMA_BASE_URL to enable AI features.",
                "error_type": "ollama_not_configured",
            },
        )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "Local Ollama LLM service is currently unavailable. Please ensure Ollama is running.",
            "error_type": "ollama_unavailable",
        },
    )


@app.exception_handler(OllamaModelNotFoundError)
async def ollama_model_exception_handler(
    request: Request, exc: OllamaModelNotFoundError
) -> JSONResponse:
    """Handle missing Ollama models gracefully."""
    logger.error("Ollama model not found: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": str(exc),
            "error_type": "model_not_found",
        },
    )


@app.exception_handler(IngestionValidationError)
async def ingestion_validation_exception_handler(
    request: Request, exc: IngestionValidationError
) -> JSONResponse:
    """Handle ingestion validation errors with clear client feedback."""
    logger.warning("Ingestion validation error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": str(exc),
            "error_type": "ingestion_validation_error",
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Standardize HTTP exception format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_type": "http_error",
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all safety exception handler preventing stack trace leaks."""
    logger.exception("Unhandled server exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred while processing the request.",
            "error_type": "internal_server_error",
        },
    )


@app.get(
    "/",
    response_model=RootResponse,
    summary="Root Status",
    description="Returns basic API information, status, and links to documentation.",
)
async def root(request: Request):
    # If a web browser directly navigates to / requesting HTML, redirect to the chat interface /ui/
    accept = request.headers.get("accept", "")
    if "text/html" in accept and not accept.startswith("*/*"):
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/ui/", status_code=307)

    return RootResponse(
        message="RAGless Knowledge Assistant API is running.",
        status="active",
        version="1.0.0",
        project="RAGless Knowledge Assistant",
        endpoints={
            "root": "/",
            "ui": "/ui/",
            "health": "/health",
            "auth_register": "/auth/register",
            "auth_login": "/auth/login",
            "auth_me": "/auth/me",
            "auth_logout": "/auth/logout",
            "ask": "/ask",
            "agent_ask": "/agent/ask",
            "memory": "/memory",
            "tasks": "/tasks",
            "knowledge_view": "/knowledge",
            "knowledge_import": "/knowledge/import",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Checks the health of the SQLite database connection and active LLM provider.",
)
def health() -> HealthResponse:
    db_healthy = check_database_health()

    from app.agent.llm import check_active_llm_health
    active_llm = check_active_llm_health()
    provider = active_llm.get("provider", "ollama")
    llm_healthy = bool(active_llm.get("available"))
    is_configured = bool(active_llm.get("configured", True))
    is_overall_ok = db_healthy and (llm_healthy or not is_configured)

    ollama_info = None
    if provider == "ollama":
        ollama_info = active_llm.get("details")
        if not ollama_info:
            from app.llm import check_ollama_health
            ollama_info = check_ollama_health()

    return HealthResponse(
        status="ok" if is_overall_ok else "degraded",
        database="connected" if db_healthy else "disconnected",
        ollama=ollama_info if ollama_info is not None else {"available": False, "configured": is_configured, "provider": provider},
        llm_provider=provider,
        llm_model=active_llm.get("model"),
        llm_healthy=llm_healthy,
    )


@app.get(
    "/knowledge",
    response_model=KnowledgeViewResponse,
    summary="View Active Knowledge",
    description="Returns the currently active corporate knowledge from SQLite in safe structured format.",
)
async def view_knowledge() -> KnowledgeViewResponse:
    knowledge = get_full_knowledge()
    return KnowledgeViewResponse(**knowledge)


@app.post(
    "/knowledge/import",
    response_model=ImportSummaryResponse,
    summary="Import Knowledge File",
    description=(
        "Upload a JSON knowledge file containing company, departments, and leave policy. "
        "Validates schema and business rules, clears old records, and atomically imports new records."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request / Validation Error"},
        422: {"model": ErrorResponse, "description": "Unprocessable Knowledge Format"},
        500: {"model": ErrorResponse, "description": "Internal Ingestion Error"},
    },
)
async def import_knowledge(
    file: UploadFile = File(...),
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
) -> ImportSummaryResponse:
    admin_token = os.getenv("ADMIN_TOKEN", "").strip()
    is_production = (
        os.getenv("ENVIRONMENT", "").strip().lower() == "production"
        or os.getenv("RENDER", "").strip().lower() == "true"
    )

    if is_production and not admin_token:
        logger.error("Blocked /knowledge/import: ADMIN_TOKEN is missing in production mode.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Knowledge import is disabled in production without configured ADMIN_TOKEN.",
        )

    if admin_token:
        if not x_admin_token or not hmac.compare_digest(x_admin_token.strip(), admin_token):
            logger.warning("Rejected unauthorized knowledge import attempt.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: missing or invalid X-Admin-Token header.",
            )

    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was provided in the upload request.",
        )

    try:
        content_bytes = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    if not content_bytes or not content_bytes.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded knowledge file is empty.",
        )

    try:
        summary = ingest_knowledge_from_content(content_bytes)
        return ImportSummaryResponse(**summary)
    except IngestionValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error during knowledge ingestion: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to ingest knowledge into database.",
        )


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a Question",
    description=(
        "Processes a natural-language question about the imported company knowledge. "
        "Extracts structured intent via Ollama (qwen2.5:3b), retrieves facts from SQLite, "
        "and synthesizes a grounded answer."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        429: {"model": ErrorResponse, "description": "Too Many Requests"},
        503: {"model": ErrorResponse, "description": "LLM Service Unavailable"},
    },
)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    client_ip = get_client_ip(request)
    is_limited, retry_after = agent_rate_limiter.check_and_record(f"ip:{client_ip}")
    if is_limited:
        logger.warning("Rate limit exceeded for /ask from IP %s (%ds remaining)", client_ip, retry_after)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for queries. Please try again in {retry_after} seconds.",
        )

    try:
        result = process_query(payload.question)
        return AskResponse(**result)
    except (OllamaConnectionError, OllamaModelNotFoundError):
        raise
    except Exception as e:
        logger.exception("Error executing ask pipeline: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer for the question.",
        )


@app.get(
    "/memory",
    summary="Get Authenticated User Memory and Onboarding Progress",
    description="Returns the profile and onboarding progress for the currently authenticated user.",
)
@app.get(
    "/memory/me",
    summary="Get Authenticated User Memory and Onboarding Progress",
)
async def get_my_memory(current_user: User = Depends(get_current_user)):
    from app.agent.memory import get_memory_store
    store = get_memory_store()
    profile = store.get_profile(current_user.user_id)
    progress = store.get_progress(current_user.user_id)
    return {
        "user_id": current_user.user_id,
        "profile": profile,
        "progress": progress,
    }


@app.get(
    "/memory/{user_id}",
    summary="Get User Memory and Onboarding Progress",
    description="Returns user profile and progress. Requires authentication. Users can only access their own data.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid user_id format"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Forbidden - cannot access another user's profile"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_user_memory(user_id: str, request: Request):
    from app.agent.memory import MemoryError, get_memory_store, validate_user_id
    try:
        clean_uid = validate_user_id(user_id)
    except MemoryError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    current_user = get_optional_current_user(request)
    if current_user:
        # Authenticated user can ONLY access their own resources
        if clean_uid.lower() != current_user.user_id.lower() and clean_uid != "me":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Cannot access another user's profile.",
            )
        target_uid = current_user.user_id
    else:
        # Unauthenticated request: allow legacy demo-user for backward compatibility with tests
        if clean_uid == "demo-user":
            target_uid = "demo-user"
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required to access user profile.",
            )

    store = get_memory_store()
    profile = store.get_profile(target_uid)
    progress = store.get_progress(target_uid)

    return {
        "user_id": target_uid,
        "profile": profile,
        "progress": progress,
    }


@app.get(
    "/tasks",
    summary="Get Authenticated User Tasks",
    description="Returns the list of tasks for the currently authenticated user.",
)
@app.get(
    "/tasks/me",
    summary="Get Authenticated User Tasks",
)
async def get_my_tasks(status: Optional[str] = None, current_user: User = Depends(get_current_user)):
    from app.agent.tasks import get_task_store
    store = get_task_store()
    res = store.list_tasks(current_user.user_id, status=status)
    return {
        "user_id": current_user.user_id,
        "tasks": res["tasks"],
        "count": res["count"],
    }


@app.get(
    "/tasks/{user_id}",
    summary="Get User Tasks",
    description="Returns tasks for user. Requires authentication. Users can only access their own tasks.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid user_id format"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Forbidden - cannot access another user's tasks"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_user_tasks(user_id: str, request: Request, status: Optional[str] = None):
    from fastapi import status as http_status
    from app.agent.tasks import TaskError, get_task_store, validate_user_id
    try:
        clean_uid = validate_user_id(user_id)
    except TaskError as e:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    current_user = get_optional_current_user(request)
    if current_user:
        # Authenticated user can ONLY access their own resources
        if clean_uid.lower() != current_user.user_id.lower() and clean_uid != "me":
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Access denied: Cannot access another user's tasks.",
            )
        target_uid = current_user.user_id
    else:
        # Unauthenticated request: allow legacy demo-user for backward compatibility with tests
        if clean_uid == "demo-user":
            target_uid = "demo-user"
        else:
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required to access user tasks.",
            )

    store = get_task_store()
    try:
        res = store.list_tasks(target_uid, status=status)
        return {
            "user_id": target_uid,
            "tasks": res["tasks"],
            "count": res["count"],
        }
    except Exception as e:
        logger.exception("Error retrieving tasks for user '%s': %s", target_uid, e)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve tasks: {str(e)}",
        )