import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile, status
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

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ragless-assistant")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for database initialization and startup checks."""
    admin_token = os.getenv("ADMIN_TOKEN")
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
        "A dynamic local AI knowledge assistant that uses Ollama (qwen2.5:3b) for natural-language "
        "understanding and answer grounding, with SQLite as the deterministic single source of truth. "
        "Supports dynamic JSON knowledge ingestion. No vector databases, embeddings, or semantic similarity search."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local development and UI integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Agent Router
app.include_router(agent_router)


@app.middleware("http")
async def add_security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/ui"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self';"
        )
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
async def root() -> RootResponse:
    return RootResponse(
        message="RAGless Knowledge Assistant API is running.",
        status="active",
        version="1.0.0",
        project="RAGless Knowledge Assistant",
        endpoints={
            "root": "/",
            "health": "/health",
            "ask": "/ask",
            "agent_ask": "/agent/ask",
            "memory": "/memory/{user_id}",
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
    description="Checks the health of the SQLite database connection and local Ollama instance.",
)
async def health() -> HealthResponse:
    db_healthy = check_database_health()
    ollama_info = check_ollama_health()

    from app.agent.llm import check_active_llm_health
    active_llm = check_active_llm_health()

    provider_available = ollama_info.get("available", False) if active_llm.get("provider") == "ollama" else active_llm.get("available", False)
    is_overall_ok = db_healthy and provider_available

    return HealthResponse(
        status="ok" if is_overall_ok else "degraded",
        database="connected" if db_healthy else "disconnected",
        ollama=ollama_info,
        llm_provider=active_llm.get("provider"),
        llm_model=active_llm.get("model"),
        llm_healthy=active_llm.get("available"),
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
    admin_token = os.getenv("ADMIN_TOKEN")
    if admin_token:
        if not x_admin_token or not hmac.compare_digest(x_admin_token.strip(), admin_token.strip()):
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
        503: {"model": ErrorResponse, "description": "LLM Service Unavailable"},
    },
)
async def ask(payload: AskRequest) -> AskResponse:
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
    "/memory/{user_id}",
    summary="Get User Memory and Onboarding Progress",
    description=(
        "Returns the stored profile and calculated onboarding progress for a given user_id. "
        "NOTE: user_id is a demonstration identifier; production multi-tenant authentication "
        "and authorization are out of scope. Do not rely on this endpoint for secure multi-tenancy."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid user_id format"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_user_memory(user_id: str):
    from app.agent.memory import MemoryError, get_memory_store, validate_user_id
    try:
        clean_uid = validate_user_id(user_id)
    except MemoryError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    store = get_memory_store()
    profile = store.get_profile(clean_uid)
    progress = store.get_progress(clean_uid)

    return {
        "user_id": clean_uid,
        "profile": profile,
        "progress": progress,
        "note": "Demo identifier only. Real multi-tenant authentication is out of scope.",
    }


@app.get(
    "/tasks/{user_id}",
    summary="Get User Tasks",
    description=(
        "Returns the list of tasks for a given user_id. "
        "NOTE: user_id is a demonstration identifier and there is no real authentication. "
        "Do not rely on this endpoint for secure multi-tenancy."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid user_id format"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def get_user_tasks(user_id: str, status: Optional[str] = None):
    from fastapi import status as http_status
    from app.agent.tasks import TaskError, get_task_store, validate_user_id
    try:
        clean_uid = validate_user_id(user_id)
    except TaskError as e:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    store = get_task_store()
    try:
        res = store.list_tasks(clean_uid, status=status)
        return {
            "user_id": clean_uid,
            "tasks": res["tasks"],
            "count": res["count"],
            "note": "user_id is a demo identifier; no real authentication.",
        }
    except Exception as e:
        logger.exception("Error retrieving tasks for user '%s': %s", clean_uid, e)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve tasks: {str(e)}",
        )