import logging
from fastapi import APIRouter, HTTPException, Request, status

from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import AgentRequest, AgentResponse
from app.auth import agent_rate_limiter, get_client_ip, get_optional_current_user
from app.models import ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])

# Single shared orchestrator instance
_orchestrator: AgentOrchestrator = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


@router.post(
    "/ask",
    response_model=AgentResponse,
    summary="Ask RAGless Agent",
    description=(
        "Processes a question using the RAGless Agent loop: "
        "understand -> plan -> validate -> execute -> verify -> respond. "
        "All queries execute through verified parameterized SQLite retrieval."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Authentication Required"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
def ask_agent(payload: AgentRequest, request: Request) -> AgentResponse:
    current_user = get_optional_current_user(request)
    if current_user:
        # Authenticated user: ALWAYS bind agent execution strictly to authenticated identity.
        # Ignores/overrides any client-supplied payload.user_id to prevent identity spoofing.
        effective_user_id = current_user.user_id
    else:
        # Unauthenticated: If caller attempts to specify an identity other than demo-user, reject!
        if payload.user_id and payload.user_id != "demo-user":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required to run agent tasks for registered user accounts.",
            )
        effective_user_id = "demo-user"

    # Enforce agent query rate limiting (per-user for authenticated, per-IP for demo users)
    rate_key = f"user:{effective_user_id}" if effective_user_id != "demo-user" else f"ip:{get_client_ip(request)}"
    is_limited, retry_after = agent_rate_limiter.check_and_record(rate_key)
    if is_limited:
        logger.warning("Agent rate limit exceeded for %s (%ds remaining)", rate_key, retry_after)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for agent queries. Please try again in {retry_after} seconds.",
        )

    try:
        orchestrator = get_orchestrator()
        response = orchestrator.run(payload.question, user_id=effective_user_id)
        return response
    except Exception as e:
        logger.exception("Unexpected error in agent /ask pipeline: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while executing the agent pipeline.",
        )
