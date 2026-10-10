import logging
from fastapi import APIRouter, HTTPException, status

from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import AgentRequest, AgentResponse
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
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)
async def ask_agent(payload: AgentRequest) -> AgentResponse:
    try:
        orchestrator = get_orchestrator()
        response = orchestrator.run(payload.question, user_id=payload.user_id)
        return response
    except Exception as e:
        logger.exception("Unexpected error in agent /ask pipeline: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while executing the agent pipeline.",
        )
