import logging
from typing import Any, Callable, Dict

from app.agent.memory import MemoryStore, get_memory_store
from app.agent.schemas import PlanStep, ToolResult
from app.query_engine import process_query

logger = logging.getLogger(__name__)


def execute_company_knowledge(query_input: str) -> ToolResult:
    """
    Executes a query against the existing RAGless query engine.
    Wraps the existing validated QueryOperation -> parameterized SQLite pipeline.
    Unsupported/missing facts (e.g. Engineering dept) return success=True with data=None.
    """
    try:
        engine_result = process_query(query_input)
        retrieved_data = engine_result.get("retrieved_data")

        return ToolResult(
            tool="company_knowledge",
            success=True,
            source=engine_result.get("source", "sqlite"),
            data=retrieved_data,
            error=None,
        )
    except Exception as e:
        logger.exception("Error executing company_knowledge tool for '%s': %s", query_input, e)
        return ToolResult(
            tool="company_knowledge",
            success=False,
            source="sqlite",
            data=None,
            error=f"Execution error: {str(e)}",
        )


def execute_memory(
    step: PlanStep,
    user_id: str,
    store: Optional[MemoryStore] = None,
) -> ToolResult:
    """
    Executes a validated memory operation for the server-injected user_id.
    Enforces read-back verification for all write operations.
    """
    mem_store = store or get_memory_store()
    op = (step.operation or "").strip().lower()
    args = step.args or {}

    try:
        if op == "get_profile":
            data = mem_store.get_profile(user_id)
            return ToolResult(tool="memory", success=True, source="sqlite", data=data)

        elif op == "get_progress":
            data = mem_store.get_progress(user_id)
            return ToolResult(tool="memory", success=True, source="sqlite", data=data)

        elif op == "set_profile":
            res = mem_store.set_profile(
                user_id=user_id,
                role=args.get("role"),
                department=args.get("department"),
                preferences=args.get("preferences"),
            )
            success = bool(res.get("success") and res.get("write_confirmed"))
            return ToolResult(
                tool="memory",
                success=success,
                source="sqlite",
                data=res,
                error=None if success else "Read-back verification failed for set_profile.",
            )

        elif op == "mark_item_complete":
            item_id = int(args["item_id"])
            res = mem_store.mark_item_complete(user_id=user_id, item_id=item_id)
            success = bool(res.get("success") and res.get("write_confirmed"))
            return ToolResult(
                tool="memory",
                success=success,
                source="sqlite",
                data=res,
                error=None if success else res.get("error", "Read-back verification failed for mark_item_complete."),
            )

        elif op == "set_onboarding_status":
            res = mem_store.set_onboarding_status(user_id=user_id, status=str(args["status"]))
            success = bool(res.get("success") and res.get("write_confirmed"))
            return ToolResult(
                tool="memory",
                success=success,
                source="sqlite",
                data=res,
                error=None if success else "Read-back verification failed for set_onboarding_status.",
            )

        else:
            return ToolResult(
                tool="memory",
                success=False,
                source="sqlite",
                data=None,
                error=f"Unsupported memory operation '{step.operation}'.",
            )
    except Exception as e:
        logger.exception("Error executing memory operation '%s' for user '%s': %s", op, user_id, e)
        return ToolResult(
            tool="memory",
            success=False,
            source="sqlite",
            data=None,
            error=f"Memory execution error: {str(e)}",
        )


class ToolExecutor:
    """Registry mapping allowed tool names to deterministic Python functions."""

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self._memory_store = memory_store or get_memory_store()

    def execute(self, step: PlanStep, user_id: str = "demo-user") -> ToolResult:
        """Executes a single validated PlanStep through the tool registry."""
        tool_name = step.tool.strip().lower()

        if tool_name == "company_knowledge":
            if not step.input:
                return ToolResult(
                    tool="company_knowledge",
                    success=False,
                    source="sqlite",
                    data=None,
                    error="company_knowledge step requires 'input'.",
                )
            return execute_company_knowledge(step.input)

        elif tool_name == "memory":
            return execute_memory(step, user_id, store=self._memory_store)

        return ToolResult(
            tool=step.tool,
            success=False,
            source="unknown",
            data=None,
            error=f"Tool '{step.tool}' is not registered in ToolExecutor.",
        )
