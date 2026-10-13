import logging
from typing import Any, Callable, Dict, Optional

from app.agent.memory import (
    MemoryStore,
    get_memory_store,
    has_explicit_profile_statement,
    validate_item_completion_intent,
    validate_onboarding_status_intent,
    validate_profile_intent,
)
from app.agent.schemas import PlanStep, ToolResult
from app.agent.tasks import (
    TaskError,
    TaskLimitError,
    TaskStore,
    get_task_store,
    validate_complete_task_intent,
    validate_create_task_intent,
    validate_onboarding_tasks_intent,
)
from app.agent.web_search import TavilyClient, WebSearchClient, get_web_search_client
from app.query_engine import process_query

logger = logging.getLogger(__name__)

WEB_MAX_CALLS_PER_REQUEST = 2
MAX_TASK_WRITES_PER_REQUEST = 10


def execute_company_knowledge(query_input: str) -> ToolResult:
    """
    Executes a query against the existing RAGless query engine.
    Wraps the existing validated QueryOperation -> parameterized SQLite pipeline.
    Unsupported/missing facts (e.g. Engineering dept) return success=True with data=None.
    """
    try:
        from app.agent.fast_path import is_fast_path_enabled
        engine_result = process_query(query_input, generate_answer=False, allow_fast_path=is_fast_path_enabled())
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


def execute_web_research(
    step: PlanStep,
    client: Optional[WebSearchClient] = None,
    call_count: int = 0,
    max_calls: int = WEB_MAX_CALLS_PER_REQUEST,
) -> ToolResult:
    """
    Executes a web search query with per-request call limits.
    Extracts ONLY result titles, snippets, and URLs. Never crashes on missing key.
    Source label is 'tavily' only when a real TavilyClient is active; 'mock' for fake/offline clients.
    """
    search_client = client or get_web_search_client()
    if isinstance(search_client, TavilyClient):
        is_avail = search_client.is_available if isinstance(search_client.is_available, bool) else bool(search_client.is_available())
        src_label = "tavily" if is_avail else "unavailable"
    else:
        src_label = "mock"

    if call_count >= max_calls:
        logger.warning("Per-request web search limit (%d) exceeded.", max_calls)
        return ToolResult(
            tool="web_research",
            success=False,
            source=src_label,
            data=None,
            error=f"Per-request web search call limit of {max_calls} exceeded.",
        )

    args = step.args or {}
    query = str(args.get("query") or step.input or "").strip()
    if not query:
        return ToolResult(
            tool="web_research",
            success=False,
            source=src_label,
            data=None,
            error="web_research requires non-empty query.",
        )

    max_results = int(args.get("max_results", 3))

    try:
        res = search_client.search(query=query, max_results=max_results)
        if res.get("success"):
            return ToolResult(
                tool="web_research",
                success=True,
                source=src_label,
                data=res.get("results"),
                error=None,
            )
        else:
            return ToolResult(
                tool="web_research",
                success=False,
                source=src_label,
                data=None,
                error=res.get("error", "web research unavailable"),
            )
    except Exception as e:
        logger.exception("Unexpected error executing web_research for '%s': %s", query, e)
        return ToolResult(
            tool="web_research",
            success=False,
            source=src_label,
            data=None,
            error=f"Web search execution error: {str(e)}",
        )


def execute_memory(
    step: PlanStep,
    user_id: str,
    store: Optional[MemoryStore] = None,
    question: Optional[str] = None,
) -> ToolResult:
    """
    Executes a validated memory operation for the server-injected user_id.
    Enforces deterministic Python intent validation and read-back verification for all write operations.
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
            role_arg = args.get("role")
            dept_arg = args.get("department")
            if not validate_profile_intent(question, role=role_arg, department=dept_arg):
                logger.warning(
                    "Rejected set_profile: query '%s' lacks explicit role/department statement or verbatim value match (role=%s, dept=%s)",
                    question, role_arg, dept_arg,
                )
                return ToolResult(
                    tool="memory",
                    success=False,
                    source="sqlite",
                    data=None,
                    error="set_profile is only permitted when user text explicitly states role/department and the value appears verbatim in user text.",
                )

            res = mem_store.set_profile(
                user_id=user_id,
                role=role_arg,
                department=dept_arg,
                preferences=args.get("preferences"),
            )
            success = bool(res.get("success") and res.get("write_confirmed"))
            return ToolResult(
                tool="memory",
                success=success,
                source="sqlite",
                data=res if success else None,
                error=None if success else res.get("error", "Read-back verification or profile validation failed for set_profile."),
            )

        elif op == "mark_item_complete":
            try:
                item_id = int(args["item_id"])
            except (KeyError, ValueError, TypeError):
                return ToolResult(
                    tool="memory",
                    success=False,
                    source="sqlite",
                    data=None,
                    error="mark_item_complete requires integer item_id.",
                )

            if not validate_item_completion_intent(question, item_id=item_id):
                logger.warning(
                    "Rejected mark_item_complete for item %s: query '%s' lacks explicit completion intent or item match",
                    item_id, question,
                )
                return ToolResult(
                    tool="memory",
                    success=False,
                    source="sqlite",
                    data=None,
                    error=f"mark_item_complete is only permitted when user text explicitly states completion and matches item {item_id} by title keyword or ID.",
                )

            res = mem_store.mark_item_complete(user_id=user_id, item_id=item_id)
            success = bool(res.get("success") and res.get("write_confirmed"))
            return ToolResult(
                tool="memory",
                success=success,
                source="sqlite",
                data=res if success else None,
                error=None if success else res.get("error", "Read-back verification failed for mark_item_complete."),
            )

        elif op == "set_onboarding_status":
            target_status = str(args.get("status", "")).strip()
            if not validate_onboarding_status_intent(question, status=target_status):
                logger.warning(
                    "Rejected set_onboarding_status for status '%s': query '%s' lacks explicit onboarding status statement",
                    target_status, question,
                )
                return ToolResult(
                    tool="memory",
                    success=False,
                    source="sqlite",
                    data=None,
                    error=f"set_onboarding_status is only permitted when user text explicitly states starting or finishing onboarding (status='{target_status}').",
                )

            res = mem_store.set_onboarding_status(user_id=user_id, status=target_status)
            success = bool(res.get("success") and res.get("write_confirmed"))
            return ToolResult(
                tool="memory",
                success=success,
                source="sqlite",
                data=res if success else None,
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


def execute_tasks(
    step: PlanStep,
    user_id: str,
    task_store: Optional[TaskStore] = None,
    question: Optional[str] = None,
    run_context: Optional["RunContext"] = None,
) -> ToolResult:
    """
    Executes a validated tasks tool operation with parameterized SQLite calls,
    explicit-intent verification, per-request write limits, and immediate read-back verification.
    """
    store = task_store or get_task_store()
    op = (step.operation or "").strip().lower()
    args = step.args or {}
    ctx = run_context or RunContext()

    # Per-request write limit check for write operations
    if op in ("create_tasks_from_onboarding", "create_task", "complete_task"):
        if ctx.task_writes_count >= MAX_TASK_WRITES_PER_REQUEST:
            return ToolResult(
                tool="tasks",
                success=False,
                source="sqlite",
                data=None,
                error=f"Per-request task write limit of {MAX_TASK_WRITES_PER_REQUEST} writes exceeded.",
            )

    try:
        if op == "list_tasks":
            status_filter = args.get("status")
            res = store.list_tasks(user_id=user_id, status=status_filter)
            return ToolResult(
                tool="tasks",
                success=True,
                source="sqlite",
                data=res,
                error=None,
            )

        elif op == "create_tasks_from_onboarding":
            if not validate_onboarding_tasks_intent(question):
                logger.warning(
                    "Rejected create_tasks_from_onboarding for question '%s': lacks explicit creation intent",
                    question,
                )
                return ToolResult(
                    tool="tasks",
                    success=False,
                    source="sqlite",
                    data=None,
                    error="create_tasks_from_onboarding is only permitted when user text explicitly requests creating or setting up onboarding tasks.",
                )
            res = store.create_tasks_from_onboarding(user_id=user_id)
            writes = res.get("created_count", 0)
            ctx.task_writes_count += max(1, writes)
            return ToolResult(
                tool="tasks",
                success=True,
                source="sqlite",
                data=res,
                error=None,
            )

        elif op == "create_task":
            title = str(args.get("title", "")).strip()
            if not validate_create_task_intent(question, title):
                logger.warning(
                    "Rejected create_task with title '%s' for question '%s': lacks explicit intent or title match",
                    title, question,
                )
                return ToolResult(
                    tool="tasks",
                    success=False,
                    source="sqlite",
                    data=None,
                    error=f"create_task is only permitted when user explicitly requests adding a task and title '{title}' appears verbatim in user text.",
                )
            res = store.create_task(user_id=user_id, title=title)
            ctx.task_writes_count += 1
            return ToolResult(
                tool="tasks",
                success=True,
                source="sqlite",
                data=res,
                error=None,
            )

        elif op == "complete_task":
            task_id = int(args.get("task_id", 0))
            task_title = None
            try:
                conn = store.get_connection()
                cur = conn.cursor()
                cur.execute("SELECT title FROM tasks WHERE id = ? AND user_id = ?", (task_id, user_id))
                r = cur.fetchone()
                if r:
                    task_title = r[0]
                conn.close()
            except Exception:
                pass

            if not validate_complete_task_intent(question, task_id, task_title):
                logger.warning(
                    "Rejected complete_task #%d for question '%s': lacks explicit completion intent matching task",
                    task_id, question,
                )
                return ToolResult(
                    tool="tasks",
                    success=False,
                    source="sqlite",
                    data=None,
                    error=f"complete_task is only permitted when user explicitly requests completing task #{task_id}.",
                )
            res = store.complete_task(user_id=user_id, task_id=task_id)
            ctx.task_writes_count += 1
            return ToolResult(
                tool="tasks",
                success=True,
                source="sqlite",
                data=res,
                error=None,
            )

        else:
            return ToolResult(
                tool="tasks",
                success=False,
                source="sqlite",
                data=None,
                error=f"Unsupported tasks operation '{op}'.",
            )

    except (TaskLimitError, TaskError) as e:
        logger.warning("Task execution error: %s", e)
        return ToolResult(
            tool="tasks",
            success=False,
            source="sqlite",
            data=None,
            error=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error executing tasks operation '%s': %s", op, e)
        return ToolResult(
            tool="tasks",
            success=False,
            source="sqlite",
            data=None,
            error=f"Task execution error: {str(e)}",
        )


from contextvars import ContextVar

current_run_context: ContextVar[Optional["RunContext"]] = ContextVar("current_run_context", default=None)
current_active_client: ContextVar[Optional[Any]] = ContextVar("current_active_client", default=None)


class RunContext:
    """
    Per-run execution context ensuring request isolation.
    Tracks taint state, web call counts, LLM call counts, task writes count, and per-call latencies per request.
    """

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id
        self.is_tainted: bool = False
        self.web_calls_count: int = 0
        self.llm_calls_count: int = 0
        self.llm_latency_ms: list[float] = []
        self.task_writes_count: int = 0

    def record_llm_call(self, latency_ms: float = 0.0) -> None:
        self.llm_calls_count += 1
        self.llm_latency_ms.append(round(latency_ms, 2))


class ToolExecutor:
    """Registry mapping allowed tool names to deterministic Python functions with run-level security guards."""

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        task_store: Optional[TaskStore] = None,
        web_client: Optional[WebSearchClient] = None,
        web_search_client: Optional[WebSearchClient] = None,
        max_web_calls: int = WEB_MAX_CALLS_PER_REQUEST,
        db_path: Any = None,
    ):
        self._memory_store = memory_store or get_memory_store()
        self._task_store = task_store or get_task_store()
        self._web_client = web_client or web_search_client
        self.max_web_calls = max_web_calls
        self._default_context = RunContext()

    def reset_run(self):
        """Reset default fallback context."""
        self._default_context = RunContext()

    @property
    def is_tainted(self) -> bool:
        return self._default_context.is_tainted

    @property
    def web_calls_count(self) -> int:
        return self._default_context.web_calls_count

    def execute(
        self,
        step: PlanStep,
        user_id: str = "demo-user",
        question: Optional[str] = None,
        user_question: Optional[str] = None,
        run_context: Optional[RunContext] = None,
    ) -> ToolResult:
        """Executes a single validated PlanStep through the tool registry."""
        ctx = run_context if run_context is not None else self._default_context
        effective_question = question or user_question
        tool_name = step.tool.strip().lower()

        # TAINT RULE ENFORCEMENT:
        # Once web content has entered a run, memory and task write operations are strictly prohibited!
        if ctx.is_tainted:
            if tool_name == "memory":
                op = (step.operation or "").strip().lower()
                if op in ("set_profile", "mark_item_complete", "set_onboarding_status"):
                    logger.warning(
                        "Rejected memory write operation '%s': execution run is tainted by web research", op
                    )
                    return ToolResult(
                        tool="memory",
                        success=False,
                        source="sqlite",
                        data=None,
                        error="Security violation: memory write prohibited after web content has entered the execution run (taint rule).",
                    )
            elif tool_name == "tasks":
                op = (step.operation or "").strip().lower()
                if op in ("create_tasks_from_onboarding", "create_task", "complete_task"):
                    logger.warning(
                        "Rejected task write operation '%s': execution run is tainted by web research", op
                    )
                    return ToolResult(
                        tool="tasks",
                        success=False,
                        source="sqlite",
                        data=None,
                        error="Security violation: task write prohibited after web content has entered the execution run (taint rule).",
                    )

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
            return execute_memory(step, user_id, store=self._memory_store, question=effective_question)

        elif tool_name == "tasks":
            return execute_tasks(
                step,
                user_id=user_id,
                task_store=self._task_store,
                question=effective_question,
                run_context=ctx,
            )

        elif tool_name == "web_research":
            res = execute_web_research(
                step,
                client=self._web_client,
                call_count=ctx.web_calls_count,
                max_calls=self.max_web_calls,
            )
            ctx.web_calls_count += 1
            if res.success and res.data:
                ctx.is_tainted = True  # Taint the execution run
                self._default_context.is_tainted = True
            return res

        return ToolResult(
            tool=step.tool,
            success=False,
            source="unknown",
            data=None,
            error=f"Tool '{step.tool}' is not registered in ToolExecutor.",
        )

    execute_step = execute

