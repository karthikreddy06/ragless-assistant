import logging
import re
from typing import Any, Dict, List, Optional, Set
from pydantic import ValidationError

from app.agent.schemas import AgentPlan, PlanStep
from app.models import FORBIDDEN_SQL_PATTERNS

logger = logging.getLogger(__name__)

ALLOWED_TOOLS: Set[str] = {"company_knowledge", "memory", "web_research", "tasks"}
MAX_PLAN_STEPS: int = 8

# Extra malicious injection patterns (SQL injection, shell execution, prompt overrides)
EXTENDED_FORBIDDEN_PATTERNS = FORBIDDEN_SQL_PATTERNS + [
    r"'\s*or\b",
    r"\"\s*or\b",
    r"\bor\s+['\"]?1['\"]?\s*=\s*['\"]?1['\"]?",
    r"\b1\s*=\s*1\b",
    r"\bxp_\w+",
    r"\bscript\b",
    r"<script>",
    r"\bignore\s+(?:all\s+|any\s+)?(?:previous\s+|prior\s+)?instructions\b",
    r"\bdisregard\s+(?:all\s+|any\s+)?(?:previous\s+|prior\s+)?instructions\b",
]

EMAIL_REGEX = re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
KEY_PATTERN_REGEX = re.compile(
    r"\b(?:tvly-[a-zA-Z0-9_-]{10,}|sk-[a-zA-Z0-9_-]{10,}|api[_-]?key[_-]?[a-zA-Z0-9_-]{8,}|bearer\s+[a-zA-Z0-9_-]{10,})\b",
    re.IGNORECASE,
)

INTERNAL_DB_TERMS = [
    r"\buser_profile\b",
    r"\buser_completed_items\b",
    r"\bonboarding_items\b",
    r"\bleave_policy\b",
    r"\bdepartments?\s+table\b",
    r"\bsqlite_\w+\b",
]

COMPANY_SPECIFIC_TERMS = [
    r"\bthis\s+company\b",
    r"\bour\s+company\b",
    r"\b(?:the\s+company|company|our)\s+(?:policy|leave|vacation|holiday|sick\s+leave|annual\s+leave)\b",
    r"\b(?:the\s+company|company|our)\s+(?:sick\s+leave|annual\s+leave)\s+(?:allowance|policy|days)\b",
    r"\bmy\s+onboarding\b",
    r"\bonboarding\s+(?:checklist|item|items|status|progress|tasks?)\b",
    r"\bhead\s+of\b",
    r"\bdepartment\s+head\b",
    r"\bwho\s+is\s+the\s+head\s+of\b",
    r"\b(?:ceo|cto|cfo|executive|founder)\b",
    r"\bcompany\s+location\b",
    r"\bhow\s+many\s+employees\b",
    r"\btotal\s+employees\b",
]


EXTERNAL_PHRASE_PATTERNS = [
    r"\b(?:uk|us|eu|statutory|minimum|law|laws|legal|industry|market|benchmark|benchmarks|competitors?|external|standards?|global|federal|nationwide|market\s+rate)\b",
]


def extract_external_phrases(text: str) -> List[str]:
    """Extracts explicit external cues or phrases from text."""
    if not text:
        return []
    t_lower = text.lower()
    matches = []
    for pat in EXTERNAL_PHRASE_PATTERNS:
        for m in re.finditer(pat, t_lower):
            matches.append(m.group(0))
    return matches


class PlanValidationError(ValueError):
    """Raised when an unvalidated agent plan violates security or schema constraints."""
    pass


def is_company_specific_question(question: str) -> bool:
    """Determines if a question refers strictly to internal company specifics."""
    if not question:
        return False
    q_lower = question.lower()
    for pat in COMPANY_SPECIFIC_TERMS:
        if re.search(pat, q_lower):
            return True
    try:
        from app.database import get_current_company_name
        cname = get_current_company_name().lower()
        if cname and re.search(r"\b" + re.escape(cname) + r"\b", q_lower):
            return True
    except Exception:
        pass
    return False


def validate_outbound_web_query(
    query: str,
    user_id: Optional[str] = None,
    user_question: Optional[str] = None,
) -> None:
    """
    Strict outbound guard for web_research tool:
    - Must not contain user_id, emails, phone numbers, API keys, or SQL text.
    - Must not contain internal database table names or memory structures.
    - Must derive from the user's question (if user_question is provided).
    """
    if not query or not query.strip():
        raise PlanValidationError("Outbound guard: web_research query cannot be empty.")
    q_lower = query.strip().lower()

    # 1. user_id leakage prevention
    if user_id:
        clean_uid = user_id.strip().lower()
        if clean_uid and re.search(r"\b" + re.escape(clean_uid) + r"\b", q_lower):
            raise PlanValidationError(f"Outbound guard: web_research query must not contain user_id '{user_id}'.")

    # 2. Email address leakage
    if EMAIL_REGEX.search(query):
        raise PlanValidationError("Outbound guard: web_research query must not contain email addresses.")

    # 3. Phone number leakage
    if PHONE_REGEX.search(query):
        raise PlanValidationError("Outbound guard: web_research query must not contain phone numbers.")

    # 4. Sensitive API key or token leakage
    if KEY_PATTERN_REGEX.search(query):
        raise PlanValidationError("Outbound guard: web_research query must not contain API keys or tokens.")

    # 5. SQL injection / query syntax
    for pat in EXTENDED_FORBIDDEN_PATTERNS:
        if re.search(pat, q_lower):
            raise PlanValidationError(f"Outbound guard: web_research query contains forbidden SQL or injection syntax matching '{pat}'.")

    # 6. Internal database structure leakage
    for pat in INTERNAL_DB_TERMS:
        if re.search(pat, q_lower):
            raise PlanValidationError(f"Outbound guard: web_research query contains internal database reference '{pat}'.")

    # 7. Query must derive from user question (topic keyword overlap)
    if user_question:
        q_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", q_lower))
        stop_words = {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "can", "could", "should", "would", "will", "shall",
            "the", "and", "for", "with", "about", "from", "into", "through", "during",
            "before", "after", "above", "below", "to", "of", "in", "on", "at", "by",
            "this", "that", "these", "those", "current", "latest", "requirements", "search"
        }
        q_topics = q_words - stop_words
        u_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", user_question.lower())) - stop_words
        if q_topics and u_words and not (q_topics & u_words):
            raise PlanValidationError("Outbound guard: web_research query must derive from the user question.")


def validate_plan(
    raw_plan: Any,
    max_steps: int = MAX_PLAN_STEPS,
    user_id: Optional[str] = None,
    user_question: Optional[str] = None,
) -> AgentPlan:
    """
    Strict security boundary validating untrusted LLM-generated plans:
    1. Enforces schema with extra='forbid'.
    2. Enforces ALLOWED_TOOLS allowlist ({"company_knowledge", "memory", "web_research"}).
    3. Enforces step count limits (1 <= steps <= max_steps).
    4. Enforces length bounds on goal and step inputs.
    5. Rejects any SQL syntax, injection fragments, and unsafe delimiters.
    6. Strictly forbids LLM specifying user_id in any step or argument.
    7. Enforces outbound query guard and routing guard on web_research.
    """
    if not isinstance(raw_plan, dict):
        raise PlanValidationError(
            f"Expected plan to be a dictionary, got {type(raw_plan).__name__}."
        )

    # 1. Pydantic validation (enforces extra='forbid' and basic field types)
    try:
        plan = AgentPlan.model_validate(raw_plan)
    except ValidationError as e:
        first_err = e.errors()[0]
        field_loc = ".".join(str(loc) for loc in first_err.get("loc", []))
        msg = first_err.get("msg", "Invalid plan schema")
        raise PlanValidationError(f"Plan schema validation failed at '{field_loc}': {msg}") from e

    # 2. Step count boundary
    if len(plan.steps) == 0:
        raise PlanValidationError("Plan must contain at least 1 step.")
    if len(plan.steps) > max_steps:
        raise PlanValidationError(
            f"Plan exceeds maximum allowed steps ({len(plan.steps)} > {max_steps})."
        )

    # 3. Routing Guard: company-specific questions
    if user_question and is_company_specific_question(user_question):
        # Company-specific questions may NOT be planned as web_research-only
        if all(step.tool.strip().lower() == "web_research" for step in plan.steps):
            raise PlanValidationError(
                "Routing guard: internal company-specific questions cannot be planned as web_research-only."
            )

        # Extended routing guard: a company-specific question may not include a web_research step
        # whose query was not derived from an explicitly external phrase in the question.
        web_steps = [s for s in plan.steps if s.tool.strip().lower() == "web_research"]
        if web_steps:
            external_cues = extract_external_phrases(user_question)
            if not external_cues:
                raise PlanValidationError(
                    "Routing guard: company-specific question may not include a web_research step without an explicitly external phrase in the question."
                )
            for ws in web_steps:
                ws_args = ws.args or {}
                ws_query = str(ws_args.get("query") or ws.input or "").lower()
                if not any(cue in ws_query for cue in external_cues):
                    raise PlanValidationError(
                        "Routing guard: web_research step query in company-specific question must be derived from an explicitly external phrase in the question."
                    )

    # 4. Tool allowlist and security inspection per step
    for idx, step in enumerate(plan.steps, start=1):
        clean_tool = step.tool.strip().lower()
        if clean_tool not in ALLOWED_TOOLS:
            raise PlanValidationError(
                f"Step {idx} specifies disallowed tool '{step.tool}'. "
                f"Allowed tools: {sorted(ALLOWED_TOOLS)}."
            )

        if clean_tool == "company_knowledge":
            if not step.input:
                raise PlanValidationError(f"Step {idx} for company_knowledge requires non-empty input.")
            clean_input = step.input.strip().lower()
            for pat in EXTENDED_FORBIDDEN_PATTERNS:
                if re.search(pat, clean_input):
                    raise PlanValidationError(
                        f"Step {idx} input contains forbidden syntax or injection pattern matching '{pat}'."
                    )
        elif clean_tool in ("memory", "tasks"):
            args = step.args or {}
            if "user_id" in args:
                raise PlanValidationError(f"Step {idx} attempts to set 'user_id'. user_id cannot be specified by the planner.")
            def scan_for_sql(val: Any):
                if isinstance(val, str):
                    v_lower = val.lower()
                    for pat in EXTENDED_FORBIDDEN_PATTERNS:
                        if re.search(pat, v_lower):
                            raise PlanValidationError(
                                f"Step {idx} argument contains forbidden syntax matching '{pat}'."
                            )
                elif isinstance(val, dict):
                    for k, v in val.items():
                        scan_for_sql(k)
                        scan_for_sql(v)
                elif isinstance(val, list):
                    for item in val:
                        scan_for_sql(item)

            scan_for_sql(args)

            if clean_tool == "tasks" and user_question:
                op = (step.operation or "").strip().lower()
                if op == "create_tasks_from_onboarding":
                    from app.agent.tasks import validate_onboarding_tasks_intent
                    if not validate_onboarding_tasks_intent(user_question):
                        raise PlanValidationError(
                            f"Step {idx} attempts 'create_tasks_from_onboarding' without explicit user intent in question: '{user_question}'"
                        )
                elif op == "create_task":
                    from app.agent.tasks import validate_create_task_intent
                    task_title = str(args.get("title", "")).strip()
                    if not validate_create_task_intent(user_question, task_title):
                        raise PlanValidationError(
                            f"Step {idx} attempts 'create_task' without explicit user intent or verbatim title match: '{task_title}'"
                        )
                elif op == "complete_task":
                    from app.agent.tasks import validate_complete_task_intent
                    tid = args.get("task_id")
                    if tid is not None and not validate_complete_task_intent(user_question, task_id=int(tid)):
                        raise PlanValidationError(
                            f"Step {idx} attempts 'complete_task' without explicit completion intent for task #{tid}"
                        )

        elif clean_tool == "web_research":
            args = step.args or {}
            query = args.get("query") or step.input or ""
            validate_outbound_web_query(query, user_id=user_id, user_question=user_question)

    return plan
