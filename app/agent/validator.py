import logging
import re
from typing import Any, Dict, Set
from pydantic import ValidationError

from app.agent.schemas import AgentPlan, PlanStep
from app.models import FORBIDDEN_SQL_PATTERNS

logger = logging.getLogger(__name__)

ALLOWED_TOOLS: Set[str] = {"company_knowledge", "memory"}
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


class PlanValidationError(ValueError):
    """Raised when an unvalidated agent plan violates security or schema constraints."""
    pass


def validate_plan(raw_plan: Any, max_steps: int = MAX_PLAN_STEPS) -> AgentPlan:
    """
    Strict security boundary validating untrusted LLM-generated plans:
    1. Enforces schema with extra='forbid'.
    2. Enforces ALLOWED_TOOLS allowlist ({"company_knowledge", "memory"}).
    3. Enforces step count limits (1 <= steps <= max_steps).
    4. Enforces length bounds on goal and step inputs.
    5. Rejects any SQL syntax, injection fragments, and unsafe delimiters.
    6. Strictly forbids LLM specifying user_id in any step or argument.
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

    # 3. Tool allowlist and security inspection per step
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
        elif clean_tool == "memory":
            args = step.args or {}
            # Reject user_id in args
            if "user_id" in args:
                raise PlanValidationError(f"Step {idx} attempts to set 'user_id'. user_id cannot be specified by the planner.")
            # Scan all string values in args for forbidden patterns
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

    return plan
