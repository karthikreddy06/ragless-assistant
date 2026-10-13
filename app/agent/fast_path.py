import logging
import os
import re
from typing import Optional

from app.agent.schemas import AgentPlan, PlanStep
from app.database import (
    get_all_department_names,
    get_all_leave_types,
    get_onboarding_items,
)

logger = logging.getLogger(__name__)

# Config toggle to enable or disable fast-path routing
FAST_PATH_ENABLED: bool = os.getenv("FAST_PATH_ENABLED", os.getenv("ENABLE_FAST_PATH", "true")).lower() in ("true", "1", "yes")
ENABLE_FAST_PATH: bool = FAST_PATH_ENABLED


def is_fast_path_enabled() -> bool:
    """Check dynamic runtime environment variable value."""
    val = os.getenv("FAST_PATH_ENABLED", os.getenv("ENABLE_FAST_PATH", "true"))
    return val.strip().lower() in ("true", "1", "yes")


def detect_fast_path_plan(question: str) -> Optional[AgentPlan]:
    """
    Deterministic fast-path router for clearly patterned requests.
    Bypasses LLM planning nondeterminism on small local models while still
    routing through the identical schema validation and executor pipeline.
    Returns an unvalidated AgentPlan (which must be validated by validate_plan),
    or None if no fast-path pattern matches.
    """
    if not is_fast_path_enabled():
        return None

    q_lower = question.strip().lower()

    # 1. Tasks operations (clearest phrasings)
    # (a) Complete task N
    task_complete_match = re.search(
        r"\b(?:complete|finish)\s+task\s+#?(\d+)\b|"
        r"\bmark\s+task\s+#?(\d+)\s+as\s+(?:done|complete)\b",
        q_lower,
    )
    if task_complete_match:
        task_id = int(task_complete_match.group(1) or task_complete_match.group(2))
        logger.info("Fast-path triggered: complete_task (id=%d) for '%s'", task_id, question)
        return AgentPlan(
            goal=f"Complete task #{task_id}",
            steps=[PlanStep(tool="tasks", operation="complete_task", args={"task_id": task_id})],
        )

    # (b) Create tasks from onboarding
    if re.search(r"\b(?:create|add|set\s*up)\s+tasks?\s+(?:for|from)\s+(?:my\s+)?(?:first-week\s+)?onboarding\b|\b(?:create|add|set\s*up)\s+(?:my\s+)?(?:first-week\s+)?onboarding\s+tasks?\b", q_lower):
        logger.info("Fast-path triggered: create_tasks_from_onboarding for '%s'", question)
        return AgentPlan(
            goal="Create tasks from onboarding checklist items",
            steps=[PlanStep(tool="tasks", operation="create_tasks_from_onboarding", args={})],
        )

    # (c) What tasks do I have
    if re.search(r"\b(?:what\s+tasks?\s+do\s+i\s+have|show\s+(?:my\s+)?tasks?|list\s+(?:my\s+)?tasks?|view\s+(?:my\s+)?tasks?)\b", q_lower):
        logger.info("Fast-path triggered: list_tasks for '%s'", question)
        return AgentPlan(
            goal="List user tasks",
            steps=[PlanStep(tool="tasks", operation="list_tasks", args={})],
        )

    # 2. Onboarding progress queries
    progress_patterns = [
        r"\bwhat\s+(?:do\s+i\s+still\s+need\s+to\s+(?:do|complete)|is\s+left\s+to\s+do|items\s+are\s+remaining)\b",
        r"\bwhat\s+do\s+i\s+need\s+to\s+do\s+next\b",
        r"\bhelp\s+me\s+complete\s+onboarding\b",
        r"\b(?:show|get|check|view)\s+(?:my\s+)?onboarding\s+progress\b",
        r"\b(?:what\s+is\s+my|check\s+my)\s+onboarding\s+status\b",
        r"\bhow\s+much\s+of\s+my\s+onboarding\s+is\s+(?:done|complete)\b",
        r"\bwhat\s+tasks?\s+are\s+left\b",
    ]
    for pat in progress_patterns:
        if re.search(pat, q_lower):
            logger.info("Fast-path triggered: get_progress for '%s'", question)
            return AgentPlan(
                goal="Retrieve user onboarding progress and remaining checklist items",
                steps=[PlanStep(tool="memory", operation="get_progress", args={})],
            )

    # 3. Mark item complete by ID or name
    # ID pattern: e.g. "mark item 1 complete", "mark item 1 as done", "completed item 2"
    id_match = re.search(r"\b(?:mark|completed?|finished?)\s+item\s*#?(\d+)(?:\s+(?:complete|done|as\s+done|as\s+complete))?\b", q_lower)
    if id_match:
        item_id = int(id_match.group(1))
        logger.info("Fast-path triggered: mark_item_complete (id=%d) for '%s'", item_id, question)
        return AgentPlan(
            goal=f"Mark onboarding item {item_id} as completed",
            steps=[PlanStep(tool="memory", operation="mark_item_complete", args={"item_id": item_id})],
        )

    # Named item pattern: e.g. "I completed security training", "finished HR documents"
    completion_phrases = ["completed", "finished", "mark", "done with"]
    if any(phrase in q_lower for phrase in completion_phrases):
        # Look up onboarding items in DB
        items = get_onboarding_items()
        for it in items:
            title_lower = it["title"].lower()
            # Check key keywords from title (e.g. "security training", "hr and tax", "email and slack")
            title_keywords = [w for w in title_lower.split() if len(w) > 3 and w not in ["complete", "setup", "review"]]
            if title_keywords and any(kw in q_lower for kw in title_keywords):
                # Verify user clearly states completion, not merely asking a read-only question
                if not any(q_word in q_lower for q_word in ["what is", "how do i", "explain", "where is", "tell me about"]):
                    logger.info("Fast-path triggered: mark_item_complete (id=%d, title='%s')", it["id"], it["title"])
                    return AgentPlan(
                        goal=f"Mark onboarding item {it['id']} ('{it['title']}') as completed",
                        steps=[PlanStep(tool="memory", operation="mark_item_complete", args={"item_id": it["id"]})],
                    )

    # 4. Profile queries
    profile_view_patterns = [
        r"\b(?:what\s+is\s+my|view\s+my|show\s+my|get\s+my)\s+profile\b",
        r"\bwho\s+am\s+i\b",
        r"\bwhat\s+is\s+my\s+(?:role|department)\b",
    ]
    for pat in profile_view_patterns:
        if re.search(pat, q_lower):
            logger.info("Fast-path triggered: get_profile for '%s'", question)
            return AgentPlan(
                goal="Retrieve user profile",
                steps=[PlanStep(tool="memory", operation="get_profile", args={})],
            )

    # 5. Profile update patterns: e.g. "set my department to Backend", "my role is Engineer"
    dept_match = re.search(r"\b(?:set\s+my\s+department\s+to|my\s+department\s+is)\s+([a-zA-Z\s]{2,30})\b", q_lower)
    role_match = re.search(r"\b(?:set\s+my\s+role\s+to|my\s+role\s+is)\s+([a-zA-Z\s]{2,30})\b", q_lower)
    if dept_match or role_match:
        dept = dept_match.group(1).strip().title() if dept_match else None
        role = role_match.group(1).strip().title() if role_match else None
        # Clean trailing punctuation
        if dept:
            dept = re.sub(r"[^\w\s]", "", dept).strip()
        if role:
            role = re.sub(r"[^\w\s]", "", role).strip()
        logger.info("Fast-path triggered: set_profile (role=%s, dept=%s)", role, dept)
        return AgentPlan(
            goal="Update user profile with specified role or department",
            steps=[PlanStep(tool="memory", operation="set_profile", args={"role": role, "department": dept})],
        )

    # 6. Company Knowledge Fast-Path (clear, unambiguous phrasings)
    # Questions with external comparison/benchmark phrasings or compound requests MUST fall through to LLM planner
    external_terms = [
        r"\b(?:compare|comparison|uk|statutory|industry|market|external|benchmark|vs|versus)\b"
    ]
    if any(re.search(pat, q_lower) for pat in external_terms):
        return None

    # Compound queries (joining multiple topics with conjunctions) must fall through to LLM planner
    if re.search(r"\b(?:and|also|plus|as\s+well\s+as)\b", q_lower):
        return None

    # (a) Company location
    if re.search(
        r"\b(?:where\s+is\s+(?:the\s+)?company(?:\s+located)?|company\s+location|where\s+are\s+we\s+located|what\s+is\s+(?:the\s+)?(?:company\s+)?(?:office\s+)?location|where\s+is\s+(?:the\s+)?office)\b",
        q_lower,
    ):
        logger.info("Fast-path triggered: company location for '%s'", question)
        return AgentPlan(
            goal="Retrieve company location",
            steps=[PlanStep(tool="company_knowledge", input=question)],
        )

    # (b) Company name
    if re.search(
        r"\b(?:what\s+is\s+(?:the\s+)?company\s+name|what\s+is\s+our\s+company\s+name|what\s+is\s+the\s+name\s+of\s+the\s+company|what\s+company\s+is\s+this|who\s+do\s+i\s+work\s+for)\b",
        q_lower,
    ):
        logger.info("Fast-path triggered: company name for '%s'", question)
        return AgentPlan(
            goal="Retrieve company name",
            steps=[PlanStep(tool="company_knowledge", input=question)],
        )

    # (c) "How many employees are in <known department>"
    # Matches against database department values dynamically
    try:
        known_depts = get_all_department_names()
    except Exception:
        known_depts = []

    matched_depts = [
        d for d in known_depts
        if re.search(rf"\b{re.escape(d.lower())}\b", q_lower)
    ]
    if len(matched_depts) == 1:
        dept_name = matched_depts[0]
        if re.search(
            r"\b(?:how\s+many\s+(?:people|employees)|headcount|number\s+of\s+employees|how\s+large\s+is|team\s+size|size\s+of\s+(?:the\s+)?|workforce|how\s+many\s+in)\b",
            q_lower,
        ):
            logger.info("Fast-path triggered: department headcount (%s) for '%s'", dept_name, question)
            return AgentPlan(
                goal=f"Retrieve headcount for {dept_name} department",
                steps=[PlanStep(tool="company_knowledge", input=question)],
            )

    # (d) Total employees
    if not matched_depts and re.search(
        r"\b(?:how\s+many\s+(?:total\s+)?employees(?:\s+are\s+there)?|total\s+employees|total\s+headcount|total\s+workforce|how\s+many\s+people\s+work\s+at\s+(?:the\s+)?company)\b",
        q_lower,
    ):
        logger.info("Fast-path triggered: total company employees for '%s'", question)
        return AgentPlan(
            goal="Retrieve total company headcount",
            steps=[PlanStep(tool="company_knowledge", input=question)],
        )

    # (e) Leave days for a known leave type
    # Matches against database leave types dynamically
    try:
        known_leaves = get_all_leave_types()
    except Exception:
        known_leaves = []

    matched_leaves = [
        lt for lt in known_leaves
        if re.search(rf"\b{re.escape(lt.lower())}\b", q_lower)
        or re.search(rf"\b{re.escape(lt.lower().replace(' leave', '').strip())}\b", q_lower)
    ]
    if len(matched_leaves) == 1:
        leave_name = matched_leaves[0]
        if re.search(
            r"\b(?:how\s+many\s+(?:days|leave\s+days)|days\s+of|leave\s+allowance|leave\s+days|how\s+much\s+leave|allowance)\b",
            q_lower,
        ):
            logger.info("Fast-path triggered: leave policy (%s) for '%s'", leave_name, question)
            return AgentPlan(
                goal=f"Retrieve leave policy allowance for {leave_name}",
                steps=[PlanStep(tool="company_knowledge", input=question)],
            )

    # (f) List onboarding items
    if re.search(
        r"\b(?:list\s+(?:the\s+)?onboarding\s+items|what\s+are\s+(?:the\s+)?onboarding\s+(?:checklist\s+)?items|show\s+(?:the\s+)?onboarding\s+items|view\s+(?:the\s+)?onboarding\s+items|onboarding\s+checklist\s+items)\b",
        q_lower,
    ):
        logger.info("Fast-path triggered: list onboarding items for '%s'", question)
        return AgentPlan(
            goal="List onboarding checklist items",
            steps=[PlanStep(tool="company_knowledge", input=question)],
        )

    return None
