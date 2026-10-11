import logging
import os
import re
from typing import Optional

from app.agent.schemas import AgentPlan, PlanStep
from app.database import get_onboarding_items

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

    # 3. Profile queries
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

    # 4. Profile update patterns: e.g. "set my department to Backend", "my role is Engineer"
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

    return None
