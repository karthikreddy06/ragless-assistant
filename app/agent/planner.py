import json
import logging
import re
from typing import Any, Dict

from app.agent.llm import LLMClient
from app.database import get_onboarding_items

logger = logging.getLogger(__name__)


def _build_checklist_catalog_prompt() -> str:
    """Build dynamic catalog of onboarding items and IDs for the planner."""
    try:
        items = get_onboarding_items()
        if items:
            lines = "\n".join(f"  - Item ID {it['id']}: {it['title']}" for it in items)
            return (
                "\nAvailable Onboarding Checklist Items Catalog:\n"
                f"{lines}\n"
                "When user states they completed or finished a checklist item, use 'mark_item_complete' "
                "with the matching integer 'item_id' from this catalog.\n"
            )
    except Exception:
        pass
    return ""


class PlanParsingError(ValueError):
    """Raised when the LLM response cannot be parsed into a JSON plan."""
    pass


class AgentPlanner:
    """
    Plans multi-step execution for user queries using an LLMClient.
    The planner only proposes a plan; it NEVER executes tools and NEVER produces SQL.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def plan(self, question: str) -> Dict[str, Any]:
        """
        Generates an unvalidated plan dictionary from the LLM.
        Strips markdown code fences and parses JSON strictly.
        """
        catalog_section = _build_checklist_catalog_prompt()

        system_prompt = (
            "You are a strict task planner for an enterprise knowledge assistant.\n"
            "Your job is to break down a user question into sequential steps using ONLY allowed tools.\n\n"
            "Allowed Tools:\n"
            "- 'company_knowledge': Queries verified SQLite corporate knowledge (company details, department headcounts, leave policy, onboarding checklist).\n"
            "  Parameter: 'input' (concise natural-language sub-question or topic).\n"
            "- 'memory': Persistent structured user profile and onboarding progress tracker.\n"
            "  Parameter: 'operation' (one of: 'get_profile', 'get_progress', 'set_profile', 'mark_item_complete', 'set_onboarding_status')\n"
            "  Parameter: 'args' (structured object for operation parameters, e.g. {'item_id': 1} or {'department': 'Backend'}).\n"
            "- 'tasks': Actionable task management for user tasks and onboarding workflow.\n"
            "  Parameter: 'operation' (one of: 'create_tasks_from_onboarding', 'create_task', 'list_tasks', 'complete_task')\n"
            "  Parameter: 'args' (structured object for operation parameters, e.g. {} for create_tasks_from_onboarding/list_tasks, {'title': '...'} for create_task, {'task_id': <int>} for complete_task).\n"
            "- 'web_research': Searches external public web for general technology, industry benchmarks, or outside technical documentation.\n"
            "  Parameter: 'args': {'query': '<concise search query derived from user question>', 'max_results': 3}.\n\n"
            f"{catalog_section}"
            "When to use Tasks operations:\n"
            "- 'create_tasks_from_onboarding': User explicitly asks to create, add, or set up tasks for their onboarding or first week (args: {}).\n"
            "- 'create_task': User explicitly asks to create or add a new task with a specific title (args: {'title': '...'}).\n"
            "- 'list_tasks': User asks what tasks they have, or to show/list their tasks (args: {} or {'status': 'open'|'done'}).\n"
            "- 'complete_task': User explicitly asks to complete, finish, or mark done a task (args: {'task_id': <int>}).\n\n"
            "When to use Memory operations:\n"
            "- 'get_progress': User asks what onboarding items are remaining, left to do, or asks for help completing onboarding.\n"
            "- 'get_profile': User asks for their own profile, role, or department.\n"
            "- 'set_profile': User explicitly specifies their role or department (args: {'role': '...', 'department': '...'}).\n"
            "- 'mark_item_complete': User explicitly states they completed or finished a checklist item (args: {'item_id': <integer ID from catalog>}).\n"
            "- 'set_onboarding_status': User explicitly requests changing their status (args: {'status': 'not_started'|'in_progress'|'completed'}).\n\n"
            "When to use Web Research vs Company Knowledge:\n"
            "- Use 'web_research' ONLY for external, industry, general technical documentation, or public knowledge (e.g. 'What are NVIDIA Nemotron deployment requirements?').\n"
            "- NEVER use 'web_research' for internal company policies, company departments, leave days, or onboarding. Internal company questions (e.g. 'Who is the head of Engineering?', 'our leave policy') MUST route to 'company_knowledge' or 'memory', NEVER 'web_research'.\n"
            "- FOR MIXED QUESTIONS that compare internal company policy with an external public benchmark or statutory standard, generate a multi-step plan containing BOTH tools: first 'company_knowledge' for the internal data, and second 'web_research' for the external benchmark.\n"
            "  Example Mixed Question: \"How does our annual leave compare to UK statutory minimum?\"\n"
            "  Example Plan:\n"
            "  {\n"
            '    "goal": "Compare internal annual leave policy with UK statutory minimum",\n'
            '    "steps": [\n'
            '      {"tool": "company_knowledge", "input": "How many days of annual leave does the company provide?"},\n'
            '      {"tool": "web_research", "args": {"query": "UK statutory minimum annual leave entitlement", "max_results": 1}}\n'
            '    ]\n'
            "  }\n\n"
            "  Example Task Creation Question: \"Create tasks for my onboarding\"\n"
            "  Example Plan (notice tool is 'tasks' with 'operation'):\n"
            "  {\n"
            '    "goal": "Create onboarding tasks for user",\n'
            '    "steps": [\n'
            '      {"tool": "tasks", "operation": "create_tasks_from_onboarding", "args": {}}\n'
            '    ]\n'
            "  }\n\n"
            "STRICT RULES:\n"
            "1. ONLY use 'company_knowledge', 'memory', 'tasks', or 'web_research'. Never invent tools (e.g. 'execute_sql', 'run_shell', 'delete_task' are forbidden).\n"
            "2. NEVER write SQL queries, table names, or code in any parameter.\n"
            "3. NEVER specify 'user_id'. user_id is injected exclusively by server context and is strictly forbidden in plan steps.\n"
            "4. READ-ONLY INTENT SAFETY: Only use write operations ('set_profile', 'mark_item_complete', 'set_onboarding_status', 'create_tasks_from_onboarding', 'create_task', 'complete_task') when the user request explicitly instructs a change. If the user asks an informational question (e.g. 'what is a task?', 'what is security training?'), use 'company_knowledge' or read-only operations and do NOT write.\n"
            "5. PRESERVE USER INTENT & ENTITIES: Faithfully retain the user's specific requested roles, entities, and questions in each step's input.\n"
            "6. Output valid JSON ONLY matching this structure:\n"
            "{\n"
            '  "goal": "Brief description of the objective",\n'
            '  "steps": [\n'
            '    {"tool": "company_knowledge", "input": "natural language sub-question"},\n'
            '    {"tool": "web_research", "args": {"query": "external search query", "max_results": 3}}\n'
            "  ]\n"
            "}"
        )

        user_prompt = f'User Question: "{question}"\n\nGenerate the plan in JSON:'

        raw_response = self.llm.generate_json(
            prompt=user_prompt,
            system=system_prompt,
            temperature=0.0,
        )

        return self._extract_json(raw_response)

    def _extract_json(self, raw_text: str) -> Dict[str, Any]:
        """Extracts and parses JSON object from LLM response text."""
        cleaned = raw_text.strip()

        # Strip markdown code blocks if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # Attempt direct JSON parse
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback regex extraction of outer JSON braces
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError as e:
                raise PlanParsingError(
                    f"LLM produced malformed JSON inside braces: {e}. Raw: {raw_text[:200]}"
                ) from e

        raise PlanParsingError(
            f"Failed to parse valid JSON plan from LLM response. Raw response: {raw_text[:200]}"
        )
