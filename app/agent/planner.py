import json
import logging
import re
from typing import Any, Dict

from app.agent.llm import LLMClient

logger = logging.getLogger(__name__)


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
        system_prompt = (
            "You are a strict task planner for an enterprise knowledge assistant.\n"
            "Your job is to break down a user question into sequential steps using ONLY allowed tools.\n\n"
            "Allowed Tools:\n"
            "- 'company_knowledge': Queries verified SQLite corporate knowledge (company details, department headcounts, leave policy, onboarding checklist).\n"
            "  Parameter: 'input' (concise natural-language sub-question or topic).\n"
            "- 'memory': Persistent structured user profile and onboarding progress tracker.\n"
            "  Parameter: 'operation' (one of: 'get_profile', 'get_progress', 'set_profile', 'mark_item_complete', 'set_onboarding_status')\n"
            "  Parameter: 'args' (structured object for operation parameters, e.g. {'item_id': 1} or {'department': 'Backend'}).\n\n"
            "When to use Memory operations:\n"
            "- 'get_progress': User asks what onboarding tasks are remaining, left to do, or asks for help completing onboarding.\n"
            "- 'get_profile': User asks for their own profile, role, or department.\n"
            "- 'set_profile': User explicitly specifies their role or department (args: {'role': '...', 'department': '...'}).\n"
            "- 'mark_item_complete': User explicitly states they completed or finished a checklist item (args: {'item_id': <integer>}).\n"
            "- 'set_onboarding_status': User explicitly requests changing their status (args: {'status': 'not_started'|'in_progress'|'completed'}).\n\n"
            "STRICT RULES:\n"
            "1. ONLY use 'company_knowledge' or 'memory'. Never invent tools (e.g. 'execute_sql', 'create_task', 'run_shell' are forbidden).\n"
            "2. NEVER write SQL queries, table names, or code in any parameter.\n"
            "3. NEVER specify 'user_id'. user_id is injected exclusively by server context and is strictly forbidden in plan steps.\n"
            "4. READ-ONLY INTENT SAFETY: Only use write operations ('set_profile', 'mark_item_complete', 'set_onboarding_status') when the user request explicitly instructs a change. If the user asks an informational question (e.g. 'what is security training?'), use 'company_knowledge' and do NOT write to memory.\n"
            "5. PRESERVE USER INTENT & ENTITIES: Faithfully retain the user's specific requested roles, entities, and questions in each step's input.\n"
            "6. Output valid JSON ONLY matching this structure:\n"
            "{\n"
            '  "goal": "Brief description of the objective",\n'
            '  "steps": [\n'
            '    {"tool": "company_knowledge", "input": "natural language sub-question"},\n'
            '    {"tool": "memory", "operation": "get_progress", "args": {}}\n'
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
