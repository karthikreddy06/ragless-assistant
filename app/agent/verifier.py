import re
from typing import Any, Dict, List, Optional, Set

from app.agent.schemas import AgentPlan, CheckItem, ToolResult, Verification


class Verifier:
    """
    Deterministic Python verifier ensuring answer grounding and plan completeness.
    Executes ZERO LLM calls.
    """

    REFUSAL_MARKERS = [
        "could not find",
        "not found",
        "unavailable",
        "does not contain",
        "no information",
        "cannot find",
        "not available",
        "only contains",
    ]

    def verify(
        self,
        question: str,
        plan: AgentPlan,
        tool_results: List[ToolResult],
        answer: str,
    ) -> Verification:
        checks: List[CheckItem] = []
        unresolved: List[str] = []

        # 1. Check: All plan steps executed
        planned_steps_count = len(plan.steps)
        executed_count = len(tool_results)
        all_executed = (planned_steps_count == executed_count) and all(tr.success for tr in tool_results)

        if all_executed:
            checks.append(CheckItem(
                name="all_steps_executed",
                passed=True,
                detail=f"All {planned_steps_count} planned steps executed successfully.",
            ))
        else:
            checks.append(CheckItem(
                name="all_steps_executed",
                passed=False,
                detail=f"Executed {executed_count}/{planned_steps_count} steps. "
                       f"Failures: {[tr.error for tr in tool_results if not tr.success]}",
            ))
            unresolved.append("Some plan steps failed to execute.")

        # 2. Extract retrieved evidence facts
        extracted_facts: Set[str] = set()
        has_any_data = False

        for idx, tr in enumerate(tool_results):
            if tr.data:
                has_any_data = True
                self._collect_facts(tr.data, extracted_facts)
            else:
                step_input = plan.steps[idx].input if idx < len(plan.steps) else "step"
                unresolved.append(f"No evidence found in database for: '{step_input}'")

        # 3. Grounding Check: When evidence is completely empty
        ans_lower = answer.lower()
        if not has_any_data:
            has_refusal = any(marker in ans_lower for marker in self.REFUSAL_MARKERS)
            # Ensure model didn't hallucinate numbers when no evidence exists
            clean_ans = re.sub(r"(?:^|[\r\n\s])(?:\d+[\.\)]|\(\d+\))\s+", " ", answer)
            list_markers = set(re.findall(r"(?:^|[\r\n\s])(\d+)[\.\)]\s+", answer))
            list_markers.update(re.findall(r"(?:^|[\r\n\s])\((\d+)\)\s+", answer))
            numbers_in_answer = [n for n in re.findall(r"\b\d+\b", clean_ans) if n not in list_markers]
            no_hallucinated_numbers = len(numbers_in_answer) == 0

            passed_empty_grounding = has_refusal and no_hallucinated_numbers
            checks.append(CheckItem(
                name="grounded_refusal_when_empty",
                passed=passed_empty_grounding,
                detail="Appropriately acknowledged unavailable information without hallucination."
                if passed_empty_grounding else
                "Failed: Answer did not acknowledge missing evidence or hallucinated numbers.",
            ))
        else:
            # 4. Grounding Check: Numbers/facts in answer must appear in evidence or question
            # Exclude structural numbered-list markers (e.g. "1.", "2)", markdown list prefixes)
            clean_ans = re.sub(r"(?:^|[\r\n\s])(?:\d+[\.\)]|\(\d+\))\s+", " ", answer)
            list_markers = set(re.findall(r"(?:^|[\r\n\s])(\d+)[\.\)]\s+", answer))
            list_markers.update(re.findall(r"(?:^|[\r\n\s])\((\d+)\)\s+", answer))

            numbers_in_answer = set(re.findall(r"\b\d+\b", clean_ans)) - list_markers
            evidence_numbers = {f for f in extracted_facts if f.isdigit()}
            question_numbers = set(re.findall(r"\b\d+\b", question))
            allowed_numbers = evidence_numbers.union(question_numbers)

            unsupported_numbers = numbers_in_answer - allowed_numbers

            numbers_grounded = len(unsupported_numbers) == 0
            checks.append(CheckItem(
                name="evidence_fact_grounding",
                passed=numbers_grounded,
                detail="All numbers in answer appear in verified evidence."
                if numbers_grounded else
                f"Potentially ungrounded numbers found in answer: {sorted(unsupported_numbers)}",
            ))

        # 5. Check: Write confirmation for memory write operations
        for tr in tool_results:
            if tr.tool == "memory" and isinstance(tr.data, dict) and "write_confirmed" in tr.data:
                confirmed = bool(tr.data.get("write_confirmed"))
                op_name = tr.data.get("operation", "write")
                checks.append(CheckItem(
                    name="write_confirmed",
                    passed=confirmed,
                    detail=f"Memory write '{op_name}' confirmed via database read-back."
                    if confirmed else
                    f"Memory write '{op_name}' read-back confirmation failed.",
                ))
                if not confirmed:
                    unresolved.append(f"Write operation '{op_name}' failed read-back verification.")

        # 6. Check: No SQL or internal exposure in final answer
        sql_leaks = [
            ";", "--", "select ", "drop table", "insert into",
            "company_info", "departments", "leave_policy",
            "onboarding_items", "user_profile", "user_completed_items"
        ]
        has_sql_leak = any(leak in ans_lower for leak in sql_leaks)
        checks.append(CheckItem(
            name="security_no_internal_leak",
            passed=not has_sql_leak,
            detail="No internal SQL table names or database syntax leaked in answer."
            if not has_sql_leak else
            "Warning: Internal database table name or SQL syntax detected in answer.",
        ))

        # Overall verification outcome
        overall_passed = all(c.passed for c in checks)

        return Verification(
            passed=overall_passed,
            checks=checks,
            unresolved=unresolved,
        )

    def _collect_facts(self, data: Any, facts: Set[str]) -> None:
        """Recursively gathers primitive values from structured tool data."""
        if isinstance(data, dict):
            for v in data.values():
                self._collect_facts(v, facts)
        elif isinstance(data, list):
            for item in data:
                self._collect_facts(item, facts)
        elif data is not None:
            facts.add(str(data).strip())
