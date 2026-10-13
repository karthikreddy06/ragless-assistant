import re
from typing import Any, Dict, List, Optional, Set, Tuple

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
                step = plan.steps[idx] if plan and idx < len(plan.steps) else None
                if step is not None and step.tool == "web_research":
                    unresolved.append("External research was unavailable.")
                else:
                    step_input = step.input if (step and step.input) else (step.operation if (step and step.operation) else "requested query")
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
            # 4. Grounding Check: Numbers/facts in answer must appear in evidence or question,
            # or be derivable as a difference, absolute difference, sum, or percentage of two numbers
            # that BOTH appear in the evidence.
            clean_ans = re.sub(r"(?:^|[\r\n\s])(?:\d+[\.\)]|\(\d+\))\s+", " ", answer)
            list_markers = set(re.findall(r"(?:^|[\r\n\s])(\d+)[\.\)]\s+", answer))
            list_markers.update(re.findall(r"(?:^|[\r\n\s])\((\d+)\)\s+", answer))

            raw_tokens = re.findall(r"\b\d+(?:\.\d+)?\b", clean_ans)
            numbers_in_answer = [n for n in raw_tokens if n not in list_markers]

            # Extract numbers from evidence (facts)
            evidence_numeric_values: Set[float] = set()
            for f in extracted_facts:
                for m in re.findall(r"\b\d+(?:\.\d+)?\b", str(f)):
                    try:
                        evidence_numeric_values.add(float(m))
                    except ValueError:
                        pass

            # Extract numbers from question
            question_numeric_values: Set[float] = set()
            for m in re.findall(r"\b\d+(?:\.\d+)?\b", question):
                try:
                    question_numeric_values.add(float(m))
                except ValueError:
                    pass

            unsupported_numbers = []
            accepted_derived: Dict[str, str] = {}

            # Extract numeric values from answer for operand presence verification
            answer_numeric_values: Set[float] = set()
            for n in numbers_in_answer:
                try:
                    answer_numeric_values.add(float(n))
                except ValueError:
                    pass

            for num_str in set(numbers_in_answer):
                try:
                    val = float(num_str)
                except ValueError:
                    unsupported_numbers.append(num_str)
                    continue

                # 1. Directly in evidence or question
                if val in evidence_numeric_values or val in question_numeric_values:
                    continue

                # 2. Derivable from two numbers that BOTH appear in evidence AND in the answer text
                derivation = self.find_derived_number_explanation(
                    val, evidence_numeric_values, answer_numbers=answer_numeric_values
                )
                if derivation:
                    accepted_derived[num_str] = derivation
                else:
                    unsupported_numbers.append(num_str)

            numbers_grounded = len(unsupported_numbers) == 0

            if numbers_grounded:
                if accepted_derived:
                    derived_text = ", ".join(accepted_derived.values())
                    detail = f"All numbers in answer appear in verified evidence (accepted derived: {derived_text})."
                else:
                    detail = "All numbers in answer appear in verified evidence."
            else:
                detail = f"Potentially ungrounded numbers found in answer: {sorted(unsupported_numbers)}"
                if accepted_derived:
                    derived_text = ", ".join(accepted_derived.values())
                    detail += f" (accepted derived: {derived_text})"

            checks.append(CheckItem(
                name="evidence_fact_grounding",
                passed=numbers_grounded,
                detail=detail,
            ))

        # 5. Check: Write confirmation for memory and task write operations
        for tr in tool_results:
            if tr.tool in ("memory", "tasks") and isinstance(tr.data, dict) and "write_confirmed" in tr.data:
                confirmed = bool(tr.data.get("write_confirmed"))
                op_name = tr.data.get("operation", "write")
                tool_label = "Memory" if tr.tool == "memory" else "Task"
                checks.append(CheckItem(
                    name="write_confirmed",
                    passed=confirmed,
                    detail=f"{tool_label} write '{op_name}' confirmed via database read-back."
                    if confirmed else
                    f"{tool_label} write '{op_name}' read-back confirmation failed.",
                ))
                if not confirmed:
                    unresolved.append(f"Write operation '{op_name}' failed read-back verification.")

        # Check: The answer must never claim a task was created or completed unless confirmed
        task_claim_patterns = [
            r"\b(?:created|added)\s+(?:the\s+)?(?:onboarding\s+)?tasks?\b",
            r"\b(?:completed|finished)\s+(?:the\s+)?task\b",
            r"\bmarked\s+task\s+#?\d+\s+as\s+(?:done|complete)\b",
        ]
        has_task_claim = any(re.search(pat, ans_lower) for pat in task_claim_patterns)
        task_write_confirmed = any(
            tr.tool == "tasks" and isinstance(tr.data, dict) and tr.data.get("write_confirmed")
            for tr in tool_results
        )
        if has_task_claim and not task_write_confirmed:
            checks.append(CheckItem(
                name="write_confirmed",
                passed=False,
                detail="Answer claims a task was created or completed, but no task write was confirmed in database.",
            ))
            unresolved.append("Unconfirmed task write claim in answer.")

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

        # 7. Check: Web research citations and claim grounding
        has_web_step = any(tr.tool == "web_research" for tr in tool_results)
        if has_web_step:
            web_results = [tr for tr in tool_results if tr.tool == "web_research"]
            successful_web_data = [tr.data for tr in web_results if tr.success and tr.data]

            if successful_web_data:
                flattened_results = []
                for d in successful_web_data:
                    if isinstance(d, list):
                        flattened_results.extend([it for it in d if isinstance(it, dict)])
                    elif isinstance(d, dict):
                        flattened_results.append(d)

                has_citation, citation_detail = self.verify_web_sources_cited(answer, flattened_results)
                checks.append(CheckItem(
                    name="web_sources_cited",
                    passed=has_citation,
                    detail=citation_detail,
                ))
                if not has_citation:
                    unresolved.append("External web research sources were not explicitly cited.")

                snippets_list = [
                    str(it.get("snippet", "")) + " " + str(it.get("title", ""))
                    for it in flattened_results
                ]
                claims_grounded, claims_detail = self.verify_web_claims_grounded(
                    answer=answer,
                    snippets=snippets_list,
                    allowed_facts=extracted_facts,
                    question=question,
                )
                checks.append(CheckItem(
                    name="web_claims_grounded",
                    passed=claims_grounded,
                    detail=claims_detail,
                ))
                if not claims_grounded:
                    unresolved.append(claims_detail)
            else:
                external_refusal_markers = [
                    "could not find", "not found", "unavailable", "cannot verify", "could not verify",
                    "unable to verify", "no information", "not available"
                ]
                acknowledged = any(m in ans_lower for m in external_refusal_markers)
                checks.append(CheckItem(
                    name="web_sources_cited",
                    passed=acknowledged,
                    detail="Correctly acknowledged that external web information could not be verified."
                    if acknowledged else
                    "Failed: Web search failed but answer did not acknowledge lack of external verification.",
                ))
                checks.append(CheckItem(
                    name="web_claims_grounded",
                    passed=acknowledged,
                    detail="No hallucinated external claims when web search returned no data."
                    if acknowledged else
                    "Failed: Answer did not state external verification was unavailable.",
                ))

        # 8. Check: External comparison coverage
        external_cues = ["compare", "versus", "statutory", "current", "latest", "industry"]
        q_lower = question.lower()
        has_external_cue = any(re.search(r"\b" + re.escape(cue) + r"\b", q_lower) for cue in external_cues)
        has_web_in_plan = any(step.tool.strip().lower() == "web_research" for step in plan.steps)

        if has_external_cue and not has_web_in_plan:
            checks.append(CheckItem(
                name="external_comparison_coverage",
                passed=False,
                detail="Question contains comparison or external cues, but no web_research step was planned.",
            ))
            unresolved.append("external comparison not performed")
        elif has_external_cue and has_web_in_plan:
            checks.append(CheckItem(
                name="external_comparison_coverage",
                passed=True,
                detail="External comparison step included in plan.",
            ))

        # Overall verification outcome
        overall_passed = all(c.passed for c in checks)

        return Verification(
            passed=overall_passed,
            checks=checks,
            unresolved=unresolved,
        )

    def verify_web_sources_cited(
        self,
        answer: str,
        web_results: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """
        Verifies that an answer cites external URLs or domain names from the retrieved web results.
        Cannot be satisfied by vague markers like 'Source:', 'sources:', or generic phrases.
        """
        if not web_results:
            return False, "No external web results available to cite."

        ans_lower = answer.lower()

        for item in web_results:
            if isinstance(item, dict):
                url = str(item.get("url", "")).strip().lower()
                if url and url in ans_lower:
                    return True, "Web research sources and URLs cited in grounded response."

                domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
                if domain_match:
                    domain = domain_match.group(1).lower()
                    if domain and len(domain) > 3 and domain in ans_lower:
                        return True, "Web research sources and URLs cited in grounded response."

        return False, "External web research sources or URLs were not explicitly cited."

    @staticmethod
    def find_derived_number_explanation(
        val: float,
        evidence_numbers: Set[float],
        answer_numbers: Optional[Set[float]] = None,
    ) -> Optional[str]:
        """
        Determines whether a numeric value equals a difference, absolute difference, sum,
        or percentage of two numbers that BOTH appear in the evidence AND in the answer text.
        Returns a human-readable explanation (e.g. '4 = 28 - 24') or None.
        """
        def _fmt(n: float) -> str:
            return str(int(n)) if n.is_integer() else str(n)

        nums = sorted(list(evidence_numbers))
        if len(nums) < 2:
            return None

        # Both operands must appear in the answer text (as well as in evidence)
        if answer_numbers is not None:
            candidate_nums = [n for n in nums if n in answer_numbers]
        else:
            candidate_nums = nums

        if len(candidate_nums) < 2:
            return None

        # 1. Percentage check (a / b * 100 where a != b, b > 0)
        for a in candidate_nums:
            for b in candidate_nums:
                if a == b or b <= 0:
                    continue
                pct = (a / b) * 100.0
                if abs(val - round(pct, 0)) < 1e-4 or abs(val - round(pct, 1)) < 1e-4:
                    return f"{_fmt(val)}% = ({_fmt(a)} / {_fmt(b)}) * 100"

        # 2. Difference check (a - b where a > b)
        for a in candidate_nums:
            for b in candidate_nums:
                if a > b and abs(val - (a - b)) < 1e-4:
                    return f"{_fmt(val)} = {_fmt(a)} - {_fmt(b)}"

        # 3. Sum check (a + b where a != b)
        for a in candidate_nums:
            for b in candidate_nums:
                if a != b and abs(val - (a + b)) < 1e-4:
                    return f"{_fmt(val)} = {_fmt(a)} + {_fmt(b)}"

        # 4. Absolute difference fallback |a - b| (where a != b)
        for a in candidate_nums:
            for b in candidate_nums:
                if a != b and abs(val - abs(a - b)) < 1e-4 and abs(a - b) > 0:
                    return f"{_fmt(val)} = |{_fmt(a)} - {_fmt(b)}|"

        return None

    def verify_web_claims_grounded(
        self,
        answer: str,
        snippets: List[str],
        allowed_facts: Optional[Set[str]] = None,
        question: str = "",
    ) -> Tuple[bool, str]:
        """Verifies that numbers and versions in answer are grounded in web snippets, evidence facts, or derived."""
        clean_ans = re.sub(r"(?:^|[\r\n\s])(?:\d+[\.\)]|\(\d+\))\s+", " ", answer)
        list_markers = set(re.findall(r"(?:^|[\r\n\s])(\d+)[\.\)]\s+", answer))
        list_markers.update(re.findall(r"(?:^|[\r\n\s])\((\d+)\)\s+", answer))

        numbers_and_versions_ans = set(re.findall(r"\b\d+(?:\.\d+)*\b", clean_ans)) - list_markers

        web_snippets_text = " ".join(snippets)
        allowed = set(allowed_facts or set())
        if question:
            allowed.update(re.findall(r"\b\d+(?:\.\d+)*\b", question))

        # Collect evidence numeric values from allowed_facts and snippets
        evidence_numeric_values: Set[float] = set()
        for f in (allowed_facts or set()):
            for m in re.findall(r"\b\d+(?:\.\d+)?\b", str(f)):
                try:
                    evidence_numeric_values.add(float(m))
                except ValueError:
                    pass
        for s in snippets:
            for m in re.findall(r"\b\d+(?:\.\d+)?\b", str(s)):
                try:
                    evidence_numeric_values.add(float(m))
                except ValueError:
                    pass

        unsupported_claims = set()
        for claim in numbers_and_versions_ans:
            if claim not in allowed and claim not in web_snippets_text:
                # Check if claim is derivable from two numbers in evidence
                is_derived = False
                try:
                    val = float(claim)
                    if self.find_derived_number_explanation(val, evidence_numeric_values) is not None:
                        is_derived = True
                except ValueError:
                    pass
                if not is_derived:
                    unsupported_claims.add(claim)

        if not unsupported_claims:
            return True, "All numbers and version strings in web-assisted answer are grounded in retrieved snippets."
        return False, f"Ungrounded claims or version strings found in answer: {sorted(unsupported_claims)}"

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
