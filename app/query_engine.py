import logging
import re
import time
from typing import Any, Dict, List, Optional

from app.database import (
    get_all_department_names,
    get_all_leave_types,
    get_company_info,
    get_current_company_name,
    get_department_employees,
    get_department_total_headcount,
    get_leave_policy,
    get_onboarding_items,
)
from app.llm import generate_grounded_answer, generate_query_plan
from app.models import QueryOperation, QueryPlan

logger = logging.getLogger(__name__)


def validate_and_refine_query_plan(
    raw_plan: Dict[str, Any],
    question: str,
    known_departments: List[str],
    known_leave_types: List[str],
) -> QueryPlan:
    """
    Python validation and safety layer:
    1. Validates the LLM-produced query plan against strict Pydantic schemas and allowlists.
    2. Rejects arbitrary SQL keywords, punctuation, or malicious commands safely.
    3. Prunes unsupported hallucinated operations.
    4. Supplements operations for multi-part questions if recognized database entities are mentioned.
    5. Deduplicates operations and enforces a maximum of 5 operations.
    """
    norm_q = question.lower()
    valid_ops: List[QueryOperation] = []

    # 1. Pydantic schema & allowlist validation
    try:
        parsed_plan = QueryPlan.model_validate(raw_plan)
        for op in parsed_plan.queries:
            # Recheck relevance against user question to prune hallucinations
            if op.intent == "leave_policy":
                leave_keywords = ["leave", "holiday", "vacation", "off day", "sick", "casual", "annual", "allowance", "days do we get"]
                if not any(w in norm_q for w in leave_keywords):
                    continue
            elif op.intent == "department_employees":
                dept_keywords = ["department", "team", "dept", "headcount", "workforce", "people", "size", "members"]
                dept_entity_mentioned = op.entity and op.entity.lower() in norm_q
                dept_known_mentioned = any(d.lower() in norm_q for d in known_departments)
                if not (dept_entity_mentioned or dept_known_mentioned or any(w in norm_q for w in dept_keywords)):
                    continue
            elif op.intent == "company_info":
                company_keywords = [
                    "company", "organization", "organisation", "business", "firm", "employer",
                    "who are you", "what is this", "name", "where", "location", "located", "city",
                    "headquarters", "workforce", "total employees", "employees work",
                    "how many employees", "employees does it have", "people are there"
                ]
                if not any(w in norm_q for w in company_keywords):
                    continue

            elif op.intent == "onboarding":
                onboarding_keywords = ["onboarding", "checklist", "orientation", "new hire", "induction", "day 1", "tasks", "due"]
                if not any(w in norm_q for w in onboarding_keywords):
                    continue

            if op.intent != "unknown" and op.query_type != "unknown":
                valid_ops.append(op)
    except Exception as e:
        logger.warning("Query plan validation rejected raw plan %s: %s", raw_plan, e)

    # 2. Check existing coverage to avoid duplicate additions
    has_company_name = any(op.intent == "company_info" and op.query_type in ("company_name", "name") for op in valid_ops)
    has_company_location = any(op.intent == "company_info" and op.query_type in ("company_location", "location") for op in valid_ops)
    has_company_employees = any(op.intent == "company_info" and op.query_type in ("company_employees", "employees") for op in valid_ops)
    has_all_company = any(op.intent == "company_info" and op.query_type in ("all_company", "company") for op in valid_ops)

    # Check known departments
    for dept in known_departments:
        if re.search(rf"\b{re.escape(dept.lower())}\b", norm_q):
            already_has_dept = any(
                op.intent == "department_employees" and (
                    (op.entity and op.entity.lower() == dept.lower()) or op.query_type == "all_depts"
                )
                for op in valid_ops
            )
            if not already_has_dept and len(valid_ops) < 5:
                valid_ops.append(
                    QueryOperation(intent="department_employees", query_type="dept_headcount", entity=dept)
                )

    # Check known leave types
    for lt in known_leave_types:
        lt_clean = lt.lower().replace("leave", "").strip()
        if lt_clean and re.search(rf"\b{re.escape(lt_clean)}\b", norm_q):
            already_has_leave = any(
                op.intent == "leave_policy" and (
                    (op.entity and lt_clean in op.entity.lower()) or op.query_type == "all_leaves"
                )
                for op in valid_ops
            )
            if not already_has_leave and len(valid_ops) < 5:
                valid_ops.append(
                    QueryOperation(intent="leave_policy", query_type="leave_days", entity=lt)
                )

    if any(w in norm_q for w in ["annual leave", "vacation allowance", "leave allowance", "annual leaves", "days do we get"]):
        already_has_annual = any(
            op.intent == "leave_policy" and (
                (op.entity and "annual" in op.entity.lower()) or op.query_type == "all_leaves"
            )
            for op in valid_ops
        )
        if not already_has_annual and len(valid_ops) < 5:
            valid_ops.append(
                QueryOperation(intent="leave_policy", query_type="leave_days", entity="Annual Leave")
            )

    # Check company information
    if any(w in norm_q for w in ["organization", "organisation", "company name", "name of the company", "what is the company name", "what company is this", "what organization is this"]):
        if not has_company_name and not has_all_company and len(valid_ops) < 5:
            valid_ops.append(
                QueryOperation(intent="company_info", query_type="company_name", entity=None)
            )
            has_company_name = True

    if any(w in norm_q for w in ["location", "located", "where is the company", "headquarters", "city"]):
        if not has_company_location and not has_all_company and len(valid_ops) < 5:
            valid_ops.append(
                QueryOperation(intent="company_info", query_type="company_location", entity=None)
            )
            has_company_location = True

    if any(w in norm_q for w in ["how many employees does it have", "company employees", "how many employees work there", "total employees work", "people are there"]):
        if not has_company_employees and not has_all_company and len(valid_ops) < 5:
            valid_ops.append(
                QueryOperation(intent="company_info", query_type="company_employees", entity=None)
            )
            has_company_employees = True

    if any(w in norm_q for w in ["onboarding", "checklist", "orientation", "new hire", "induction"]):
        already_has_onboarding = any(op.intent == "onboarding" for op in valid_ops)
        if not already_has_onboarding and len(valid_ops) < 5:
            dept_found = None
            for d in known_departments:
                if d.lower() in norm_q:
                    dept_found = d
                    break
            valid_ops.append(
                QueryOperation(intent="onboarding", query_type="onboarding_checklist", entity=dept_found)
            )

    # Fallback to unknown if no operations found
    if not valid_ops:
        valid_ops = [QueryOperation(intent="unknown", query_type="unknown", entity=None)]

    return QueryPlan(queries=valid_ops[:5])


def execute_query_operation(op: QueryOperation) -> List[Dict[str, Any]]:
    """
    Executes a single validated QueryOperation using predefined parameterized database functions.
    Arbitrary SQL execution is strictly impossible.
    Returns tagged structured records.
    """
    results: List[Dict[str, Any]] = []

    if op.intent == "company_info":
        if op.query_type in ["company_name", "name"]:
            rows = get_company_info("name")
        elif op.query_type in ["company_location", "location"]:
            rows = get_company_info("location")
        elif op.query_type in ["company_employees", "employees"]:
            rows = get_company_info("employees")
        else:
            rows = get_company_info()

        for r in rows:
            results.append({
                "source": "sqlite",
                "type": "company",
                "source_type": "company",
                "category": r.get("category", "company"),
                "key": r.get("key"),
                "value": r.get("value"),
            })

    elif op.intent == "department_employees":
        if op.query_type in ["dept_headcount", "department_employees"]:
            rows = get_department_employees(op.entity)
            for r in rows:
                results.append({
                    "source": "sqlite",
                    "type": "department",
                    "source_type": "department",
                    "name": r.get("name"),
                    "department": r.get("name"),
                    "employees": r.get("employees"),
                })
        elif op.query_type == "total_dept_headcount":
            total = get_department_total_headcount()
            results.append({
                "source": "sqlite",
                "type": "department_total",
                "source_type": "department_total",
                "total_department_employees": total,
                "employees": total,
            })
        else:  # all_depts
            rows = get_department_employees()
            for r in rows:
                results.append({
                    "source": "sqlite",
                    "type": "department",
                    "source_type": "department",
                    "name": r.get("name"),
                    "department": r.get("name"),
                    "employees": r.get("employees"),
                })

    elif op.intent == "leave_policy":
        if op.query_type in ["leave_days", "leave_policy"]:
            rows = get_leave_policy(op.entity)
        else:  # all_leaves
            rows = get_leave_policy()

        for r in rows:
            results.append({
                "source": "sqlite",
                "type": "leave_policy",
                "source_type": "leave_policy",
                "leave_type": r.get("leave_type"),
                "days": r.get("days"),
            })

    elif op.intent == "onboarding":
        rows = get_onboarding_items(op.entity)
        for r in rows:
            results.append({
                "source": "sqlite",
                "type": "onboarding",
                "source_type": "onboarding",
                "id": r.get("id"),
                "title": r.get("title"),
                "description": r.get("description"),
                "department": r.get("department"),
                "day_due": r.get("day_due"),
            })

    return results


def _is_duplicate_record(existing: List[Dict[str, Any]], item: Dict[str, Any]) -> bool:
    """Helper to prevent duplicate facts in the retrieved data list."""
    for ex in existing:
        if ex.get("type") == item.get("type"):
            if item.get("type") == "company" and ex.get("key") == item.get("key"):
                return True
            if item.get("type") == "department" and str(ex.get("name")).lower() == str(item.get("name")).lower():
                return True
            if item.get("type") == "leave_policy" and str(ex.get("leave_type")).lower() == str(item.get("leave_type")).lower():
                return True
            if item.get("type") == "onboarding" and ex.get("id") == item.get("id"):
                return True
    return False


def execute_structured_retrieval(
    intent: str,
    query_type: Optional[str],
    entity: Optional[str],
    question: str,
) -> tuple[Optional[List[Dict[str, Any]]], str, Optional[str]]:
    """
    Backward-compatible single-query retrieval interface.
    Maps intent and query_type safely to parameterized database calls.
    Returns: (retrieved_data, resolved_intent, resolved_query_type)
    """
    norm_q = question.lower()
    known_depts = get_all_department_names()
    known_leaves = get_all_leave_types()

    # Rule-assisted refinement if intent was unknown
    if intent == "unknown":
        if any(w in norm_q for w in ["leave", "holiday", "vacation", "off day", "sick", "casual", "annual"]):
            intent = "leave_policy"
        elif any(w in norm_q for w in ["department", "team", "dept"]) or any(d.lower() in norm_q for d in known_depts):
            intent = "department_employees"
        elif any(w in norm_q for w in ["company", "located", "location", "who are you", "total employees", "employees"]):
            intent = "company_info"

    if intent == "unknown":
        return None, "unknown", "unknown"

    try:
        op = QueryOperation(
            intent=intent,
            query_type=query_type or "unknown",
            entity=entity,
        )
    except Exception:
        # Default query type based on intent
        default_qtypes = {
            "company_info": "all_company",
            "department_employees": "all_depts",
            "leave_policy": "all_leaves",
            "onboarding": "onboarding_checklist",
        }
        op = QueryOperation(
            intent=intent,
            query_type=default_qtypes.get(intent, "unknown"),
            entity=entity,
        )

    data = execute_query_operation(op)
    return (data if data else None), op.intent, op.query_type


def process_query(question: str) -> Dict[str, Any]:
    """
    Complete RAGless multi-query orchestration pipeline:
    1. Receive user natural-language question.
    2. Decompose question into structured QueryPlan via local Ollama (qwen2.5:3b).
    3. Pass through Python validation layer to enforce allowlists and security boundaries.
    4. Execute safe predefined parameterized SQLite functions for each operation.
    5. Aggregate and tag structured records with source and domain types (zero cross-contamination).
    6. Synthesize grounded final response with local Ollama using ONLY retrieved facts.
    7. Return response alongside transparent execution metadata.
    """
    start_time = time.perf_counter()

    known_depts = get_all_department_names()
    known_leaves = get_all_leave_types()

    # Step 1: LLM query plan generation
    raw_plan = generate_query_plan(question)

    # Step 2: Python validation and refinement
    query_plan = validate_and_refine_query_plan(
        raw_plan=raw_plan,
        question=question,
        known_departments=known_depts,
        known_leave_types=known_leaves,
    )

    # Step 3: Execute predefined retrieval operations
    combined_data: List[Dict[str, Any]] = []
    executed_ops: List[QueryOperation] = []

    for op in query_plan.queries:
        if op.intent == "unknown":
            continue
        op_results = execute_query_operation(op)
        if op_results:
            executed_ops.append(op)
            for item in op_results:
                if not _is_duplicate_record(combined_data, item):
                    combined_data.append(item)

    # Step 4: Determine metadata fields
    unique_intents = list(dict.fromkeys(op.intent for op in executed_ops))
    if len(unique_intents) == 1:
        resolved_intent = unique_intents[0]
        resolved_query_type = executed_ops[0].query_type if len(executed_ops) == 1 else "all_company"
    elif len(unique_intents) > 1:
        resolved_intent = "multi_query"
        resolved_query_type = "multi_table"
    else:
        resolved_intent = "unknown"
        resolved_query_type = "unknown"

    # Step 5: Grounded answer generation using ONLY retrieved data
    if combined_data:
        answer = generate_grounded_answer(
            question=question,
            retrieved_data=combined_data,
            intent=resolved_intent,
        )
    else:
        company_name = get_current_company_name()
        answer = (
            f"I could not find information about that in the company database. "
            f"{company_name}'s knowledge base only contains company details (name, location, total employees), "
            f"department headcounts, and leave policies."
        )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "question": question,
        "answer": answer,
        "source": "sqlite",
        "intent": resolved_intent,
        "query_type": resolved_query_type,
        "retrieved_data": combined_data if combined_data else None,
        "processing_time_ms": elapsed_ms,
        "queries_executed": len(executed_ops),
    }
