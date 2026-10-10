import json
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

ALLOWED_INTENTS = {"company_info", "department_employees", "leave_policy", "unknown"}

ALLOWED_QUERY_TYPES = {
    # company operations
    "company_name", "company_location", "company_employees", "all_company",
    "name", "location", "employees",
    # department operations
    "dept_headcount", "all_depts", "total_dept_headcount", "department_employees",
    # leave operations
    "leave_days", "all_leaves", "leave_policy",
    # unknown
    "unknown"
}

FORBIDDEN_SQL_PATTERNS = [
    r";", r"--", r"/\*", r"\*/",
    r"\bselect\b", r"\binsert\b", r"\bupdate\b", r"\bdelete\b",
    r"\bdrop\b", r"\balter\b", r"\btruncate\b", r"\bunion\b",
    r"\bexec\b", r"\bexecute\b",
]

class QueryOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
    query_type: str
    entity: Optional[str] = None

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in ALLOWED_INTENTS:
            raise ValueError(f"Disallowed intent '{v}'")
        return s

    @field_validator("query_type")
    @classmethod
    def validate_query_type(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in ALLOWED_QUERY_TYPES:
            raise ValueError(f"Disallowed query_type '{v}'")
        return s

    @field_validator("entity")
    @classmethod
    def sanitize_entity(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        if not s or s.lower() in ("null", "none"):
            return None
        s_lower = s.lower()
        for pat in FORBIDDEN_SQL_PATTERNS:
            if re.search(pat, s_lower):
                raise ValueError(f"Forbidden SQL syntax detected: '{v}'")
        return s

class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: List[QueryOperation] = Field(..., min_length=1, max_length=5)


def validate_and_refine_query_plan(
    raw_plan: Dict[str, Any],
    question: str,
    known_departments: List[str],
    known_leave_types: List[str]
) -> QueryPlan:
    """
    Python validation layer:
    1. Validates raw plan against Pydantic schema and allowlists.
    2. Filters out hallucinations (e.g. leave operation if leave not mentioned).
    3. Supplements mixed questions if LLM missed a supported topic.
    4. Enforces max 5 operations and deduplication.
    """
    norm_q = question.lower()
    clean_ops: List[QueryOperation] = []

    # Try parsing into Pydantic
    try:
        plan = QueryPlan.model_validate(raw_plan)
        for op in plan.queries:
            # Recheck relevance
            if op.intent == "leave_policy":
                if not any(w in norm_q for w in ["leave", "holiday", "vacation", "off day", "sick", "casual", "annual"]):
                    continue
            elif op.intent == "department_employees":
                # If specific entity, check if mentioned or generic department query
                if op.entity and op.entity.lower() not in norm_q and not any(w in norm_q for w in ["department", "team", "dept", "workforce", "people", "headcount"]):
                    continue
            elif op.intent == "company_info":
                if not any(w in norm_q for w in ["company", "name", "where", "location", "located", "city", "total employee", "how many employee", "workforce", "employees"]):
                    continue
            if op.intent != "unknown" and op.query_type != "unknown":
                clean_ops.append(op)
    except Exception as e:
        # If raw_plan failed validation completely (e.g. SQL fields or invalid types)
        pass

    # Domain supplementation: check if known entities/domains are in question but missing from clean_ops
    existing_types = {f"{op.intent}:{op.query_type}:{op.entity or ''}" for op in clean_ops}

    # 1. Check departments
    for dept in known_departments:
        if re.search(rf"\b{re.escape(dept.lower())}\b", norm_q):
            key = f"department_employees:dept_headcount:{dept}"
            if key not in existing_types and len(clean_ops) < 5:
                clean_ops.append(QueryOperation(intent="department_employees", query_type="dept_headcount", entity=dept))
                existing_types.add(key)

    # 2. Check leave policies
    for lt in known_leave_types:
        lt_clean = lt.lower().replace("leave", "").strip()
        if lt_clean and re.search(rf"\b{re.escape(lt_clean)}\b", norm_q):
            key = f"leave_policy:leave_days:{lt}"
            if key not in existing_types and len(clean_ops) < 5:
                clean_ops.append(QueryOperation(intent="leave_policy", query_type="leave_days", entity=lt))
                existing_types.add(key)
    if any(w in norm_q for w in ["annual leave", "vacation allowance", "leave allowance", "annual leaves"]) and not any(op.intent == "leave_policy" for op in clean_ops):
        if len(clean_ops) < 5:
            clean_ops.append(QueryOperation(intent="leave_policy", query_type="leave_days", entity="Annual Leave"))

    # 3. Check company info
    if any(w in norm_q for w in ["company name", "name of the company", "what is the company name"]):
        key = "company_info:company_name:None"
        if key not in existing_types and len(clean_ops) < 5:
            clean_ops.append(QueryOperation(intent="company_info", query_type="company_name", entity=None))
            existing_types.add(key)
    if any(w in norm_q for w in ["location", "located", "where is the company", "headquarters", "city"]):
        key = "company_info:company_location:None"
        if key not in existing_types and len(clean_ops) < 5:
            clean_ops.append(QueryOperation(intent="company_info", query_type="company_location", entity=None))
            existing_types.add(key)
    if any(w in norm_q for w in ["how many employees does it have", "company employees", "how many employees work there", "total employees work"]):
        key = "company_info:company_employees:None"
        if key not in existing_types and len(clean_ops) < 5:
            clean_ops.append(QueryOperation(intent="company_info", query_type="company_employees", entity=None))
            existing_types.add(key)

    if not clean_ops:
        clean_ops = [QueryOperation(intent="unknown", query_type="unknown", entity=None)]

    return QueryPlan(queries=clean_ops[:5])


# Test with sample queries
test_cases = [
    ({"queries": [{"intent": "unknown", "query_type": "unknown", "entity": None}]}, "What is the CEO's name and how many employees are in AI?"),
    ({"queries": [{"intent": "unknown", "query_type": "unknown", "entity": None}]}, "What is the company revenue and annual leave allowance?"),
    ({"queries": [{"intent": "leave_policy", "query_type": "leave_days", "entity": "Annual Leave"}]}, "What is the company revenue?"),
    ({"queries": [{"intent": "unknown", "query_type": "unknown", "entity": None}]}, "Who is the CEO?"),
    ({"queries": [{"intent": "company_info", "query_type": "company_location", "entity": None}, {"intent": "department_employees", "query_type": "dept_headcount", "entity": "AI"}]}, "Where is the company located and how many employees work in AI?"),
    ({"queries": [{"sql": "DROP TABLE departments"}]}, "DROP TABLE departments"),
]

depts = ["AI", "Backend"]
leaves = ["Annual Leave"]

for raw, q in test_cases:
    refined = validate_and_refine_query_plan(raw, q, depts, leaves)
    print("=" * 60)
    print("Q:", q)
    print("Refined:", refined.model_dump())
