import json
import requests

PROMPT_TEMPLATE = """You are a strict query planner for a company knowledge database.
The database ONLY has facts about NextGen AI:
1. Company Information:
   - name: NextGen AI
   - location: Chennai
   - employees: 1000
2. Departments:
   - AI: 200 employees
   - Backend: 150 employees
3. Leave Policy:
   - Annual Leave: 30 days

Allowed operations:
- intent: "company_info", query_type: "company_name" | "company_location" | "company_employees" | "all_company"
- intent: "department_employees", query_type: "dept_headcount" (entity: department name) | "all_depts" | "total_dept_headcount"
- intent: "leave_policy", query_type: "leave_days" (entity: leave type) | "all_leaves"
- intent: "unknown", query_type: "unknown", entity: null

Examples:
Question: "Where is the company located and how many employees work in AI?"
JSON: {{"queries": [{{"intent": "company_info", "query_type": "company_location", "entity": null}}, {{"intent": "department_employees", "query_type": "dept_headcount", "entity": "AI"}}]}}

Question: "What is the CEO's name and how many employees are in AI?"
JSON: {{"queries": [{{"intent": "department_employees", "query_type": "dept_headcount", "entity": "AI"}}]}}

Question: "What is the company revenue and annual leave allowance?"
JSON: {{"queries": [{{"intent": "leave_policy", "query_type": "leave_days", "entity": "Annual Leave"}}]}}

Question: "Who is the CEO and what is the company revenue?"
JSON: {{"queries": [{{"intent": "unknown", "query_type": "unknown", "entity": null}}]}}

Output JSON with format:
{{
  "queries": [
    {{
      "intent": "...",
      "query_type": "...",
      "entity": "..." or null
    }}
  ]
}}
Do NOT generate SQL. Only output the JSON object.

User Question: "{question}"
Respond with ONLY the JSON object.
"""

questions = [
    "What is the CEO's name and how many employees are in AI?",
    "What is the company revenue and annual leave allowance?",
    "Where is the company located, how many employees work in AI, and how many annual leave days do employees get?",
    "Who is the CEO?",
    "What is the company revenue?",
]

for q in questions:
    p = PROMPT_TEMPLATE.format(question=q)
    resp = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": "qwen2.5:3b", "prompt": p, "format": "json", "stream": False, "options": {"temperature": 0.0}},
        timeout=30,
    )
    res_text = resp.json().get("response", "")
    print("=" * 60)
    print("Q:", q)
    print("Plan:", res_text)
