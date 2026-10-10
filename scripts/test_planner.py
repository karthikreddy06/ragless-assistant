import json
import requests

prompt = """You are a strict query planner for a company knowledge database.
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

Your task is to identify all database retrieval operations needed to answer the user question.
Decompose compound questions into separate query operations.
If a question contains multiple topics, generate an operation for EACH valid topic.

Allowed Operations:
- intent: 'company_info' -> query_type: 'company_name', 'company_location', 'company_employees', or 'all_company'
- intent: 'department_employees' -> query_type: 'dept_headcount' (entity=dept name), 'all_depts'
- intent: 'leave_policy' -> query_type: 'leave_days' (entity=leave type), 'all_leaves'
- intent: 'unknown' -> query_type: 'unknown' (use ONLY when NO part of the question can be answered by the database)

Examples:
Q: "How many employees are in AI and how many annual leave days are available?"
{"queries": [{"intent": "department_employees", "query_type": "dept_headcount", "entity": "AI"}, {"intent": "leave_policy", "query_type": "leave_days", "entity": "Annual Leave"}]}

Q: "What is the company name, where is it located, and how many people work in AI?"
{"queries": [{"intent": "company_info", "query_type": "company_name", "entity": null}, {"intent": "company_info", "query_type": "company_location", "entity": null}, {"intent": "department_employees", "query_type": "dept_headcount", "entity": "AI"}]}

Q: "What is the CEO's name and how many employees are in AI?"
{"queries": [{"intent": "department_employees", "query_type": "dept_headcount", "entity": "AI"}]}

Q: "What is the capital of France?"
{"queries": [{"intent": "unknown", "query_type": "unknown", "entity": null}]}

Question: "What is the CEO's name and how many employees are in AI?"
Return ONLY JSON.
"""

r = requests.post(
    "http://127.0.0.1:11434/api/generate",
    json={"model": "qwen2.5:3b", "prompt": prompt, "stream": False, "format": "json"},
    timeout=30,
)

print(r.json()["response"])
