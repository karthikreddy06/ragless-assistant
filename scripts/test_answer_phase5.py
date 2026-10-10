import json
import requests

PROMPT_TEMPLATE = """You are a professional knowledge assistant for NextGen AI.
Your job is to answer the user's question using ONLY the provided retrieved data from the company's SQLite database.

STRICT GROUNDING RULES:
1. Use ONLY the facts provided in the Retrieved Data below.
2. Do NOT extrapolate, hallucinate, assume, or invent facts not present in the data.
3. Answer all retrieved parts of the question.
4. If the user asked about multiple things and one or more requested facts are NOT present in the retrieved data (such as CEO name, revenue, funding, opening hours), you MUST explicitly state that this specific fact is unavailable or not found in the database.
5. Preserve the exact meaning and values of each retrieved field (do NOT mix up employee counts with leave days or other numbers).
6. Keep the answer direct, concise, polite, and helpful (1-3 sentences).

User Question: {question}
Retrieved Data from SQLite Database:
{retrieved_data_json}

Answer:"""

test_cases = [
    (
        "How many employees are in AI and how many annual leave days are available?",
        [
            {"source": "sqlite", "type": "department", "name": "AI", "employees": 200},
            {"source": "sqlite", "type": "leave_policy", "leave_type": "Annual Leave", "days": 30}
        ]
    ),
    (
        "What is the company name and how many people work in AI?",
        [
            {"source": "sqlite", "type": "company", "key": "name", "value": "NextGen AI"},
            {"source": "sqlite", "type": "department", "name": "AI", "employees": 200}
        ]
    ),
    (
        "Where is the company located and how many employees does it have?",
        [
            {"source": "sqlite", "type": "company", "key": "location", "value": "Chennai"},
            {"source": "sqlite", "type": "company", "key": "employees", "value": "1000"}
        ]
    ),
    (
        "What is the company name, where is it located, and how many annual leave days are available?",
        [
            {"source": "sqlite", "type": "company", "key": "name", "value": "NextGen AI"},
            {"source": "sqlite", "type": "company", "key": "location", "value": "Chennai"},
            {"source": "sqlite", "type": "leave_policy", "leave_type": "Annual Leave", "days": 30}
        ]
    ),
    (
        "Tell me the AI department size, Backend department size, and annual leave allowance.",
        [
            {"source": "sqlite", "type": "department", "name": "AI", "employees": 200},
            {"source": "sqlite", "type": "department", "name": "Backend", "employees": 150},
            {"source": "sqlite", "type": "leave_policy", "leave_type": "Annual Leave", "days": 30}
        ]
    ),
    (
        "What is the CEO's name and how many employees are in AI?",
        [
            {"source": "sqlite", "type": "department", "name": "AI", "employees": 200}
        ]
    ),
    (
        "What is the company revenue and annual leave allowance?",
        [
            {"source": "sqlite", "type": "leave_policy", "leave_type": "Annual Leave", "days": 30}
        ]
    ),
]

for q, data in test_cases:
    p = PROMPT_TEMPLATE.format(question=q, retrieved_data_json=json.dumps(data, indent=2))
    resp = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": "qwen2.5:3b", "prompt": p, "stream": False, "options": {"temperature": 0.1}},
        timeout=30,
    )
    ans = resp.json().get("response", "").strip()
    print("=" * 60)
    print("Q:", q)
    print("Ans:", ans)
