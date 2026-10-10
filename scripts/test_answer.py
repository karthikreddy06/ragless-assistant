import requests

prompt = """You are a professional knowledge assistant for NextGen AI.
Your job is to answer the user's question using ONLY the provided retrieved data from the company's SQLite database.

STRICT RULES:
1. Use ONLY the facts provided in the Retrieved Data below.
2. Do NOT extrapolate, hallucinate, or add facts not present in the data.
3. If one requested fact is missing or unsupported (e.g. CEO name or revenue), clearly state that specific fact is not available in the database.
4. Keep the answer concise, direct, polite, and helpful (1-2 sentences).

User Question: What is the CEO's name and how many employees are in AI?
Retrieved Data from SQLite Database:
[
  {
    "source": "sqlite",
    "type": "department",
    "name": "AI",
    "employees": 200
  }
]

Answer:"""

r = requests.post(
    "http://127.0.0.1:11434/api/generate",
    json={"model": "qwen2.5:3b", "prompt": prompt, "stream": False},
    timeout=30,
)

print(r.json()["response"])
