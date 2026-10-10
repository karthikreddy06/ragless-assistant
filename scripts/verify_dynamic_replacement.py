import json
import requests

base = "http://127.0.0.1:8000"

print("====================================")
print("STEP 1: IMPORT COMPANY A (TechNova)")
print("====================================")
with open("data/sample_knowledge.json", "rb") as f:
    r = requests.post(f"{base}/knowledge/import", files={"file": ("sample_knowledge.json", f, "application/json")})
print("Import Status:", r.status_code)
print("Import Response:", json.dumps(r.json(), indent=2))

print("\n--- GET /knowledge (Company A) ---")
r = requests.get(f"{base}/knowledge")
print("Status:", r.status_code)
print(json.dumps(r.json(), indent=2))

questions = [
    "What is the company name?",
    "Where is the company located?",
    "How many employees are in the AI department?",
    "How many annual leave days are available?"
]

print("\n--- Questions for Company A ---")
results_a = []
for q in questions:
    resp = requests.post(f"{base}/ask", json={"question": q}).json()
    print(f"Q: {q}")
    print(f"Answer: {resp['answer']}")
    print(f"Retrieved Data: {resp['retrieved_data']}")
    print(f"Source: {resp['source']}")
    print("-" * 40)
    results_a.append((q, resp['answer'], resp['retrieved_data']))

print("\n====================================")
print("STEP 2: IMPORT COMPANY B (NextGen AI)")
print("====================================")
with open("data/test_company.json", "rb") as f:
    r = requests.post(f"{base}/knowledge/import", files={"file": ("test_company.json", f, "application/json")})
print("Import Status:", r.status_code)
print("Import Response:", json.dumps(r.json(), indent=2))

print("\n--- GET /knowledge (Company B) ---")
r = requests.get(f"{base}/knowledge")
print("Status:", r.status_code)
print(json.dumps(r.json(), indent=2))

print("\n--- Questions for Company B ---")
results_b = []
for q in questions:
    resp = requests.post(f"{base}/ask", json={"question": q}).json()
    print(f"Q: {q}")
    print(f"Answer: {resp['answer']}")
    print(f"Retrieved Data: {resp['retrieved_data']}")
    print(f"Source: {resp['source']}")
    print("-" * 40)
    results_b.append((q, resp['answer'], resp['retrieved_data']))

print("\n====================================")
print("STEP 3: GROUNDING / HALLUCINATION TEST")
print("====================================")
ceo_q = "What is the CEO's name?"
resp_ceo = requests.post(f"{base}/ask", json={"question": ceo_q}).json()
print(f"Q: {ceo_q}")
print(f"Answer: {resp_ceo['answer']}")
print(f"Retrieved Data: {resp_ceo['retrieved_data']}")
print(f"Intent: {resp_ceo['intent']}")
