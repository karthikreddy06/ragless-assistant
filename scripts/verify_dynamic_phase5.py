import json
import requests

BASE_URL = "http://127.0.0.1:8000"

print("=" * 70)
print("PHASE 5 DYNAMIC KNOWLEDGE REGRESSION TEST")
print("=" * 70)

# Step 1: Ingest TechNova
print("\n[Step 1] Ingesting TechNova knowledge...")
with open("data/sample_knowledge.json", "rb") as f:
    resp = requests.post(f"{BASE_URL}/knowledge/import", files={"file": ("sample_knowledge.json", f, "application/json")})
assert resp.status_code == 200, f"Import failed: {resp.text}"
print("TechNova imported successfully:", resp.json())

# Step 2: Multi-table query on TechNova
question = "How many employees are in AI and how many annual leave days are available?"
print(f"\n[Step 2] Asking: '{question}'")
resp = requests.post(f"{BASE_URL}/ask", json={"question": question})
assert resp.status_code == 200, f"Ask failed: {resp.text}"
res_technova = resp.json()
print("TechNova Answer:", res_technova["answer"])
print("TechNova Retrieved:", res_technova["retrieved_data"])

assert "80" in res_technova["answer"], f"Expected 80 in answer: {res_technova['answer']}"
assert "24" in res_technova["answer"], f"Expected 24 in answer: {res_technova['answer']}"
print("TechNova Multi-Table Verification: PASSED (AI: 80, Annual Leave: 24)")

# Step 3: Ingest NextGen AI
print("\n[Step 3] Ingesting NextGen AI knowledge...")
with open("data/test_company.json", "rb") as f:
    resp = requests.post(f"{BASE_URL}/knowledge/import", files={"file": ("test_company.json", f, "application/json")})
assert resp.status_code == 200, f"Import failed: {resp.text}"
print("NextGen AI imported successfully:", resp.json())

# Step 4: Multi-table query on NextGen AI
print(f"\n[Step 4] Asking: '{question}'")
resp = requests.post(f"{BASE_URL}/ask", json={"question": question})
assert resp.status_code == 200, f"Ask failed: {resp.text}"
res_nextgen = resp.json()
print("NextGen AI Answer:", res_nextgen["answer"])
print("NextGen AI Retrieved:", res_nextgen["retrieved_data"])

assert "200" in res_nextgen["answer"], f"Expected 200 in answer: {res_nextgen['answer']}"
assert "30" in res_nextgen["answer"], f"Expected 30 in answer: {res_nextgen['answer']}"
print("NextGen AI Multi-Table Verification: PASSED (AI: 200, Annual Leave: 30)")

print("\n" + "=" * 70)
print("DYNAMIC KNOWLEDGE REGRESSION TEST RESULT: ALL PASSED")
print("=" * 70)
