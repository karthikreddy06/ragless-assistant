import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
import requests

from app.database import (
    get_current_company_name,
    get_full_knowledge,
)

logger = logging.getLogger(__name__)

# Configurable via environment variables with defaults
OLLAMA_DEFAULT_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))

# Cache resolved working URL to avoid repeated latency
_WORKING_URL: Optional[str] = None


class OllamaError(Exception):
    """Base exception for Ollama interactions."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when Ollama server cannot be reached."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when the specified model is not installed in Ollama."""
    pass


def get_base_url() -> str:
    """
    Returns the resolved Ollama base URL.
    Checks configured URL (default http://localhost:11434). If running on Windows
    where 'localhost' might route to IPv6 WSL/Docker without the model, automatically
    verifies IPv4 127.0.0.1 fallback.
    """
    global _WORKING_URL
    if _WORKING_URL is not None:
        return _WORKING_URL

    configured = OLLAMA_DEFAULT_URL.rstrip("/")
    candidates = [configured]

    # If localhost is configured, also consider 127.0.0.1 as a candidate
    if "localhost" in configured:
        candidates.append(configured.replace("localhost", "127.0.0.1"))

    for candidate in candidates:
        try:
            resp = requests.get(f"{candidate}/api/tags", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                # If target model is in this instance, prioritize it immediately
                if any(OLLAMA_MODEL in m for m in models):
                    _WORKING_URL = candidate
                    return _WORKING_URL
                # Otherwise keep as working if no better candidate found
                if _WORKING_URL is None:
                    _WORKING_URL = candidate
        except Exception:
            pass

    if _WORKING_URL is not None:
        return _WORKING_URL

    # Default to configured URL even if currently unreachable
    return configured


def check_ollama_health() -> Dict[str, Any]:
    """
    Check if Ollama server is up and verify if the configured model is installed.
    Does not crash on failure; returns structured health status.
    """
    base_url = get_base_url()
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            model_present = any(OLLAMA_MODEL in m for m in models)
            return {
                "available": True,
                "base_url": base_url,
                "model": OLLAMA_MODEL,
                "model_installed": model_present,
                "installed_models": models,
            }
        return {
            "available": False,
            "base_url": base_url,
            "model": OLLAMA_MODEL,
            "error": f"Ollama returned HTTP status {resp.status_code}",
        }
    except requests.exceptions.RequestException as e:
        return {
            "available": False,
            "base_url": base_url,
            "model": OLLAMA_MODEL,
            "error": f"Failed to connect to Ollama: {str(e)}",
        }


def _call_ollama_generate(prompt: str, json_format: bool = False, temperature: float = 0.0) -> str:
    """Internal helper to invoke Ollama's /api/generate endpoint."""
    base_url = get_base_url()
    url = f"{base_url}/api/generate"

    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
        "options": {
            "temperature": temperature,
        },
    }

    if json_format:
        payload["format"] = "json"

    import time as _time
    t0 = _time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    except requests.exceptions.ConnectionError as e:
        raise OllamaConnectionError(
            f"Could not connect to Ollama at {base_url}. Ensure Ollama is running."
        ) from e
    except requests.exceptions.Timeout as e:
        raise OllamaConnectionError(
            f"Ollama request timed out after {OLLAMA_TIMEOUT} seconds."
        ) from e
    except requests.exceptions.RequestException as e:
        raise OllamaConnectionError(f"Ollama HTTP communication error: {str(e)}") from e

    if response.status_code == 404:
        raise OllamaModelNotFoundError(
            f"Model '{OLLAMA_MODEL}' not found in Ollama at {base_url}. Please run: ollama pull {OLLAMA_MODEL}"
        )
    if response.status_code != 200:
        raise OllamaError(f"Ollama returned error status {response.status_code}: {response.text}")

    latency_ms = round((_time.perf_counter() - t0) * 1000, 2)
    try:
        from app.agent.executor import current_run_context
        ctx = current_run_context.get()
        if ctx is not None:
            ctx.record_llm_call(latency_ms)
    except Exception:
        pass

    data = response.json()
    return data.get("response", "").strip()


def _build_dynamic_knowledge_context() -> str:
    """Build a context summary from the currently active SQLite database records."""
    knowledge = get_full_knowledge()
    company = knowledge.get("company", {})
    name = company.get("name", "the company")
    location = company.get("location", "unknown")
    employees = company.get("employees", "unknown")
    departments = knowledge.get("departments", [])
    leave_policy = knowledge.get("leave_policy", [])

    dept_lines = "\n".join(f"   - {d['name']}: {d['employees']} employees" for d in departments) or "   - None"
    leave_lines = "\n".join(f"   - {l['leave_type']}: {l['days']} days" for l in leave_policy) or "   - None"

    return f"""The database ONLY has facts about {name}:
1. Company Information:
   - name: {name}
   - location: {location}
   - employees: {employees}
2. Departments:
{dept_lines}
3. Leave Policy:
{leave_lines}"""


def generate_query_plan(question: str) -> Dict[str, Any]:
    """
    Prompt the local LLM (qwen2.5:3b) to decompose the user's natural language question
    into a structured query plan containing 1 to 5 predefined retrieval operations.
    Returns a dict with 'queries': List[Dict[str, Any]].
    """
    knowledge_context = _build_dynamic_knowledge_context()

    planning_prompt = f"""You are a strict query planner for a company knowledge database.
{knowledge_context}

Allowed operations:
- intent: "company_info", query_type: "company_name" | "company_location" | "company_employees" | "all_company"
- intent: "department_employees", query_type: "dept_headcount" (entity: department name) | "all_depts" | "total_dept_headcount"
- intent: "leave_policy", query_type: "leave_days" (entity: leave type) | "all_leaves"
- intent: "unknown", query_type: "unknown", entity: null

Rules:
1. If the question asks about multiple topics, generate a separate query operation for each topic in the 'queries' list.
2. If the question asks about multiple departments (e.g. AI and Backend), output a separate operation for each department.
3. If the question mentions both supported database topics and unsupported topics (e.g. CEO name, revenue, stock price), include operations for the supported topics.
4. If NONE of the requested topics exist in the database, return a single operation with intent "unknown", query_type "unknown", entity null.
5. Output JSON with format:
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

    from app.agent.llm import get_llm_client
    try:
        client = get_llm_client()
        raw_response = client.generate_json(planning_prompt, temperature=0.0)
    except Exception as e:
        logger.warning("Error calling active LLM client for query plan: %s; falling back to direct Ollama.", e)
        raw_response = _call_ollama_generate(planning_prompt, json_format=True, temperature=0.0)

    try:
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict) and "queries" in parsed and isinstance(parsed["queries"], list):
            return parsed
        # If model returned a single object instead of {"queries": [...]}
        if isinstance(parsed, dict) and "intent" in parsed:
            return {"queries": [parsed]}
        return {"queries": [{"intent": "unknown", "query_type": "unknown", "entity": None}]}
    except json.JSONDecodeError:
        logger.warning("Failed to decode JSON from query planner LLM response: %s", raw_response)
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict) and "queries" in parsed and isinstance(parsed["queries"], list):
                    return parsed
                if isinstance(parsed, dict) and "intent" in parsed:
                    return {"queries": [parsed]}
            except Exception:
                pass
        return {"queries": [{"intent": "unknown", "query_type": "unknown", "entity": None}]}


def classify_question_intent(question: str) -> Dict[str, Any]:
    """
    Backward-compatible helper: extracts the primary/first intent from the generated query plan.
    """
    plan_dict = generate_query_plan(question)
    queries = plan_dict.get("queries", [])
    if queries and isinstance(queries[0], dict):
        return {
            "intent": queries[0].get("intent", "unknown"),
            "query_type": queries[0].get("query_type", "unknown"),
            "entity": queries[0].get("entity"),
        }
    return {"intent": "unknown", "query_type": "unknown", "entity": None}


def generate_grounded_answer(
    question: str,
    retrieved_data: Any,
    intent: Optional[str] = None
) -> str:
    """
    Pass retrieved structured SQLite data and user question to the local LLM
    to generate a polite, clear, natural-language response strictly grounded in the data.
    Enforces multi-table reasoning and explicit refusal for unavailable/missing facts.
    """
    company_name = get_current_company_name()

    if retrieved_data is None or (isinstance(retrieved_data, list) and len(retrieved_data) == 0):
        return (
            f"I could not find relevant information in the company database. "
            f"I can answer questions regarding {company_name}'s company details (name, location, total employees), "
            f"department headcounts, leave policies, and onboarding checklist items."
        )

    synthesis_prompt = f"""You are a professional knowledge assistant for {company_name}.
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
{json.dumps(retrieved_data, indent=2)}

Answer:"""

    from app.agent.llm import get_llm_client
    try:
        client = get_llm_client()
        answer = client.generate_text(synthesis_prompt, temperature=0.1)
    except Exception as e:
        logger.warning("Error calling active LLM client for grounded answer: %s; falling back to direct Ollama.", e)
        answer = _call_ollama_generate(synthesis_prompt, json_format=False, temperature=0.1)
    return answer.strip()
