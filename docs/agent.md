# Agent Architecture & Security Specification

## 1. Overview

The **RAGless Agent** extends the deterministic SQLite knowledge assistant with an agentic lifecycle:
**Understand → Plan → Validate → Execute → Observe → Verify → Respond**.

Unlike traditional vector RAG or unconstrained tool-calling agents:
- The system operates without vector databases or semantic similarity embeddings.
- Relational SQLite serves as the deterministic single source of truth.
- The LLM **never** writes raw SQL and cannot access arbitrary tools.
- Verification is performed deterministically in Python with **zero extra LLM calls**.

---

## 2. Agent Architecture

```text
User Question
      │
      ▼
[1. Orchestrator] ──► Pre-validate Input (reject SQL injection / dangerous tokens)
      │
      ▼
[2. Planner] ───────► Prompt LLM (qwen2.5:3b) to generate JSON plan
      │
      ▼
[3. Validator] ─────► Security Boundary (allowlist, step count, SQL filters, extra="forbid")
      │
      ├── (Invalid Plan) ──► Replan (max 1 attempt) or Structured Refusal
      │
      ▼ (Valid Plan)
[4. Executor] ──────► Execute validated steps sequentially via Tool Registry
      │               └── Tool 'company_knowledge' ──► Parameterized SQLite queries
      ▼
[5. Synthesizer] ───► Grounded answer synthesis using ONLY retrieved evidence
      │
      ▼
[6. Verifier] ──────► Deterministic Python checks (step execution, grounding, safety)
      │
      ▼
AgentResponse (JSON with question, answer, plan, tool_results, verification, latency)
```

---

## 3. Core Components

### 3.1 LLM Provider Abstraction (`app/agent/llm.py`)
- Defines abstract base class `LLMClient` with `generate_text()` and `generate_json()`.
- Implements `OllamaClient` reusing existing local configuration (`http://localhost:11434`, `qwen2.5:3b`).
- Provides factory `get_llm_client()` driven by `LLM_PROVIDER` environment variable (default: `"ollama"`). Designed to allow drop-in addition of cloud providers without touching agent logic.

### 3.2 Strict Pydantic Schemas (`app/agent/schemas.py`)
All models enforce `model_config = ConfigDict(extra="forbid")`:
- **`PlanStep`**: `{tool: str, input: str}` with length bounds.
- **`AgentPlan`**: `{goal: str, steps: list[PlanStep]}` (enforces 1 to 8 steps).
- **`ToolResult`**: `{tool: str, success: bool, source: "sqlite", data: Any, error: Optional[str]}`.
- **`Verification`**: `{passed: bool, checks: list[CheckItem], unresolved: list[str]}`.
- **`AgentRequest`**: `{question: str}` (1 to 500 characters, whitespace stripped).
- **`AgentResponse`**: `{question, answer, plan, tool_results, verification, processing_time_ms}`.

### 3.3 Planner (`app/agent/planner.py`)
- Prompts the LLM to output a JSON task plan specifying sequential sub-questions.
- Explicitly informs the model of the sole permitted tool (`company_knowledge`) and strictly forbids SQL generation.
- Handles markdown code fence stripping and malformed JSON recovery.
- Produces an unvalidated plan; the planner never executes tools and never produces SQL.

### 3.4 Tool Validator — The Security Boundary (`app/agent/validator.py`)
Treats LLM output as untrusted user input:
- **Tool Allowlist**: Only `ALLOWED_TOOLS = {"company_knowledge"}` is permitted. Invented tools (e.g. `execute_sql`, `run_command`, `web_search`) are rejected immediately.
- **Schema Enforcement**: Rejects unexpected or extra fields.
- **Step Limits**: Enforces `1 <= steps <= 8`.
- **Injection Pattern Filtering**: Scans step inputs and questions for SQL injection fragments (`;`, `--`, `/*`, `DROP`, `SELECT`, `OR '1'='1'`) and scripting tags.
- Raises typed `PlanValidationError`.

### 3.5 Tool Executor (`app/agent/executor.py`)
- Central registry mapping tool names to deterministic Python functions.
- `company_knowledge` wraps the existing RAGless query engine (`process_query`).
- Unsupported entities (e.g. `Engineering` department, `CEO`) return `success=True` with `data=None` rather than crashing, enabling the assistant to cleanly refuse without hallucination.

### 3.6 Deterministic Verifier (`app/agent/verifier.py`)
Runs 100% in Python without additional LLM calls:
1. **`all_steps_executed`**: Verifies every planned step has a matching successful `ToolResult`.
2. **`grounded_refusal_when_empty`**: When tool evidence is null or empty, ensures the answer acknowledges unavailability and contains no hallucinated numbers.
3. **`evidence_fact_grounding`**: When evidence is present, verifies that numbers in the answer match retrieved database facts.
4. **`security_no_internal_leak`**: Ensures internal SQL table names or database delimiters are not leaked in the answer.
5. Populates `unresolved` with missing facts or execution failures.

### 3.7 Orchestrator (`app/agent/orchestrator.py`)
- Coordinates the complete lifecycle.
- Caps step count at `MAX_AGENT_STEPS = 8` and replan attempts at `MAX_REPLAN_ATTEMPTS = 1` to prevent infinite loops.
- Reuses grounding prompts ensuring answers rely strictly on retrieved JSON data.
- Traps connection, model, or tool errors, returning structured `AgentResponse` payloads with `verification.passed = False` rather than unhandled 500 exceptions.

---

## 4. API Endpoints

- **`POST /agent/ask`**: Primary agent endpoint taking `AgentRequest` and returning `AgentResponse`.
- **Existing Endpoints Preserved**:
  - `GET /`: Root metadata including all available endpoints.
  - `GET /health`: SQLite connection and Ollama health checks.
  - `GET /knowledge`: Structured view of active corporate knowledge.
  - `POST /knowledge/import`: Dynamic JSON knowledge ingestion.
  - `POST /ask`: Legacy direct factual Q&A endpoint.

---

## 5. Security Checklist

| Threat / Risk | Defense Mechanism | Verified |
|---|---|:---:|
| Invented Tools (e.g. `execute_sql`) | Strict `ALLOWED_TOOLS = {"company_knowledge"}` allowlist | YES |
| Schema Poisoning / Extra Fields | Pydantic `ConfigDict(extra="forbid")` on all models | YES |
| SQL Injection in Questions or Plans | Regex pattern filtering + Parameterized SQLite (`?`) queries | YES |
| Prompt Injection / System Override | Grounding prompt isolated to JSON evidence; input pre-validation | YES |
| Infinite Loop / Denial of Service | Step cap (`MAX_AGENT_STEPS = 8`) and replan cap (`MAX_REPLAN = 1`) | YES |
| Secret / Stack Trace Leakage | Structured exception handling; sensitive data omitted | YES |
