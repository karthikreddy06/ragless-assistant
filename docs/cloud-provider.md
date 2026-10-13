# Cloud Provider: OpenAI-Compatible Client (Nebius / NVIDIA Models)

## 1. Overview

The **RAGless Assistant** includes an OpenAI-compatible cloud provider client (`NebiusClient`), unit-tested with mocks only. The active LLM provider is switchable via environment configuration (`LLM_PROVIDER=ollama` or `LLM_PROVIDER=nebius`), routing both the outer agent planner, inner knowledge retrieval operations, and grounded answer synthesis through the configured client.

---

## 2. Architecture & Provider Abstraction

```text
                     [ get_llm_client() ]
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
       [ OllamaClient ]              [ NebiusClient ]
       (Local qwen2.5:3b)      (OpenAI-compatible / mock-tested)
       http://localhost:11434   https://api.tokenfactory.nebius.com/v1/
               │                             │
               └──────────────┬──────────────┘
                              ▼
        ┌──────────────────────────────────────────┐
        │ 1. Agent Planner (JSON query plan)       │
        │ 2. Inner Query Planner (company_knowledge│
        │ 3. Grounded Answer Synthesis             │
        └──────────────────────────────────────────┘
```

Both clients implement the common abstract base class [`LLMClient`](../app/agent/llm.py):
- `provider_name -> str`
- `model_name -> str`
- `generate_text(prompt, system, temperature) -> str`
- `generate_json(prompt, system, temperature) -> str`

---

## 3. Configuration & How to Switch Providers

Configuration is controlled entirely via environment variables (or `.env` file). See [`.env.example`](../.env.example) for a template.

### To Run with Local Ollama (Default):
```bash
export LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen2.5:3b
```

### To Run with OpenAI-Compatible Cloud Endpoint (Nebius / Other):
```bash
export LLM_PROVIDER=nebius
export NEBIUS_API_KEY="your-token-here"
export NEBIUS_BASE_URL="https://api.tokenfactory.nebius.com/v1/"
export NEBIUS_MODEL="nvidia/nemotron-3-super-120b-a12b"
export LLM_TIMEOUT_SECONDS=30
export LLM_MAX_RETRIES=3
export LLM_TEMPERATURE=0.0
```

---

## 4. Key Architectural Implementations

### 4.1 OpenAI Compatibility Without Heavy Dependencies
[`NebiusClient`](../app/agent/llm.py) connects directly to standard OpenAI-compatible endpoints (`/chat/completions`) using Python's standard `requests` library. No additional heavy SDKs or package upgrades are required.

### 4.2 Reasoning Token Stripping (`<think>...</think>`)
Certain reasoning models may output internal reasoning traces wrapped in `<think>` and `</think>` tags.
The function [`strip_reasoning()`](../app/agent/llm.py) strips these blocks before JSON parsing and before returning final text, guaranteeing:
- JSON plan parsing is never corrupted by reasoning preambles.
- Internal chain-of-thought traces never leak into user-facing responses.

### 4.3 Structured Output (`response_format`)
When `generate_json()` is invoked, `NebiusClient` supplies `response_format={"type": "json_object"}` to the completions payload, paired with JSON schema prompting and regex fence extraction fallbacks.

### 4.4 Secret Redaction & Zero Credential Leaks
API keys are strictly shielded:
- [`redact_secrets()`](../app/agent/llm.py) masks any occurrence of the API key with `[REDACTED]` in error strings and logs.
- Dedicated unit test in [`TestSecretRedaction`](../tests/test_nebius.py) asserts that keys never appear in exceptions or responses.
- `.env` is ignored in `.gitignore`.

### 4.5 Resilient HTTP Retries & Backoff
`NebiusClient` handles transient rate limits (HTTP 429) and gateway errors (HTTP 5xx) with exponential backoff up to `LLM_MAX_RETRIES` (default: 3).

### 4.6 Intent Preservation
To prevent intent drift (e.g., rewriting *"Who is the head of Engineering?"* into *"headcount Engineering"*), the planner prompt includes explicit anti-drift instructions to preserve the user's exact semantic intent and roles.

---

## 5. Additive Diagnostics & Evidence

1. **`GET /health`**: Additively reports `llm_provider`, `llm_model`, and `llm_healthy` without exposing keys or credentials.
2. **`POST /agent/ask`**: Returns `llm_provider`, `llm_model`, and `llm_calls_count` in [`AgentResponse`](../app/agent/schemas.py) for transparent per-request verification.
