# RAGless Web UI Documentation

## 1. Overview
The RAGless Web UI provides an honest, transparent single-page interface served directly by FastAPI at `/ui`. It makes the agent's internal planning, tool executions, source provenance, and deterministic verification immediately visible to users.

### Architectural Principles
- **Zero Build Step**: Pure vanilla HTML5, CSS3, and JavaScript. No npm, Webpack, Vite, or bundle steps.
- **Zero External CDNs**: Local system font stack (`system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif`) and inline SVG icons. Zero external network resources.
- **Strict Content-Security-Policy (CSP)**: `default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self';` (no `'unsafe-inline'` or `'unsafe-eval'`).
- **Complete XSS Immunity**: Dynamic data from LLM responses, web snippets, and database rows is rendered strictly using `textContent` and `createElement`—never `innerHTML` or `outerHTML`.
- **Honest Trace**: All timeline steps, tool executions, provenance labels, and verification checks reflect real server response data. No artificial streaming or simulated delays.
- **SSE Decision**: Server-Sent Events (`/agent/ask/stream`) were **skipped** to preserve the validated, deterministic security boundaries and per-run taint context of `AgentOrchestrator` without introducing concurrency risks. Instead, the UI displays an honest "Working..." state with a live client-side elapsed timer, followed by the complete verified trace upon response arrival.

---

## 2. Interface Components

### 2.1 Header & Live Status
- **Title**: RAGless · Autonomous Company Operations Agent.
- **Status Pill**: Fetches `GET /health` on initialization and displays the active provider and model (e.g., `ollama · qwen2.5:3b`). It dynamically reflects actual provider availability and never displays Nebius/Nemotron unless live reported.

### 2.2 Objective & Goal Box
- **User ID Field**: Configurable user identifier (defaults to `demo-user`), annotated clearly as a demo identifier without multi-tenant authentication.
- **Suggestion Chips**: One-click quick prompts:
  - "Prepare everything I need for my first week"
  - "What do I still need to do?"
  - "Create tasks for my onboarding"
  - "How does our annual leave compare to the UK statutory minimum?"
- **Run Agent Button**: Triggers `POST /agent/ask`, displays live elapsed timer, and handles client-side 150s timeouts.

### 2.3 Answer Card & Verification
- **Grounded Answer**: Displays the final answer.
- **Verification Badge**: Shows `Verified` (green) if all deterministic assertions passed, or `Needs Review` (amber) with plain-language unresolved items.

### 2.4 Agent Activity Trace
- **Plan & Execution Timeline**: Each plan step shows the tool name, source provenance (`sqlite` vs `tavily`), execution status badge (`Success` / `Failed`), and step details.
- **Verification Checks Grid**: Lists all individual deterministic assertions (e.g., `answer_grounded`, `numbers_grounded` including accepted derived mathematical operations such as `4 = 28 - 24`, and `write_confirmed`).

### 2.5 Collapsible Evidence Panel
- **Web Sources**: External citations rendered as secure links (`rel="noopener noreferrer" target="_blank"`), with snippets quoted as untrusted data blocks.
- **Internal Knowledge**: Tabular display of retrieved corporate facts.

### 2.6 Actions & State Panels
- **State Modifications**: Summarizes confirmed memory writes and task changes.
- **Tasks Panel**: Refreshed via `GET /tasks/{user_id}` after every execution, displaying open and completed tasks.
- **Progress Panel**: Refreshed via `GET /memory/{user_id}`, displaying the onboarding checklist completion count and animated progress bar.

### 2.7 Footer Stats
- Displays end-to-end execution latency (ms), LLM call count, web call count, and active provider/model ID.

---

## 3. Manual Test Script

To manually verify the UI in a browser:

1. **Launch the server**:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
   ```
2. **Open browser**:
   Navigate to `http://localhost:8000/ui`.
3. **Verify Header Status**:
   - Status pill should turn green with label `ollama · qwen2.5:3b`.
4. **Test Task Creation Workflow**:
   - Click chip: `Create tasks for my onboarding`.
   - Click `Run Agent`.
   - Observe "Working..." timer.
   - Verify Answer card shows confirmed task creation.
   - Verify Actions panel shows 7 confirmed `create_tasks_from_onboarding` items.
   - Verify Tasks panel in the sidebar updates to show 7 tasks.
5. **Test Grounded Information Query**:
   - Enter: `How many employees are in the AI department?`
   - Click `Run Agent`.
   - Verify Answer shows `80` employees.
   - Verify Activity Trace shows `company_knowledge` with source `sqlite`.
6. **Test XSS Inertness**:
   - Enter: `<script>alert('xss')</script><img src=x onerror=alert(1)>`
   - Observe that the input is safely rejected by security pre-filter or rendered inertly as visible text characters without executing any script.

---

## 4. Screenshot Instructions

*Note: Screenshots must be captured directly from live browser sessions and never fabricated or simulated.*

To capture genuine screenshots for documentation:
1. Ensure the backend is running with local Ollama (`qwen2.5:3b`).
2. Open `http://localhost:8000/ui` in Chrome, Firefox, or Edge.
3. Capture the following screens:
   - `docs/screenshots/01_ui_initial_state.png`: Initial page load showing status pill, chips, and sidebar panels.
   - `docs/screenshots/02_ui_tasks_creation.png`: Completed run for "Create tasks for my onboarding" showing the activity trace and updated tasks sidebar.
   - `docs/screenshots/03_ui_verification_trace.png`: Close-up of Activity Trace showing individual verification checks.
4. Save screenshots into `docs/screenshots/`.
