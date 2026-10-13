# RAGless Web UI Documentation

## 1. Overview
The RAGless Web UI provides an honest, transparent single-page interface served directly by FastAPI at `/ui`. It makes the agent's internal planning, tool executions, source provenance, and deterministic verification immediately visible to users.

### Architectural Principles
- **Zero Build Step**: Pure vanilla HTML5, CSS3, and JavaScript. No npm, Webpack, Vite, or bundle steps.
- **Zero External CDNs**: Local system font stack (`system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif`) and inline SVG icons. Zero external network resources or font requests.
- **Strict Content-Security-Policy (CSP)**: `default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self';` (no `'unsafe-inline'` or `'unsafe-eval'`).
- **Complete XSS Immunity**: Dynamic data from LLM responses, web snippets, and database rows is rendered strictly using `textContent` and `createElement`—never `innerHTML` or `outerHTML`.
- **Honest Trace**: All timeline steps, tool executions, provenance labels, and verification checks reflect real server response data. No artificial streaming or simulated delays.
- **SSE Decision**: Server-Sent Events (`/agent/ask/stream`) were **skipped** to preserve the validated, deterministic security boundaries and per-run taint context of `AgentOrchestrator` without introducing concurrency risks. Instead, the UI displays an honest "Working..." state with a live client-side elapsed timer, followed by the complete verified trace upon response arrival.

---

## 2. Visual Direction & Design System

The visual design is inspired by a **bougainvillea-and-blue-sky** palette, featuring natural cream backgrounds, subtle sun-glow gradients, deep sky-blue interactive elements, and delicate bougainvillea-pink accents.

### 2.1 Color Tokens & CSS Variables

All colors and spacing are declared as CSS custom properties in `:root` and `@media (prefers-color-scheme: dark)`.

| Token | Light Theme | Dark Theme | Purpose |
| :--- | :--- | :--- | :--- |
| `--bg-page` | `#F8F6F3` | `#12201A` | Canvas background with radial sun glow & sky/blush gradient |
| `--bg-card` | `#FFFFFF` | `#182C24` | Primary card container surface |
| `--bg-card-subtle` | `#F3EFEA` | `#1F362C` | Secondary container surface, inputs, step rows |
| `--primary` | `#1A78C2` | `#58A6E6` | Main brand sky-blue for Run Agent CTA, focus rings, step badges |
| `--text-main` | `#1C2F26` | `#F1F5F3` | Primary body and heading text |
| `--text-muted` | `#415B4E` | `#A2BFB2` | Subtitles, helper text, and secondary labels |
| `--accent-pink` | `#E58FB4` | `#F2A9C6` | Bougainvillea accent for progress bar fill & action highlights |
| `--bg-chip` | `#FBE9F0` | `#271B23` | Suggestion chips fill |
| `--border-chip` | `#C05D87` | `#B06588` | Suggestion chips border (3.5:1+ UI component contrast) |
| `--success` | `#3F7D4E` | `#68C47F` | Success indicators, verified badges, completed tasks |
| `--warning` | `#D9703F` | `#F29C6D` | Review badges, warning alerts, mock data indicator |
| `--danger` | `#D32F2F` | `#F58585` | Execution errors and failed check alerts |

### 2.2 WCAG AA Contrast Verification

Every text and interactive element was mathematically evaluated for WCAG AA conformance (4.5:1 for body text, 3:1 for large text and UI component borders):

| Element & Pair | Contrast Ratio | WCAG Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Light: Body text** (`#1C2F26`) on Card Surface (`#FFFFFF`) | **14.15:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Light: Body text** (`#1C2F26`) on Page Background (`#F8F6F3`) | **13.12:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Light: Body text** (`#1C2F26`) on Suggestion Chip (`#FBE9F0`) | **12.14:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Light: Muted text** (`#415B4E`) on Card Surface (`#FFFFFF`) | **7.42:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Light: Button text** (`#FFFFFF`) on Primary Button (`#1A78C2`) | **4.66:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Light: Primary link** (`#115E9B`) on Card Surface (`#FFFFFF`) | **6.78:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Light: Success text** (`#286036`) on Success Subtle (`#EAF4ED`) | **6.63:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Light: Warning text** (`#943E15`) on Warning Subtle (`#FDF0EA`) | **6.36:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Light: Danger text** (`#A0272B`) on Danger Subtle (`#FCEEEE`) | **6.54:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Light: Pink UI border** (`#C05D87`) on Blush Chip (`#FBE9F0`) | **3.49:1** | UI border ≥ 3.0:1 | **PASS** (AA) |
| **Light: Pink UI border** (`#C05D87`) on Card Surface (`#FFFFFF`) | **4.07:1** | UI border ≥ 3.0:1 | **PASS** (AA) |
| **Dark: Body text** (`#F1F5F3`) on Card Surface (`#182C24`) | **13.40:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Body text** (`#F1F5F3`) on Page Background (`#12201A`) | **15.31:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Muted text** (`#A2BFB2`) on Card Surface (`#182C24`) | **7.46:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Button text** (`#0E1B15`) on Primary Button (`#58A6E6`) | **6.75:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Primary link** (`#72B6ED`) on Card Surface (`#182C24`) | **6.76:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Success text** (`#68C47F`) on Card Surface (`#182C24`) | **6.87:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Warning text** (`#F29C6D`) on Card Surface (`#182C24`) | **6.85:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Danger text** (`#F58585`) on Card Surface (`#182C24`) | **6.02:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Pink UI border** (`#B06588`) on Dark Chip (`#271B23`) | **4.00:1** | UI border ≥ 3.0:1 | **PASS** (AA) |
| **Dark: Pink UI border** (`#B06588`) on Card Surface (`#182C24`) | **3.55:1** | UI border ≥ 3.0:1 | **PASS** (AA) |
| **Dark: Accent pink text** (`#F3AEC9`) on Card Surface (`#182C24`) | **8.22:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |

---

## 3. Interface States & Components

### 3.1 Initial Empty State
Before the first execution, the results region displays a dedicated **Empty State Card** (`#empty-state-card`) highlighting architectural guarantees (Parameterized SQLite, Deterministic Verifier, Taint Isolation) rather than a blank canvas.

### 3.2 Loading State with Live Timer
When the user submits an objective, `#working-card` appears with an animated spinner and a client-side live counter (`0.0s`), providing immediate feedback while the agent executes multi-step plans.

### 3.3 Intentional Grounded Refusal State
When a query asks about unrecorded company facts (e.g., "Who is the head of Engineering?"), the system deterministically flags missing evidence. The UI presents an intentional **Refusal Banner** (`#refusal-banner`) stating that the information is not present in verified database records, avoiding ambiguity or confusion with backend crashes.

### 3.4 Error State
If network or validation errors occur, `#error-card` displays an accessible alert with the exact HTTP error code and message.

### 3.5 Objective & Suggestion Chips
- **User ID Field**: Configurable user identifier (`demo-user`), annotated clearly as a demo identifier without multi-tenant authentication.
- **Suggestion Chips**: One-click quick prompts in blush with pink borders:
  - "Prepare everything I need for my first week"
  - "What do I still need to do?"
  - "Create tasks for my onboarding"
  - "How does our annual leave compare to the UK statutory minimum?"
- **Run Agent Button**: Primary sky-blue button with keyboard focus ring and active states.

### 3.6 Answer Card & Verification
- **Grounded Answer**: Displays the final answer.
- **Verification Badge**: Shows `Verified` (green) if all assertions passed, or `Needs Review` (amber) with plain-language unresolved items.

### 3.7 Agent Activity Trace
- **Plan & Execution Timeline**: Each plan step shows the tool name, source provenance (`sqlite` vs `tavily`), execution status badge (`Success` / `Failed`), and step details.
- **Verification Checks Grid**: Lists all individual deterministic assertions (e.g., `answer_grounded`, `numbers_grounded` including accepted derived mathematical operations such as `4 = 28 - 24`, and `write_confirmed`).

### 3.8 Collapsible Evidence Panel
- **Web Sources**: External citations rendered as secure links (`rel="noopener noreferrer" target="_blank"`), with snippets quoted as untrusted data blocks.
- **Internal Knowledge**: Tabular display of retrieved corporate facts.
- **MOCK DATA Badge**: Displayed whenever mock search client responses are present in tool results.

### 3.9 Actions & State Panels
- **State Modifications**: Summarizes confirmed memory writes and task changes.
- **Tasks Panel**: Refreshed via `GET /tasks/{user_id}` after every execution, displaying open and completed tasks.
- **Progress Panel**: Refreshed via `GET /memory/{user_id}`, displaying the onboarding checklist completion count and animated pink progress bar.

### 3.10 Footer Stats
- Displays end-to-end execution latency (ms), LLM call count, web call count, and active provider/model ID.

---

## 4. Manual Verification & Visual Inspection

To inspect the redesigned interface visually:

1. **Launch the FastAPI server**:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
   ```
2. **Open in browser**:
   Navigate to `http://127.0.0.1:8000/ui/`.
3. **Verify Light & Dark themes**:
   - In light mode, observe the `#F8F6F3` background with the soft top-left radial sun glow and sky-to-blush gradient, `#1A78C2` primary button, and blush suggestion chips with pink borders.
   - Switch your OS preference to dark mode to observe the deep pine `#12201A` background, `#182C24` card surfaces, and muted pink highlights.
4. **Verify Empty State**:
   - Confirm `#empty-state-card` is rendered with feature tags before any prompt is submitted.
5. **Verify Live Working Timer**:
   - Click a suggestion chip and click `Run Agent`.
   - Observe the loading card with live elapsed timer (`0.0s`, `0.1s`, etc.).
6. **Verify Grounded Refusal**:
   - Submit: `Who is the head of Engineering?`
   - Observe the intentional Refusal Banner indicating that the department head is not recorded in verified database records.
