# RAGless Web UI Documentation

## 1. Overview
The RAGless Web UI provides an honest, transparent single-page interface served directly by FastAPI at `/ui`. It is designed with a modern, professional, enterprise-SaaS look (Linear, Vercel, and Stripe dashboard aesthetic), making internal agent planning, deterministic verifications, source provenance, and live execution metrics immediately legible.

### Architectural Principles
- **Zero Build Step**: Pure vanilla HTML5, CSS3, and JavaScript. No npm, Webpack, Vite, or bundle steps.
- **Zero External CDNs**: System font stack (`system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`) and inline SVG icons. Zero external network resources or web font requests.
- **Strict Content-Security-Policy (CSP)**: `default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self';` (no `'unsafe-inline'` or `'unsafe-eval'`).
- **Complete XSS Immunity**: Dynamic data from LLM responses, web snippets, and database rows is rendered strictly using `textContent` and `createElement`—never `innerHTML` or `outerHTML`.
- **Honest Trace**: All timeline steps, tool executions, provenance labels, and verification checks reflect real server response data. No artificial streaming or simulated delays.
- **SSE Decision**: Server-Sent Events (`/agent/ask/stream`) were **skipped** to preserve the validated, deterministic security boundaries and per-run taint context of `AgentOrchestrator` without introducing concurrency risks. Instead, the UI displays an honest "Working..." state with a live client-side elapsed timer, followed by the complete verified trace upon response arrival.

---

## 2. Design System & Tokens

### 2.1 Palette: Neutral Slate with Single Blue Accent

The interface uses a clean neutral slate palette with a single blue accent (`#2563EB` light / `#3B82F6` dark). Semantic colors are reserved strictly for status indicators (green for verified/confirmed, amber for needs review/mock, red for errors). No gradients, no glows, and no decorative colored borders are used.

| Token | Light Theme | Dark Theme | Purpose |
| :--- | :--- | :--- | :--- |
| `--bg-page` | `#F8FAFC` | `#0B1220` | Canvas background |
| `--bg-surface` | `#FFFFFF` | `#111827` | Primary card container surface |
| `--bg-subtle` | `#F1F5F9` | `#1E293B` | Secondary container surface, inputs, rows |
| `--border-color` | `#E2E8F0` | `#1F2937` | Default 1px card and divider borders |
| `--border-strong` | `#7C8894` | `#6B7280` | Input control boundaries (WCAG 3:1 compliant) |
| `--text-main` | `#0F172A` | `#E5E7EB` | Primary body and heading text |
| `--text-muted` | `#64748B` | `#9CA3AF` | Subtitles, helper text, and secondary labels |
| `--accent` | `#2563EB` | `#3B82F6` | Primary action button, focus rings, progress fill |
| `--status-green-text` | `#15803D` | `#4ADE80` | Verified checkmarks, confirmed writes, completed tasks |
| `--status-amber-text` | `#B45309` | `#FBBF24` | Needs review, unconfirmed items, mock data badge |
| `--status-red-text` | `#B91C1C` | `#F87171` | Execution errors and failed check alerts |

### 2.2 Typography & Spacing
- **Font Stack**: `system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`.
- **Scale**: 14px base body text, 13px secondary, 16px section headers, 20px page titles. Weights restricted to 400 (regular), 500 (medium), and 600 (semibold).
- **Tabular Numbers**: `font-variant-numeric: tabular-nums` for timers and execution counters.
- **Monospace**: `ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace` strictly for identifiers, latency measurements, and code.
- **Spacing Grid**: Strict 8px grid (`8px`, `16px`, `24px`, `32px`).
- **Corner Radii**: `8px` for cards and inputs, `6px` for status badges.
- **Borders & Shadows**: Subtle 1px borders throughout; cards have a single minimal shadow in light mode (`0 1px 2px 0 rgba(0, 0, 0, 0.04)`) and zero shadow in dark mode.

---

## 3. WCAG AA Contrast Verification

Every text and interactive element was mathematically evaluated for WCAG AA conformance (4.5:1 for body text, 3:1 for UI component borders/icons) using `scripts/check_contrast.py`:

| Element & Pair | Ratio | WCAG AA Standard | Status |
| :--- | :--- | :--- | :--- |
| **Light: Body text** (`#0F172A`) on Surface (`#FFFFFF`) | **17.85:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Light: Body text** (`#0F172A`) on Page (`#F8FAFC`) | **17.06:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Light: Muted text** (`#64748B`) on Surface (`#FFFFFF`) | **4.76:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Light: Muted text** (`#64748B`) on Page (`#F8FAFC`) | **4.55:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Light: Button text** (`#FFFFFF`) on Primary Accent (`#2563EB`) | **5.17:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Light: Primary link** (`#1D4ED8`) on Surface (`#FFFFFF`) | **6.70:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Light: Verified text** (`#15803D`) on Surface (`#FFFFFF`) | **5.02:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Light: Verified text** (`#15803D`) on Subtle Green (`#F0FDF4`) | **4.79:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Light: Review text** (`#B45309`) on Surface (`#FFFFFF`) | **5.02:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Light: Review text** (`#B45309`) on Subtle Amber (`#FFFBEB`) | **4.84:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Light: Error text** (`#B91C1C`) on Surface (`#FFFFFF`) | **6.47:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Light: Error text** (`#B91C1C`) on Subtle Red (`#FEF2F2`) | **5.91:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Light: Focus ring** (`#2563EB`) on Surface (`#FFFFFF`) | **5.17:1** | UI border ≥ 3.0:1 | **PASS** (AA) |
| **Light: Input border** (`#7C8894`) on Surface (`#FFFFFF`) | **3.62:1** | UI border ≥ 3.0:1 | **PASS** (AA) |
| **Dark: Body text** (`#E5E7EB`) on Surface (`#111827`) | **14.33:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Body text** (`#E5E7EB`) on Page (`#0B1220`) | **15.12:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Muted text** (`#9CA3AF`) on Surface (`#111827`) | **6.99:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Muted text** (`#9CA3AF`) on Page (`#0B1220`) | **7.37:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Button text** (`#FFFFFF`) on Primary Accent (`#2563EB`) | **5.17:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Dark: Primary accent text** (`#60A5FA`) on Surface (`#111827`) | **6.98:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Verified text** (`#4ADE80`) on Surface (`#111827`) | **10.18:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Verified text** (`#4ADE80`) on Dark Green (`#052E16`) | **8.55:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Review text** (`#FBBF24`) on Surface (`#111827`) | **10.63:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Review text** (`#FBBF24`) on Dark Amber (`#451A03`) | **8.97:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Error text** (`#F87171`) on Surface (`#111827`) | **6.41:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Error text** (`#F87171`) on Dark Red (`#450A0A`) | **5.84:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Dark: Focus ring** (`#3B82F6`) on Surface (`#111827`) | **4.82:1** | UI border ≥ 3.0:1 | **PASS** (AA) |
| **Dark: Input border** (`#6B7280`) on Surface (`#111827`) | **3.67:1** | UI border ≥ 3.0:1 | **PASS** (AA) |

---

## 4. Layout & Interface States

### 4.1 Slim Top Bar
- **Logo Mark**: Flat square mark in solid blue accent `#2563EB`.
- **Title**: "RAGless" in 600 weight with muted "Operations Agent" subtitle.
- **Controls**: Compact User ID input and dynamic status pill (`ollama · qwen2.5:3b` matching `/health`).

### 4.2 Objective Input Card
- **Textarea**: 14px system font with subtle 1px border and blue focus outline.
- **Keyboard Shortcuts**: Supports `⌘↵` (Cmd+Enter on macOS) and `Ctrl↵` (Ctrl+Enter on Windows/Linux) to submit.
- **Suggestions**: Small, quiet, neutral text buttons (no pills or bright colors).
- **Primary CTA**: 40px solid blue accent button with white text.

### 4.3 Empty State
- Displays a single quiet muted text line below the input: *"Ask about company data, your onboarding progress, or your tasks."*
- Architectural guarantees (Parameterized SQLite, Deterministic Verifier, Taint Isolation) are placed as a discreet, single line in the page footer.

### 4.4 Loading State
- Clean, thin (3px) indeterminate progress bar directly below the input with a live elapsed counter (`Working... 3.2s`). Zero glowing spinners.

### 4.5 Results Display
1. **Answer Card**: Large, readable typography with a status badge (`Verified` in green or `Needs Review` in amber) and plain bulleted unresolved items if present.
2. **Refusal Notice**: Informational neutral bordered notice stating *"No matching record in the company database"* (distinguished from system errors).
3. **Agent Activity Stepper**: Clean vertical stepper showing step index, tool name, success/failure indicator, provenance badge (`sqlite`, `web`, `mock`), per-step latency (if present), and verification checks grid below.
4. **Evidence Panel**: Collapsible panel (collapsed by default) with external web links, snippet quotations, and internal SQLite records. Displays `MOCK DATA` badge when mock searches are used.
5. **Actions Card**: Lists confirmed memory writes and task modifications.
6. **Results Footer**: Thin stats row showing end-to-end duration, LLM call count, answer mode, web call count, and model identifier in tabular numerals.

### 4.6 Right Rail
- **Onboarding Progress**: Thin 4px blue progress bar, "0 of 7 complete", and profile key-value rows.
- **Tasks**: Compact task list with open/completed indicators and total count badge.

---

## 5. Visual Inspection States

To verify each UI state in a browser:

1. **Start the server**:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
   ```
2. **Open in browser**:
   Navigate to `http://127.0.0.1:8000/ui/`.
3. **States to inspect**:
   - **Empty state**: Open `http://127.0.0.1:8000/ui/`. Observe the single muted line under the input without any large placeholder cards.
   - **Loading state**: Click an example prompt (e.g., "First week") and press Ctrl+Enter; observe the thin indeterminate blue progress bar and live timer.
   - **Verified answer**: After the query finishes, observe the green "Verified" badge, readable answer text, activity stepper, and stats footer.
   - **Refusal state**: Enter `Who is the head of Engineering?` and submit; observe the neutral informational notice *"No matching record in the company database."*
   - **Error state**: Stop the local Ollama daemon or submit an invalid payload; observe the neutral bordered error notice with HTTP code.
   - **Dark mode**: Toggle your OS theme to dark mode; observe `#0B1220` page background, `#111827` card surfaces, and slate typography.
   - **Phone width**: Open DevTools, switch to 375px width (iPhone); confirm zero horizontal scrolling, responsive stacked columns, and accessible touch targets.
