# RAGless Web UI Documentation

## 1. Overview
The RAGless Web UI provides an honest, transparent single-page interface served directly by FastAPI at `/ui`. It features a modern, professional **light pink theme** on a **full-width page layout**, ensuring immediate clarity for agent planning steps, deterministic verifications, source provenance, and execution metrics.

### Architectural Principles
- **Zero Build Step**: Pure vanilla HTML5, CSS3, and JavaScript. No npm, Webpack, Vite, or bundle steps.
- **Zero External CDNs**: System font stack (`system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`) and inline SVG icons. Zero external network requests or web font downloads.
- **Strict Content-Security-Policy (CSP)**: `default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self';` (no `'unsafe-inline'` or `'unsafe-eval'`).
- **Complete XSS Immunity**: Dynamic data from LLM responses, web snippets, and database rows is rendered strictly using `textContent` and `createElement`—never `innerHTML` or `outerHTML`.
- **Honest Trace**: All timeline steps, tool executions, provenance labels, and verification checks reflect real server response data. No artificial streaming or simulated delays.
- **SSE Decision**: Server-Sent Events (`/agent/ask/stream`) were **skipped** to preserve the validated, deterministic security boundaries and per-run taint context of `AgentOrchestrator` without introducing concurrency risks. Instead, the UI displays an honest "Working..." state with a live client-side elapsed timer, followed by the complete verified trace upon response arrival.

---

## 2. Design System & Tokens

### 2.1 Palette: Light Pink with Plum Typography

The design uses a single unified light pink theme (identical across OS dark/light mode settings) with dark plum body text for contrast and a rich `#BE185D` accent. Pink is reserved strictly for canvas backgrounds, subtle card fills, and borders. No gradients, glows, or colored top borders are used.

| Token | Value | Purpose |
| :--- | :--- | :--- |
| `--bg-page` | `#FFF5F8` | Canvas full-width background |
| `--bg-surface` | `#FFFFFF` | Card surface |
| `--bg-subtle-pink` | `#FCE7F0` | Subtle pink surface (chips, hover, table alternate rows) |
| `--border-color` | `#F3D5E0` | Subtle 1px dividers and card borders |
| `--border-strong` | `#E4B5C8` | Structural card borders |
| `--border-control` | `#9E5677` | Input control boundaries (WCAG 3:1 compliant) |
| `--text-main` | `#3B1F2B` | Dark plum primary body text |
| `--text-muted` | `#7A5566` | Muted secondary text and labels |
| `--accent` | `#BE185D` | Primary action button, links, progress fill, focus ring |
| `--accent-hover` | `#9D174D` | Primary button hover state |
| `--status-green-text` | `#15803D` | Verified checkmarks, confirmed writes, completed tasks (`#ECFDF3` bg) |
| `--status-amber-text` | `#B45309` | Needs review, unconfirmed items, mock data badge (`#FFF7E6` bg) |
| `--status-red-text` | `#B91C1C` | Execution errors and failed check alerts (`#FEF2F2` bg) |

### 2.2 Typography & Spacing
- **Font Stack**: `system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`.
- **Scale**: 14px base body text, 13px secondary, 16px section headers, 20px page titles. Weights restricted to 400 (regular), 500 (medium), and 600 (semibold).
- **Tabular Numbers**: `font-variant-numeric: tabular-nums` for timers and execution counters.
- **Monospace**: `ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace` strictly for IDs, latency, and code.
- **Corner Radii**: `8px` for cards and inputs, `6px` for status badges.
- **Borders & Shadows**: Flat cards with subtle 1px borders and a single minimal shadow (`0 1px 3px 0 rgba(59, 31, 43, 0.04)`).

---

## 3. WCAG AA Contrast Verification

Every text and interactive element was mathematically evaluated for WCAG AA conformance (4.5:1 for body text, 3:1 for UI component borders/icons) via `python scripts/check_contrast.py`:

| Element & Pair | Ratio | WCAG AA Standard | Status |
| :--- | :--- | :--- | :--- |
| **Body text** (`#3B1F2B`) on Surface (`#FFFFFF`) | **14.82:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Body text** (`#3B1F2B`) on Page Background (`#FFF5F8`) | **13.88:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Body text** (`#3B1F2B`) on Subtle Pink (`#FCE7F0`) | **12.58:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Muted text** (`#7A5566`) on Surface (`#FFFFFF`) | **6.33:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Muted text** (`#7A5566`) on Page Background (`#FFF5F8`) | **5.93:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Muted text** (`#7A5566`) on Subtle Pink (`#FCE7F0`) | **5.37:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Button text** (`#FFFFFF`) on Accent (`#BE185D`) | **6.04:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Accent link** (`#BE185D`) on Surface (`#FFFFFF`) | **6.04:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Success text** (`#15803D`) on Surface (`#FFFFFF`) | **5.02:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Success text** (`#15803D`) on Success Background (`#ECFDF3`) | **4.76:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Warning text** (`#B45309`) on Surface (`#FFFFFF`) | **5.02:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Warning text** (`#B45309`) on Warning Background (`#FFF7E6`) | **4.71:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Error text** (`#B91C1C`) on Surface (`#FFFFFF`) | **6.47:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Error text** (`#B91C1C`) on Error Background (`#FEF2F2`) | **5.91:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Accent focus ring** (`#BE185D`) on Surface (`#FFFFFF`) | **6.04:1** | UI border ≥ 3.0:1 | **PASS** (AA) |
| **Input control border** (`#9E5677`) on Surface (`#FFFFFF`) | **5.18:1** | UI border ≥ 3.0:1 | **PASS** (AA) |
| **Input control border** (`#9E5677`) on Subtle Pink (`#FCE7F0`) | **4.40:1** | UI border ≥ 3.0:1 | **PASS** (AA) |

---

## 4. Full-Width Layout & Component States

### 4.1 Full-Width Viewport & Top Bar
- Spans 100% of the viewport width with 24px side padding (16px on mobile screens).
- Flat square logo mark displaying the letter **"R"** in white on `#BE185D`.
- "RAGless" in 600 weight with muted "Operations Agent" subtitle.
- Right: Status pill (dot + active model, e.g. `ollama · qwen2.5:3b`) and compact User ID field.
- **Fail-Safe Connection Handling**: Shows neutral "Loading" before first fetch; switches to "Unavailable" with an amber dot if the health endpoint fails or times out (4s deadline).

### 4.2 Main Area (70% / 30% Grid)
- **Left Column (70%)**:
  - Input card: textarea + primary solid accent "Run Agent" button + muted shortcut hint **"Ctrl+Enter"** vertically aligned to the right.
  - Example prompts on `#FCE7F0` subtle pink surface (quiet, neutral text buttons).
  - Empty state: Single muted line (*"Ask about company data, your onboarding progress, or your tasks."*).
  - Loading state: Thin indeterminate progress bar + live counter (*"Working... 3.2s"*).
  - Results order:
    1. **Answer card**: Large, readable text, status badge (`Verified` / `Needs Review`), plain bulleted unresolved items if present.
    2. **Refusal notice**: Informational neutral notice stating *"No matching record in the company database."*
    3. **Agent activity stepper**: Tool name, success/failure icon, provenance badge (`sqlite`, `web`, `mock`), per-step latency (if available), and verification checks grid below.
    4. **Evidence panel**: Collapsible panel (collapsed by default) with external links, snippet quotes, and SQLite records. `MOCK DATA` badge appears if mock results are present.
    5. **Actions card**: Confirmed memory writes and task changes.
    6. **Results footer**: Thin stats row (duration, LLM calls, answer mode, web calls, model).
- **Right Rail (30% Sticky)**:
  - Sticky below the top bar on desktop, stacking below 900px.
  - **Onboarding progress**: 4px progress bar with accent fill (strictly **empty at 0%**, never full), "N of M complete", and key-value role/department rows.
  - **Tasks**: Compact task list with open/completed indicators and total count badge.

### 4.3 Page Footer
- Spans full width with a single muted line summarizing architecture guarantees:
  *"Deterministic multi-step execution · Parameterized SQLite · Taint isolation"*.

---

## 5. Visual Inspection Across Screen Widths

Verify the layout across standard breakpoints:
- **1920px & 1440px (Wide Desktop)**: Content spans the full viewport width; 70% left column and 30% sticky right rail with generous spacing and zero horizontal scrolling.
- **1024px (Tablet Landscape)**: Clean two-column layout adhering to the 320px minimum right rail boundary.
- **768px (Tablet Portrait)**: Stacks into a single column at ≤900px; the right rail flows cleanly beneath the left column results.
- **375px (Mobile Phone)**: 16px side padding; top bar controls, input footer, and stat items reflow cleanly with zero horizontal scroll and fully accessible touch targets.
