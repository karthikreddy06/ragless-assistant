# RAGless Web UI Documentation

## 1. Overview
The RAGless Web UI provides an honest, transparent single-page interface served directly by FastAPI at `/ui`. It features an elegant **Soft Pink & Sky Blue Marble** design on a **full-width page layout**, ensuring immediate clarity for agent planning steps, deterministic verifications, source provenance, and execution metrics.

### Architectural Principles
- **Zero Build Step**: Pure vanilla HTML5, CSS3, and JavaScript. No npm, Webpack, Vite, or bundle steps.
- **Zero External CDNs**: System font stack (`system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`) and inline SVG icons. Zero external network requests or web font downloads.
- **Strict Content-Security-Policy (CSP)**: `default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self';` (no `'unsafe-inline'` or `'unsafe-eval'`).
- **Complete XSS Immunity**: Dynamic data from LLM responses, web snippets, and database rows is rendered strictly using `textContent` and `createElement`—never `innerHTML` or `outerHTML`.
- **Honest Trace**: All timeline steps, tool executions, provenance labels, and verification checks reflect real server response data. No artificial streaming or simulated delays.
- **SSE Decision**: Server-Sent Events (`/agent/ask/stream`) were **skipped** to preserve the validated, deterministic security boundaries and per-run taint context of `AgentOrchestrator` without introducing concurrency risks. Instead, the UI displays an honest "Working..." state with a live client-side elapsed timer, followed by the complete verified trace upon response arrival.

---

## 2. Design System: Soft Pink & Sky Blue Marble

### 2.1 Palette & Tokens

The visual style is inspired by dreamy, watercolor-marble art, blending soft pastel pink organically into light sky blue. The design is elegant, calming, modern, and professional—not childish or overly colorful.

| Token | Hex Value | Purpose |
| :--- | :--- | :--- |
| `--bg-page` | `#FDF7FA` | Canvas background with subtle watercolor/marble gradients |
| `--bg-surface` | `#FFFFFF` | Card surfaces |
| `--soft-pink` | `#F4C6D8` | Primary button fill, logo background, selected states |
| `--light-blush` | `#F9E3EC` | Prompt chips, subtle highlights, alternate rows |
| `--sky-blue` | `#B9E5F5` | Complementary secondary accents, step numbers, badges |
| `--pale-blue` | `#D9F1FA` | Table headers, informational banners, source badges |
| `--border-color` | `#EADBE5` | Subtle 1px dividers and card borders |
| `--border-control`| `#8E6B83` | Input control boundaries (WCAG 3:1 compliant) |
| `--text-main` | `#382936` | Primary body text (deep plum) |
| `--text-muted` | `#796B79` | Secondary text and labels |
| `--accent-primary`| `#C76B98` | Focus rings, progress bar fill, button borders |
| `--accent-hover` | `#AD507F` | Primary button and link hover state |
| `--status-green-text` | `#286036` | Verified checkmarks, confirmed writes, completed tasks (`#EDF7EE` bg) |
| `--status-amber-text` | `#8A4E12` | Needs review, unconfirmed items, mock data badge (`#FFF8E1` bg) |
| `--status-red-text` | `#991B1B` | Execution errors and failed check alerts (`#FEE2E2` bg) |

### 2.2 Background & Surfaces
- **Watercolor-Marble Background**: A soft, organic blend of radial gradients (`#B9E5F5` sky blue, `#F4C6D8` soft pink, and `#F9E3EC` light blush) over a light `#FDF7FA` linear base. The wash is low-contrast and soft, ensuring 100% text legibility.
- **Card Surfaces**: Pure white `#FFFFFF` cards with 1px `#EADBE5` borders and minimal subtle shadows (`0 1px 3px 0 rgba(56, 41, 54, 0.04)`).
- **Corner Radii**: Rounded corners of `8–10px` (`8px` for inputs/badges, `10px` for cards).

---

## 3. WCAG AA Contrast Verification

Every text and interactive element was mathematically evaluated for WCAG AA conformance (4.5:1 for body text, 3:1 for UI component borders/icons) via `python scripts/check_contrast.py`:

| Element & Pair | Computed Ratio | Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Primary Text** (`#382936`) on Surface (`#FFFFFF`) | **13.65:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Primary Text** (`#382936`) on Page Background (`#FDF7FA`) | **12.91:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Primary Text** (`#382936`) on Soft Pink Button (`#F4C6D8`) | **9.04:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Primary Text** (`#382936`) on Light Blush (`#F9E3EC`) | **11.20:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Primary Text** (`#382936`) on Sky Blue (`#B9E5F5`) | **10.13:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Primary Text** (`#382936`) on Pale Blue (`#D9F1FA`) | **11.64:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Secondary Text** (`#796B79`) on Surface (`#FFFFFF`) | **5.01:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Secondary Text** (`#796B79`) on Page Background (`#FDF7FA`) | **4.74:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Accent Hover** (`#AD507F`) on Surface (`#FFFFFF`) | **4.96:1** | Body text ≥ 4.5:1 | **PASS** (AA) |
| **Success Text** (`#286036`) on Surface (`#FFFFFF`) | **7.46:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Success Text** (`#286036`) on Success Subtle (`#EDF7EE`) | **6.80:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Warning Text** (`#8A4E12`) on Surface (`#FFFFFF`) | **6.61:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Warning Text** (`#8A4E12`) on Warning Subtle (`#FFF8E1`) | **6.22:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Error Text** (`#991B1B`) on Surface (`#FFFFFF`) | **8.31:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Error Text** (`#991B1B`) on Error Subtle (`#FEE2E2`) | **6.80:1** | Body text ≥ 4.5:1 | **PASS** (AAA) |
| **Focus Ring / Accent** (`#C76B98`) on Surface (`#FFFFFF`) | **3.51:1** | UI border ≥ 3.0:1 | **PASS** (AA) |
| **Input Control Border** (`#8E6B83`) on Surface (`#FFFFFF`) | **4.58:1** | UI border ≥ 3.0:1 | **PASS** (AA) |
| **Input Control Border** (`#8E6B83`) on Soft Pink (`#F4C6D8`) | **3.04:1** | UI border ≥ 3.0:1 | **PASS** (AA) |
| **Input Control Border** (`#8E6B83`) on Light Blush (`#F9E3EC`)| **3.76:1** | UI border ≥ 3.0:1 | **PASS** (AA) |

---

## 4. Full-Width Layout & Component States

### 4.1 Full-Width Top Bar
- Spans 100% of the viewport width with 24px side padding (16px on mobile).
- Flat square logo mark displaying the letter **"R"** on pastel pink `#F4C6D8`.
- "RAGless" in 600 weight with muted "Operations Agent" subtitle.
- Right: Status pill in pale blue `#D9F1FA` (dot + active model, e.g. `ollama · qwen2.5:3b`) and compact User ID input.
- **Fail-Safe Timeout**: Displays "Loading..." before first fetch; switches to "Unavailable" with an amber dot if the health endpoint fails or exceeds 4 seconds.

### 4.2 Main Area (70% / 30% Grid)
- **Left Column (70%)**:
  - Input card: textarea + primary pastel pink `#F4C6D8` "Run Agent" button + muted **"Ctrl+Enter"** shortcut label.
  - Example prompts on `#F9E3EC` light blush surface (quiet, neutral text buttons).
  - Empty state: Single muted line (*"Ask about company data, your onboarding progress, or your tasks."*).
  - Loading state: Thin indeterminate progress bar + live counter (*"Working... 3.2s"*).
  - Results order:
    1. **Answer card**: Large, readable text, status badge (`Verified` / `Needs Review`), plain bulleted unresolved items if present.
    2. **Refusal notice**: Informational neutral notice in pale blue stating *"No matching record in the company database."*
    3. **Agent activity stepper**: Tool name, success/failure icon, provenance badge (`sqlite`, `web`, `mock`), per-step latency (if present), and verification checks grid below.
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

## 5. Responsive Behavior Across Breakpoints

Automated rule verification via `python scripts/verify_responsive.py`:
- **1920px & 1440px (Wide Desktop)**: Full monitor width, 70% left column, 30% sticky right rail, 24px gutters, no horizontal scroll.
- **1024px (Tablet Landscape)**: Two-column layout preserving minimum 320px right rail.
- **768px (Tablet Portrait)**: Stacks into a single column at ≤900px.
- **375px (Mobile Phone)**: 16px side padding; top bar controls, input footer, and stat items reflow cleanly with zero horizontal scroll.
