# RAGless Web UI Documentation

## 1. Overview
The RAGless Web UI provides an honest, transparent single-page interface served directly by FastAPI at `/ui`. It features an elegant **Soft Pink & Sky Blue Marble** design on a **full-screen dashboard layout**, ensuring immediate clarity for agent planning steps, deterministic verifications, source provenance, and execution metrics.

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
- **Watercolor-Marble Background**: A soft, organic blend of radial gradients (`#B9E5F5` sky blue, `#F4C6D8` soft pink, and `#F9E3EC` light blush) over a light `#FDF7FA` linear base. Gradients are placed around page edges, ensuring the central workspace remains high-contrast and legible.
- **Card Surfaces**: Pure white `#FFFFFF` cards with 1px `#EADBE5` borders and minimal subtle shadows (`0 1px 3px 0 rgba(56, 41, 54, 0.04), 0 4px 12px 0 rgba(56, 41, 54, 0.02)`).
- **Corner Radii**: Rounded corners of `8–12px` (`8px` for inputs/badges, `10px` for cards, `12px` for illustrations).

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

## 4. Full-Screen Dashboard Layout & Component Architecture

### 4.1 Full-Width Top Bar
- Spans 100% of viewport width with 24px side padding (16px on mobile).
- Flat square logo mark displaying the letter **"R"** on pastel pink `#F4C6D8` with `#C76B98` border.
- "RAGless" in 600 weight with muted "Operations Agent" subtitle.
- Right: Status pill in pale blue `#D9F1FA` (dot + active model, e.g. `ollama · qwen2.5:3b`) and User ID input (`demo-user`).
- **Fail-Safe Timeout**: Displays "Loading..." initially; switches to "Unavailable" with an amber dot if `/health` fails or exceeds 4 seconds.

### 4.2 Main Area (~70% Workspace / ~30% Sidebar Grid)
- **Left Column (~70%)**:
  - **Agent Query Composer**:
    - Header with spark icon, title, and "Ctrl + Enter to run" shortcut badge.
    - Spacious textarea (`min-height: 96px`, padding `14px 16px`).
    - Quick prompts in light blush (`#F9E3EC`) with border `#EADBE5`.
    - Primary pastel pink `#F4C6D8` "Run Agent" button with dark text `#382936` and `#C76B98` border.
    - Loading state: indeterminate progress bar + live counter.
  - **Polished Welcome Section** (`#welcome-section`):
    - Appears before any query has run, filling the workspace thoughtfully instead of leaving empty space.
    - "AI Operations Workspace" badge pill.
    - Heading: *"Your operations, intelligently connected."*
    - Subheading: *"Query company knowledge, review onboarding progress, and manage tasks with deterministic verification and verified evidence."*
    - **3 Interactive Suggestion Cards**:
      1. *First-Week Roadmap*: Prepare policies, schedules, & tools (`Prepare everything I need for my first week`)
      2. *Remaining Onboarding*: Review pending steps & requirements (`What do I still need to do?`)
      3. *Leave vs UK Statutory*: Compare company policy to UK minimum (`How does our annual leave compare to the UK statutory minimum?`)
      *(Clicking any card populates the composer and focuses it).*
    - **Tasteful Connected-Node AI Illustration**: Lightweight inline SVG illustrating connected policy, web research, memory, and task nodes around an operations core, with soft watercolor radial aura.
    - Automatically hides when the agent runs or displays results/errors.
  - **Results Order (when query completes)**:
    1. **Answer card**: Large readable text, status badge (`Verified` / `Needs Review`), plain bulleted unresolved items if present.
    2. **Refusal notice**: Informational neutral notice in pale blue stating *"No matching record in the company database."*
    3. **Agent activity stepper**: Tool name, success/failure status, provenance badge (`sqlite`, `web`, `mock`), per-step latency, and verification checks.
    4. **Evidence panel**: Collapsible panel with external web links, snippet quotes, and SQLite records table. `MOCK DATA` badge appears if mock results are present.
    5. **Actions card**: Confirmed memory writes and task changes.
    6. **Results footer**: Thin stats row (duration, LLM calls, answer mode, web calls, model).
- **Right Rail (~30% Sticky)**:
  - Sticky below the top bar on desktop (`position: sticky; top: 76px;`).
  - **Onboarding progress**: 6px progress bar with accent fill (strictly empty at 0%), "N of M complete", bold percentage, and key-value role/department rows.
  - **Tasks**: Count badge in header, structured task list with status badges, and clean empty state with clipboard icon and concise explanation when no tasks exist.

### 4.3 Page Footer
- Spans full width at viewport bottom using `margin-top: auto`.
- Summary of architecture guarantees: *"Deterministic multi-step execution · Parameterized SQLite · Taint isolation"*.

---

## 5. Responsive Behavior Across Breakpoints

Automated rule verification via `python scripts/verify_responsive.py`:
- **1920px & 1440px (Wide Desktop)**: Full monitor width, 70% left column, 30% sticky right rail, 24px gutters, no horizontal scroll.
- **1024px (Tablet Landscape)**: Two-column layout preserving minimum 320px right rail.
- **768px (Tablet Portrait)**: Stacks into a single column at ≤900px (`grid-template-columns: 1fr`, welcome section reflows cleanly).
- **375px (Mobile Phone)**: 16px side padding; top bar controls, input footer, and stat items reflow cleanly with zero horizontal scroll.
- **Motion Accessibility**: Full support for `prefers-reduced-motion: reduce`.
