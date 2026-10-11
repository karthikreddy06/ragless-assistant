# Minimal Task Tools Documentation

## 1. Overview
The `tasks` tool gives the agent safe, constrained, actionable capabilities. It is not an arbitrary task system; it is restricted to fixed operations with strict boundary controls.

## 2. SQLite Database Schema
The `tasks` table stores task records in SQLite:
```sql
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('onboarding', 'user')),
    onboarding_item_id INTEGER NULL,
    status TEXT NOT NULL CHECK(status IN ('open', 'done')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_user_onboarding
ON tasks(user_id, onboarding_item_id)
WHERE onboarding_item_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tasks_user_status
ON tasks(user_id, status);
```

All queries use parameterized SQL (`?` placeholders).

## 3. Allowed Operations
The `tasks` tool is registered in `ALLOWED_TOOLS` with four fixed operations:
1. `create_tasks_from_onboarding`:
   - Deterministic and idempotent.
   - Populates tasks from the `onboarding_items` table matching the user's department (plus department-null items).
   - Skips items that already have a task for the user (`INSERT OR IGNORE`).
   - Uses zero LLM-generated titles.
2. `create_task`:
   - Arguments: `{"title": "<string>"}`.
   - Title is length-capped (<= 150 chars), charset-allowlisted (`^[a-zA-Z0-9\s.,_\-?!()'\":/]+$`), and screened against forbidden SQL injection patterns.
3. `list_tasks`:
   - Read-only operation.
   - Optional arguments: `{"status": "open" | "done"}`.
4. `complete_task`:
   - Arguments: `{"task_id": <positive integer>}`.
   - Verified against the server-injected `user_id`.

No delete, update-title, bulk-delete, or arbitrary operations exist.

## 4. Deterministic Intent Verification in Python (Executor Level)
Task write operations cannot be triggered by LLM planning alone; they are strictly gated at the executor level in Python:
- `create_tasks_from_onboarding`: Only permitted when user text clearly asks to create, add, or set up onboarding tasks (e.g., "create tasks for my onboarding", "add my onboarding tasks").
- `create_task`: Only permitted on explicit "create/add a task ..." phrasing, AND the title must appear verbatim (case-insensitive) in the user text.
- `complete_task`: Only permitted on explicit completion phrasing, AND the task must match the user text by ID or title keywords.
- Read-only inquiries (e.g. "what tasks do I have?", "what is a task?") cause zero writes across all write operations.

## 5. Security & Taint Enforcement
- **Taint Rule**: If web content enters an execution run (via `web_research`), all subsequent task writes (`create_tasks_from_onboarding`, `create_task`, `complete_task`) are immediately rejected with a clean security violation error.
- **Per-request write limit**: Maximum 10 task writes per request.
- **Per-user open tasks limit**: Maximum 50 open tasks per user.
- **User isolation**: `user_id` is never accepted from LLM arguments (`extra="forbid"` and explicit validator checks). It is strictly injected by the server.

## 6. Read-Back Verification
Every write operation executes an immediate database read-back to confirm row creation or status transition. The verifier verifies `write_confirmed`, and final answers are blocked from falsely claiming task creation or completion unless confirmed by the database. Confirmed writes are surfaced in `AgentResponse.tasks_changed`.

## 7. Relationship to Memory: Explicit & Separate
- **Memory (`memory.mark_item_complete`)**: Tracks official onboarding checklist progress (compliance status, completed item IDs in `user_completed_items`).
- **Tasks (`tasks.complete_task`)**: Tracks day-to-day actionable workflow tasks for the user.
- **Design Decision**: Completing a task linked to an onboarding item does NOT silently mark the onboarding checklist item complete in memory. They are kept explicit and separate to prevent side-channel state changes and maintain deterministic auditability.

## 8. API Endpoints
- `GET /tasks/{user_id}`: Read-only listing of tasks for the user, with optional `status=open|done` filter.
- **Authentication Note**: `user_id` is a demonstration identifier; no real multi-tenant authentication is provided.
