import logging
import re
from typing import Any, Dict, List, Optional
import sqlite3

from app.database import get_connection
from app.models import FORBIDDEN_SQL_PATTERNS

logger = logging.getLogger(__name__)

USER_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")
MAX_TASK_TITLE_LEN = 150
TASK_TITLE_ALLOWED_CHARS = re.compile(r"^[a-zA-Z0-9\s.,_\-?!()'\":/]+$")
MAX_OPEN_TASKS_PER_USER = 50

# Regex patterns for explicit task intent validation
ONBOARDING_TASKS_CREATE_INTENT_PATTERN = re.compile(
    r"\b(?:"
    r"(?:create|add|set\s*up|populate|generate)\b.*?\b(?:onboarding\s+tasks?|tasks?\s+(?:for|from)\s+(?:my\s+)?onboarding|tasks?\s+for\s+(?:my\s+)?first\s+week)"
    r"|(?:create|add|set\s*up)\b.*?\btasks?\b.*?\b(?:onboarding|first\s+week)\b"
    r")",
    re.IGNORECASE,
)

CREATE_TASK_INTENT_PATTERN = re.compile(
    r"\b(?:create|add|new)\s+(?:a\s+)?task\b",
    re.IGNORECASE,
)

COMPLETE_TASK_INTENT_PATTERN = re.compile(
    r"\b(?:"
    r"(?:complete|finish|mark)\s+task\b"
    r"|task\s+#?\d+\s+(?:is\s+)?(?:complete|completed|done|finished)"
    r"|mark\s+task\s+#?\d+\s+as\s+(?:done|complete|completed|finished)"
    r"|i(?:'ve|\s+have)?\s+(?:completed|finished)\s+task\b"
    r"|(?:complete|finished?)\s+#?\d+"
    r")",
    re.IGNORECASE,
)

READ_ONLY_TASK_PATTERNS = [
    re.compile(r"^(?:what|which|show|list|view|do\s+i\s+have|are\s+there|how\s+many)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:is|are)\s+(?:a\s+)?tasks?\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+tasks?\s+do\s+i\s+have\b", re.IGNORECASE),
]

TITLE_STOP_WORDS = {
    "and", "or", "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "as", "by", "complete", "completed", "setup", "set", "review", "submit", "meet",
    "task", "tasks", "item", "items", "first", "week"
}


class TaskError(ValueError):
    """Domain error for task operations."""
    pass


class TaskLimitError(TaskError):
    """Raised when open tasks limit is exceeded."""
    pass


def validate_user_id(user_id: str) -> str:
    """Validate server-injected user_id format."""
    if not user_id or not isinstance(user_id, str):
        raise TaskError("user_id must be a non-empty string.")
    uid = user_id.strip()
    if not USER_ID_REGEX.match(uid):
        raise TaskError(f"Invalid user_id format: '{user_id}'.")
    return uid


def validate_task_title(title: str) -> str:
    """Validate task title for length, allowed charset, and SQL patterns."""
    if not title or not isinstance(title, str):
        raise TaskError("Task title must be a non-empty string.")
    clean = title.strip()
    if not clean:
        raise TaskError("Task title cannot be empty.")
    if len(clean) > MAX_TASK_TITLE_LEN:
        raise TaskError(f"Task title exceeds maximum length of {MAX_TASK_TITLE_LEN} characters.")
    if not TASK_TITLE_ALLOWED_CHARS.match(clean):
        raise TaskError("Task title contains disallowed characters.")
    clean_lower = clean.lower()
    for pat in FORBIDDEN_SQL_PATTERNS:
        if re.search(pat, clean_lower):
            raise TaskError(f"Task title contains forbidden SQL syntax: '{title}'.")
    return clean


INTERROGATIVE_MODAL_PATTERN = re.compile(
    r"\b(?:do|does|did|should|can|could|what|how|why|is|are|must)\b",
    re.IGNORECASE,
)

ONBOARDING_TASKS_VERB_PATTERN = re.compile(
    r"\b(?:create|add|set\s*up|populate|generate)\b",
    re.IGNORECASE,
)

CREATE_TASK_VERB_PATTERN = re.compile(
    r"\b(?:create|add|new)\b",
    re.IGNORECASE,
)

COMPLETE_TASK_VERB_PATTERN = re.compile(
    r"\b(?:complete|finish|mark|completed|finished)\b",
    re.IGNORECASE,
)


def check_imperative_write_intent(text: str, verb_pattern: re.Pattern) -> bool:
    """
    Enforces that write intent requires imperative phrasing:
    1. Any text containing '?' is rejected.
    2. Any text starting with or containing an interrogative/modal
       (do, should, can, could, what, how, why, is, are, must) before the action verb is rejected.
    """
    if not text:
        return False
    if "?" in text:
        return False

    match = verb_pattern.search(text)
    if not match:
        return False

    prefix = text[: match.start()]
    if INTERROGATIVE_MODAL_PATTERN.search(prefix):
        return False

    return True


def validate_onboarding_tasks_intent(question: Optional[str]) -> bool:
    """
    Validates explicit intent to create onboarding tasks.
    Must clearly ask to create/add/set up onboarding tasks.
    Write intent requires imperative phrasing: any text containing '?' or containing
    an interrogative/modal (do, should, can, could, what, how, why, is, are, must)
    before the action verb is rejected.
    """
    if not question:
        return False
    q = question.strip()

    if not check_imperative_write_intent(q, ONBOARDING_TASKS_VERB_PATTERN):
        return False

    return bool(ONBOARDING_TASKS_CREATE_INTENT_PATTERN.search(q))


def validate_create_task_intent(question: Optional[str], title: str) -> bool:
    """
    Validates explicit intent to create an individual task:
    1. Write intent requires imperative phrasing: any text containing '?' or
       starting with an interrogative/modal before the verb is rejected.
    2. Question must explicitly state 'create/add a task ...'.
    3. The title must appear verbatim (case-insensitive) in the user text.
    """
    if not question or not title:
        return False
    q = question.strip()
    q_lower = q.lower()

    if not check_imperative_write_intent(q, CREATE_TASK_VERB_PATTERN):
        return False

    if not CREATE_TASK_INTENT_PATTERN.search(q):
        return False

    title_clean = title.strip().lower()
    if not title_clean or title_clean not in q_lower:
        return False

    return True


def validate_complete_task_intent(
    question: Optional[str],
    task_id: int,
    task_title: Optional[str] = None,
) -> bool:
    """
    Validates explicit intent to complete a task:
    1. Write intent requires imperative phrasing: any text containing '?' or
       starting with an interrogative/modal before the verb is rejected.
    2. Question must have explicit completion phrasing.
    3. The task must match the user's text by id or title keywords.
    """
    if not question or not task_id:
        return False
    q = question.strip()

    if not check_imperative_write_intent(q, COMPLETE_TASK_VERB_PATTERN):
        return False

    if not COMPLETE_TASK_INTENT_PATTERN.search(q):
        return False

    # Check match by task ID
    if str(task_id) in q:
        return True

    # Check match by title keywords
    if task_title:
        title_words = [
            w for w in re.findall(r"\b[a-zA-Z]{3,}\b", task_title.lower())
            if w not in TITLE_STOP_WORDS
        ]
        q_lower = q.lower()
        if title_words and any(w in q_lower for w in title_words):
            return True

    return False


class TaskStore:
    """
    Persistent store for user tasks in SQLite.
    All operations are strictly parameterized, user-isolated, and include immediate read-back verification.
    """

    def __init__(self, connection_factory=get_connection):
        self.get_connection = connection_factory

    def get_open_tasks_count(self, user_id: str) -> int:
        """Count currently open tasks for the user."""
        uid = validate_user_id(user_id)
        connection = self.get_connection()
        try:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM tasks WHERE user_id = ? AND status = 'open'",
                (uid,),
            )
            row = cursor.fetchone()
            return int(row[0]) if row else 0
        finally:
            connection.close()

    def create_tasks_from_onboarding(self, user_id: str) -> Dict[str, Any]:
        """
        Deterministic, idempotent task creation from onboarding checklist items.
        Creates tasks for user's department plus department-null items, skipping ones that exist.
        No LLM-provided titles are used.
        """
        uid = validate_user_id(user_id)
        connection = self.get_connection()
        try:
            with connection:
                cursor = connection.cursor()

                # 1. Fetch user department
                cursor.execute(
                    "SELECT department FROM user_profile WHERE user_id = ?",
                    (uid,),
                )
                prof_row = cursor.fetchone()
                dept = prof_row["department"] if prof_row and prof_row["department"] else None

                # 2. Fetch matching onboarding items
                if dept:
                    cursor.execute(
                        "SELECT id, title FROM onboarding_items WHERE department = ? OR department IS NULL ORDER BY day_due ASC, id ASC",
                        (dept,),
                    )
                else:
                    cursor.execute(
                        "SELECT id, title FROM onboarding_items ORDER BY day_due ASC, id ASC"
                    )
                items = cursor.fetchall()

                # 3. Check open tasks limit
                cursor.execute(
                    "SELECT COUNT(*) FROM tasks WHERE user_id = ? AND status = 'open'",
                    (uid,),
                )
                open_count = cursor.fetchone()[0]

                # Check which items already have tasks for this user
                cursor.execute(
                    "SELECT onboarding_item_id FROM tasks WHERE user_id = ? AND onboarding_item_id IS NOT NULL",
                    (uid,),
                )
                existing_item_ids = {r[0] for r in cursor.fetchall()}

                new_items = [it for it in items if it["id"] not in existing_item_ids]
                if open_count + len(new_items) > MAX_OPEN_TASKS_PER_USER:
                    raise TaskLimitError(
                        f"Task limit exceeded: creating {len(new_items)} onboarding tasks would exceed the maximum of {MAX_OPEN_TASKS_PER_USER} open tasks."
                    )

                # 4. Insert new tasks idempotently
                created_tasks = []
                for it in new_items:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO tasks (user_id, title, source, onboarding_item_id, status)
                        VALUES (?, ?, 'onboarding', ?, 'open')
                        """,
                        (uid, it["title"], it["id"]),
                    )
                    task_id = cursor.lastrowid
                    if task_id:
                        created_tasks.append({
                            "id": task_id,
                            "title": it["title"],
                            "onboarding_item_id": it["id"],
                        })

                # 5. Read-back verification
                cursor.execute(
                    """
                    SELECT id, user_id, title, source, onboarding_item_id, status, created_at, completed_at
                    FROM tasks
                    WHERE user_id = ? AND source = 'onboarding'
                    ORDER BY id ASC
                    """,
                    (uid,),
                )
                all_onboarding_rows = cursor.fetchall()
                all_tasks = [dict(r) for r in all_onboarding_rows]

                # Verify all expected onboarding items exist in read-back
                stored_item_ids = {r["onboarding_item_id"] for r in all_tasks if r["onboarding_item_id"]}
                for it in items:
                    if it["id"] not in stored_item_ids:
                        raise TaskError(f"Read-back verification failed: missing task for onboarding item {it['id']}.")

                tasks_changed = [
                    {
                        "operation": "create_tasks_from_onboarding",
                        "task_id": t["id"],
                        "title": t["title"],
                        "confirmed": True,
                    }
                    for t in created_tasks
                ]

                return {
                    "success": True,
                    "write_confirmed": True,
                    "operation": "create_tasks_from_onboarding",
                    "created_count": len(created_tasks),
                    "total_onboarding_tasks": len(all_tasks),
                    "tasks": all_tasks,
                    "tasks_changed": tasks_changed,
                }
        finally:
            connection.close()

    def create_task(self, user_id: str, title: str) -> Dict[str, Any]:
        """
        Creates a custom user task with immediate read-back verification.
        Enforces MAX_OPEN_TASKS_PER_USER and strict title validation.
        """
        uid = validate_user_id(user_id)
        valid_title = validate_task_title(title)

        connection = self.get_connection()
        try:
            with connection:
                cursor = connection.cursor()

                # 1. Enforce open tasks limit
                cursor.execute(
                    "SELECT COUNT(*) FROM tasks WHERE user_id = ? AND status = 'open'",
                    (uid,),
                )
                open_count = cursor.fetchone()[0]
                if open_count >= MAX_OPEN_TASKS_PER_USER:
                    raise TaskLimitError(
                        f"Task limit exceeded: user has reached the maximum of {MAX_OPEN_TASKS_PER_USER} open tasks."
                    )

                # 2. Insert new task
                cursor.execute(
                    """
                    INSERT INTO tasks (user_id, title, source, onboarding_item_id, status)
                    VALUES (?, ?, 'user', NULL, 'open')
                    """,
                    (uid, valid_title),
                )
                task_id = cursor.lastrowid
                if not task_id:
                    raise TaskError("Database insert failed: no row id generated.")

                # 3. Read-back verification
                cursor.execute(
                    """
                    SELECT id, user_id, title, source, onboarding_item_id, status, created_at, completed_at
                    FROM tasks
                    WHERE id = ? AND user_id = ?
                    """,
                    (task_id, uid),
                )
                row = cursor.fetchone()
                if not row or row["title"] != valid_title or row["status"] != "open":
                    raise TaskError(f"Read-back verification failed for newly created task #{task_id}.")

                task_data = dict(row)
                return {
                    "success": True,
                    "write_confirmed": True,
                    "operation": "create_task",
                    "task": task_data,
                    "tasks_changed": [
                        {
                            "operation": "create_task",
                            "task_id": task_data["id"],
                            "title": task_data["title"],
                            "confirmed": True,
                        }
                    ],
                }
        finally:
            connection.close()

    def list_tasks(self, user_id: str, status: Optional[str] = None) -> Dict[str, Any]:
        """
        Read-only listing of user tasks, optionally filtered by status ('open' or 'done').
        """
        uid = validate_user_id(user_id)
        connection = self.get_connection()
        try:
            cursor = connection.cursor()
            if status:
                st = status.strip().lower()
                if st not in ("open", "done"):
                    raise TaskError(f"Invalid status filter '{status}'. Must be 'open' or 'done'.")
                cursor.execute(
                    """
                    SELECT id, user_id, title, source, onboarding_item_id, status, created_at, completed_at
                    FROM tasks
                    WHERE user_id = ? AND status = ?
                    ORDER BY id ASC
                    """,
                    (uid, st),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, user_id, title, source, onboarding_item_id, status, created_at, completed_at
                    FROM tasks
                    WHERE user_id = ?
                    ORDER BY id ASC
                    """,
                    (uid,),
                )
            rows = cursor.fetchall()
            task_list = [dict(r) for r in rows]
            return {
                "success": True,
                "operation": "list_tasks",
                "count": len(task_list),
                "tasks": task_list,
            }
        finally:
            connection.close()

    def complete_task(self, user_id: str, task_id: int) -> Dict[str, Any]:
        """
        Marks a task completed with immediate read-back verification.
        Task must belong to server-injected user_id.
        """
        uid = validate_user_id(user_id)
        if not isinstance(task_id, int) or task_id < 1:
            raise TaskError(f"task_id must be a positive integer, got: {task_id}")

        connection = self.get_connection()
        try:
            with connection:
                cursor = connection.cursor()

                # 1. Verify existence and ownership
                cursor.execute(
                    """
                    SELECT id, user_id, title, source, onboarding_item_id, status, created_at, completed_at
                    FROM tasks
                    WHERE id = ? AND user_id = ?
                    """,
                    (task_id, uid),
                )
                row = cursor.fetchone()
                if not row:
                    raise TaskError(f"Task #{task_id} not found for user '{uid}'.")

                # 2. Update status
                cursor.execute(
                    """
                    UPDATE tasks
                    SET status = 'done', completed_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND user_id = ?
                    """,
                    (task_id, uid),
                )

                # 3. Read-back verification
                cursor.execute(
                    """
                    SELECT id, user_id, title, source, onboarding_item_id, status, created_at, completed_at
                    WHERE id = ? AND user_id = ?
                    """ if False else """
                    SELECT id, user_id, title, source, onboarding_item_id, status, created_at, completed_at
                    FROM tasks
                    WHERE id = ? AND user_id = ?
                    """,
                    (task_id, uid),
                )
                updated_row = cursor.fetchone()
                if not updated_row or updated_row["status"] != "done" or not updated_row["completed_at"]:
                    raise TaskError(f"Read-back verification failed for task #{task_id} completion.")

                task_data = dict(updated_row)
                return {
                    "success": True,
                    "write_confirmed": True,
                    "operation": "complete_task",
                    "task": task_data,
                    "tasks_changed": [
                        {
                            "operation": "complete_task",
                            "task_id": task_data["id"],
                            "title": task_data["title"],
                            "confirmed": True,
                        }
                    ],
                }
        finally:
            connection.close()


def get_task_store() -> TaskStore:
    """Return default TaskStore instance."""
    return TaskStore()
