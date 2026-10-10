import json
import logging
import re
from typing import Any, Dict, List, Optional
import sqlite3

from app.database import get_connection, get_onboarding_items
from app.models import FORBIDDEN_SQL_PATTERNS

logger = logging.getLogger(__name__)

USER_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")
MAX_PREFERENCES_BYTES = 1000
MAX_TEXT_FIELD_LEN = 100
VALID_ONBOARDING_STATUSES = {"not_started", "in_progress", "completed"}

FORBIDDEN_ROLES = {
    "admin", "administrator", "root", "superuser", "system", "sysadmin", "owner", "auth"
}
ROLE_ALLOWED_CHARS_REGEX = re.compile(r"^[a-zA-Z\s\-]+$")
MAX_ROLE_LEN = 40
MIN_ROLE_LEN = 2

SELF_ASSIGN_ROLE_DEPT_PATTERN = re.compile(
    r"\b(?:"
    r"my\s+role\s+is"
    r"|set\s+my\s+role\s+to"
    r"|my\s+department\s+is"
    r"|set\s+my\s+department\s+to"
    r"|i\s+work\s+(?:in|as)\b"
    r"|i\s+am\s+(?:an?\s+)?(?:[a-zA-Z\-]+\s+)*(?:engineer|developer|intern|lead|manager|analyst|designer|architect|consultant|specialist|worker|employee)\b"
    r")",
    re.IGNORECASE,
)

COMPLETION_INTENT_PATTERN = re.compile(
    r"\b(?:"
    r"i(?:'ve|\s+have)?\s+(?:completed|finished|did|done)\b"
    r"|mark\s+.+?\s+(?:as\s+)?(?:done|complete|completed|finished)\b"
    r")",
    re.IGNORECASE,
)

START_ONBOARDING_PATTERN = re.compile(
    r"\b(?:"
    r"i(?:'ve|\s+have)?\s+started\s+onboarding"
    r"|start(?:ed)?\s+onboarding"
    r")\b",
    re.IGNORECASE,
)

FINISH_ONBOARDING_PATTERN = re.compile(
    r"\b(?:"
    r"i(?:'ve|\s+have)?\s+(?:finished|completed)\s+onboarding"
    r"|(?:finish(?:ed)?|complete(?:d)?)\s+onboarding"
    r")\b",
    re.IGNORECASE,
)

TITLE_STOP_WORDS = {
    "and", "or", "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "as", "by", "complete", "completed", "setup", "set", "review", "submit", "meet", "first",
    "item", "items", "task", "tasks", "checklist"
}


def has_explicit_profile_statement(text: str) -> bool:
    """Checks whether user text explicitly states their own role or department."""
    if not text:
        return False
    return bool(SELF_ASSIGN_ROLE_DEPT_PATTERN.search(text))


def validate_profile_intent(
    question: Optional[str],
    role: Optional[str] = None,
    department: Optional[str] = None,
) -> bool:
    """
    Validates explicit intent for set_profile:
    1. User text must explicitly state their own role or department.
    2. Any specified role must appear verbatim (case-insensitive) in the user text.
    3. Any specified department must appear verbatim (case-insensitive) in the user text.
    """
    if not question:
        return False

    q_lower = question.lower()
    has_role = bool(role and str(role).strip())
    has_dept = bool(department and str(department).strip())

    if not has_role and not has_dept:
        return False

    if has_role:
        r_clean = str(role).strip().lower()
        if r_clean not in q_lower:
            return False
        role_explicit = (
            bool(re.search(r"\b(?:my\s+role\s+is|set\s+my\s+role\s+to|i\s+work\s+as)\b", question, re.I))
            or bool(re.search(r"\bi\s+am\s+(?:an?\s+)?" + re.escape(r_clean) + r"\b", question, re.I))
            or has_explicit_profile_statement(question)
        )
        if not role_explicit:
            return False

    if has_dept:
        d_clean = str(department).strip().lower()
        if not re.search(r"\b" + re.escape(d_clean) + r"\b", question, re.I):
            return False
        dept_explicit = (
            bool(re.search(r"\b(?:my\s+department\s+is|set\s+my\s+department\s+to|i\s+work\s+in)\b", question, re.I))
            or has_explicit_profile_statement(question)
        )
        if not dept_explicit:
            return False

    return True


def validate_item_completion_intent(
    question: Optional[str],
    item_id: int,
    item_title: Optional[str] = None,
) -> bool:
    """
    Validates explicit intent for mark_item_complete:
    1. User text must clearly state completion ("I completed/finished/did X", "mark X as done/complete").
    2. The item must match the user's text by title keyword or ID.
    """
    if not question:
        return False

    q_lower = question.lower()

    # 1. Check completion phrasing intent
    if not COMPLETION_INTENT_PATTERN.search(question):
        return False

    # 2. Check item match by ID
    id_pattern = re.compile(r"\b" + str(item_id) + r"\b")
    if id_pattern.search(question):
        return True

    # 3. Check item match by title keyword
    if not item_title:
        from app.database import get_connection
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT title FROM onboarding_items WHERE id = ?", (item_id,))
            row = cur.fetchone()
            if row:
                item_title = row["title"]
        except Exception:
            item_title = None
        finally:
            if "conn" in locals() and conn:
                conn.close()

    if item_title:
        words = [re.sub(r"[^\w]", "", w.lower()) for w in item_title.split()]
        keywords = [w for w in words if len(w) >= 2 and w not in TITLE_STOP_WORDS]
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
                return True

    return False


def validate_onboarding_status_intent(question: Optional[str], status: str) -> bool:
    """
    Validates explicit intent for set_onboarding_status:
    Allowed only on explicit user statements (e.g. 'I've started onboarding', 'I finished onboarding').
    """
    if not question:
        return False

    clean_status = (status or "").strip().lower()

    if clean_status == "in_progress":
        return bool(START_ONBOARDING_PATTERN.search(question))
    elif clean_status == "completed":
        return bool(FINISH_ONBOARDING_PATTERN.search(question))
    elif clean_status == "not_started":
        return bool(re.search(r"\b(?:haven't|have not)\s+started\s+onboarding\b", question, re.I))

    return False


def validate_role(role: Optional[str]) -> Optional[str]:
    """Validate role against length cap, allowlisted character pattern, and forbidden privileged names."""
    if role is None:
        return None
    r = role.strip()
    if not r:
        return None
    if len(r) < MIN_ROLE_LEN or len(r) > MAX_ROLE_LEN:
        raise MemoryError(f"Role must be between {MIN_ROLE_LEN} and {MAX_ROLE_LEN} characters.")
    if not ROLE_ALLOWED_CHARS_REGEX.match(r):
        raise MemoryError("Role contains disallowed characters. Only letters, spaces, and hyphens are allowed.")
    if r.lower() in FORBIDDEN_ROLES:
        raise MemoryError(f"Privileged role '{r}' is prohibited.")
    return r


def validate_department(dept: Optional[str]) -> Optional[str]:
    """Validate department against known departments loaded in SQLite."""
    if dept is None:
        return None
    d = dept.strip()
    if not d:
        return None
    from app.database import get_department_employees
    try:
        known = {row["name"].strip().lower() for row in get_department_employees()}
    except Exception:
        known = set()
    if not known:
        known = {"ai", "backend"}
    if d.lower() not in known:
        raise MemoryError(f"Unknown department '{d}'. Must be one of: {sorted(known)}.")
    return d


class MemoryError(ValueError):
    """Raised when memory operations violate constraints or validation."""
    pass


def validate_user_id(user_id: str) -> str:
    """Validate server-injected user_id."""
    if not isinstance(user_id, str):
        raise MemoryError("user_id must be a string.")
    clean = user_id.strip()
    if not USER_ID_REGEX.match(clean):
        raise MemoryError(f"Invalid user_id '{user_id}'. Must be 1-64 alphanumeric characters, dashes, or underscores.")
    return clean


def sanitize_text_field(field_name: str, value: Optional[str]) -> Optional[str]:
    """Sanitize and validate text fields (role, department), rejecting SQL injection."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise MemoryError(f"Field '{field_name}' must be a string.")
    s = value.strip()
    if not s:
        return None
    if len(s) > MAX_TEXT_FIELD_LEN:
        raise MemoryError(f"Field '{field_name}' exceeds maximum length of {MAX_TEXT_FIELD_LEN} characters.")
    s_lower = s.lower()
    for pat in FORBIDDEN_SQL_PATTERNS:
        if re.search(pat, s_lower):
            raise MemoryError(f"Forbidden SQL syntax or pattern detected in '{field_name}': '{value}'.")
    return s


def validate_preferences(preferences: Optional[Dict[str, Any]]) -> str:
    """Validate preferences dictionary and enforce size limit."""
    if preferences is None:
        return "{}"
    if not isinstance(preferences, dict):
        raise MemoryError("preferences must be a dictionary.")
    serialized = json.dumps(preferences)
    if len(serialized.encode("utf-8")) > MAX_PREFERENCES_BYTES:
        raise MemoryError(f"preferences exceed size limit of {MAX_PREFERENCES_BYTES} bytes.")
    return serialized


class MemoryStore:
    """
    Structured persistent memory manager for user state in SQLite.
    All operations are strictly parameterized and user-isolated.
    Provides read-back verification for all write operations.
    """

    def __init__(self, connection_factory=get_connection):
        self.get_connection = connection_factory

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user profile, returning default profile if not yet created."""
        uid = validate_user_id(user_id)
        connection = self.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(
                """
                SELECT user_id, role, department, onboarding_status, preferences, updated_at, last_agent_run
                FROM user_profile
                WHERE user_id = ?
                """,
                (uid,),
            )
            row = cursor.fetchone()
            if row:
                try:
                    pref_dict = json.loads(row["preferences"] or "{}")
                except Exception:
                    pref_dict = {}
                return {
                    "user_id": row["user_id"],
                    "role": row["role"],
                    "department": row["department"],
                    "onboarding_status": row["onboarding_status"],
                    "preferences": pref_dict,
                    "updated_at": row["updated_at"],
                    "last_agent_run": row["last_agent_run"],
                }
            return {
                "user_id": uid,
                "role": None,
                "department": None,
                "onboarding_status": "not_started",
                "preferences": {},
                "updated_at": None,
                "last_agent_run": None,
            }
        finally:
            connection.close()

    def set_profile(
        self,
        user_id: str,
        role: Optional[str] = None,
        department: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update user role, department, or preferences with immediate read-back verification.
        Does not allow setting arbitrary fields or user_id override.
        """
        uid = validate_user_id(user_id)
        clean_role = sanitize_text_field("role", role)
        clean_dept = sanitize_text_field("department", department)

        valid_role = validate_role(clean_role)
        valid_dept = validate_department(clean_dept)

        connection = self.get_connection()
        try:
            with connection:
                cursor = connection.cursor()
                cursor.execute("SELECT user_id, role, department, preferences FROM user_profile WHERE user_id = ?", (uid,))
                existing = cursor.fetchone()

                if existing:
                    new_role = valid_role if valid_role is not None else existing["role"]
                    new_dept = valid_dept if valid_dept is not None else existing["department"]
                    if preferences is not None:
                        new_prefs_str = validate_preferences(preferences)
                    else:
                        new_prefs_str = existing["preferences"] or "{}"

                    cursor.execute(
                        """
                        UPDATE user_profile
                        SET role = ?, department = ?, preferences = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = ?
                        """,
                        (new_role, new_dept, new_prefs_str, uid),
                    )
                else:
                    new_prefs_str = validate_preferences(preferences) if preferences is not None else "{}"
                    cursor.execute(
                        """
                        INSERT INTO user_profile (user_id, role, department, onboarding_status, preferences)
                        VALUES (?, ?, ?, 'not_started', ?)
                        """,
                        (uid, valid_role, valid_dept, new_prefs_str),
                    )

            # Read-back verification
            profile = self.get_profile(uid)
            write_confirmed = True
            if valid_role is not None and profile.get("role") != valid_role:
                write_confirmed = False
            if valid_dept is not None and profile.get("department") != valid_dept:
                write_confirmed = False

            return {
                "success": write_confirmed,
                "operation": "set_profile",
                "user_id": uid,
                "profile": profile,
                "write_confirmed": write_confirmed,
            }
        finally:
            connection.close()

    def set_onboarding_status(self, user_id: str, status: str) -> Dict[str, Any]:
        """Update onboarding status enum with immediate read-back verification."""
        uid = validate_user_id(user_id)
        clean_status = status.strip().lower()
        if clean_status not in VALID_ONBOARDING_STATUSES:
            raise MemoryError(
                f"Invalid onboarding status '{status}'. Must be one of {sorted(VALID_ONBOARDING_STATUSES)}."
            )

        connection = self.get_connection()
        try:
            with connection:
                cursor = connection.cursor()
                cursor.execute("SELECT user_id FROM user_profile WHERE user_id = ?", (uid,))
                if cursor.fetchone():
                    cursor.execute(
                        """
                        UPDATE user_profile
                        SET onboarding_status = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = ?
                        """,
                        (clean_status, uid),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO user_profile (user_id, onboarding_status)
                        VALUES (?, ?)
                        """,
                        (uid, clean_status),
                    )

            # Read-back verification
            profile = self.get_profile(uid)
            write_confirmed = (profile.get("onboarding_status") == clean_status)

            return {
                "success": write_confirmed,
                "operation": "set_onboarding_status",
                "user_id": uid,
                "status": clean_status,
                "write_confirmed": write_confirmed,
            }
        finally:
            connection.close()

    def mark_item_complete(self, user_id: str, item_id: int) -> Dict[str, Any]:
        """
        Mark an onboarding checklist item complete for the user with read-back verification.
        Validates item existence. Idempotent on duplicates.
        """
        uid = validate_user_id(user_id)
        if not isinstance(item_id, int):
            raise MemoryError(f"item_id must be an integer, got {type(item_id).__name__}.")

        connection = self.get_connection()
        cursor = connection.cursor()
        try:
            # 1. Verify item exists in onboarding_items
            cursor.execute("SELECT id, title, department, day_due FROM onboarding_items WHERE id = ?", (item_id,))
            item = cursor.fetchone()
            if not item:
                return {
                    "success": False,
                    "operation": "mark_item_complete",
                    "user_id": uid,
                    "item_id": item_id,
                    "error": f"Onboarding item with id {item_id} does not exist.",
                    "write_confirmed": False,
                }

            # 2. Insert into user_completed_items
            with connection:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO user_completed_items (user_id, item_id)
                    VALUES (?, ?)
                    """,
                    (uid, item_id),
                )
                # Update status to in_progress if not_started
                cursor.execute(
                    """
                    INSERT INTO user_profile (user_id, onboarding_status)
                    VALUES (?, 'in_progress')
                    ON CONFLICT(user_id) DO UPDATE SET
                        onboarding_status = CASE
                            WHEN onboarding_status = 'not_started' THEN 'in_progress'
                            ELSE onboarding_status
                        END,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (uid,),
                )

            # 3. Read-back verification
            cursor.execute(
                "SELECT 1 FROM user_completed_items WHERE user_id = ? AND item_id = ?",
                (uid, item_id),
            )
            verified = cursor.fetchone() is not None

            return {
                "success": verified,
                "operation": "mark_item_complete",
                "user_id": uid,
                "item_id": item_id,
                "item_title": item["title"],
                "write_confirmed": verified,
            }
        finally:
            connection.close()

    def get_progress(self, user_id: str) -> Dict[str, Any]:
        """
        Calculates onboarding progress DETERMINISTICALLY IN PYTHON:
        compares total applicable checklist items against completed items.
        Zero LLM calls.
        """
        uid = validate_user_id(user_id)
        profile = self.get_profile(uid)
        user_dept = profile.get("department")

        # 1. Fetch applicable checklist items (company-wide + user's department)
        applicable_items = get_onboarding_items(user_dept)

        # 2. Fetch completed items for this user
        connection = self.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(
                """
                SELECT item_id, completed_at
                FROM user_completed_items
                WHERE user_id = ?
                ORDER BY completed_at ASC
                """,
                (uid,),
            )
            completed_rows = cursor.fetchall()
            completed_ids = {r["item_id"] for r in completed_rows}

            # 3. Deterministic Python comparison
            completed_items = [it for it in applicable_items if it["id"] in completed_ids]
            remaining_items = [it for it in applicable_items if it["id"] not in completed_ids]

            total_count = len(applicable_items)
            completed_count = len(completed_items)
            remaining_count = len(remaining_items)
            pct = round((completed_count / max(1, total_count)) * 100, 1)

            # Auto-update status to completed if all done
            status = profile.get("onboarding_status", "not_started")
            if total_count > 0 and remaining_count == 0 and status != "completed":
                self.set_onboarding_status(uid, "completed")
                status = "completed"

            return {
                "user_id": uid,
                "role": profile.get("role"),
                "department": user_dept,
                "onboarding_status": status,
                "total_items": total_count,
                "completed_count": completed_count,
                "remaining_count": remaining_count,
                "progress_percent": pct,
                "completed_items": completed_items,
                "remaining_items": remaining_items,
            }
        finally:
            connection.close()


# Shared memory store instance
_default_memory_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    global _default_memory_store
    if _default_memory_store is None:
        _default_memory_store = MemoryStore()
    return _default_memory_store
