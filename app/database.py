import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DATABASE_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge.db"


def get_connection() -> sqlite3.Connection:
    """Create and return a SQLite connection with Row factory enabled."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    """Initialize the SQLite database schema if tables do not exist."""
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS company_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            employees INTEGER NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leave_policy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            leave_type TEXT NOT NULL,
            days INTEGER NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS onboarding_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            department TEXT,
            day_due INTEGER NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            user_id TEXT PRIMARY KEY,
            role TEXT,
            department TEXT,
            onboarding_status TEXT NOT NULL DEFAULT 'not_started',
            preferences TEXT DEFAULT '{}',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_agent_run TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_completed_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            item_id INTEGER NOT NULL,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, item_id),
            FOREIGN KEY(item_id) REFERENCES onboarding_items(id)
        )
    """)

    connection.commit()
    connection.close()


def seed_database() -> None:
    """Seed the database with baseline company data if tables are empty."""
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM company_info")
    company_count = cursor.fetchone()[0]

    if company_count == 0:
        company_data = [
            ("company", "name", "ABC Technologies"),
            ("company", "location", "Bangalore"),
            ("company", "employees", "250"),
        ]

        cursor.executemany(
            "INSERT INTO company_info (category, key, value) VALUES (?, ?, ?)",
            company_data,
        )

    cursor.execute("SELECT COUNT(*) FROM departments")
    department_count = cursor.fetchone()[0]

    if department_count == 0:
        departments = [
            ("AI", 40),
            ("Backend", 60),
            ("Frontend", 35),
            ("Data Science", 30),
        ]

        cursor.executemany(
            "INSERT INTO departments (name, employees) VALUES (?, ?)",
            departments,
        )

    cursor.execute("SELECT COUNT(*) FROM leave_policy")
    leave_count = cursor.fetchone()[0]

    if leave_count == 0:
        leave_data = [
            ("Annual Leave", 20),
            ("Sick Leave", 10),
            ("Casual Leave", 12),
        ]

        cursor.executemany(
            "INSERT INTO leave_policy (leave_type, days) VALUES (?, ?)",
            leave_data,
        )

    cursor.execute("SELECT COUNT(*) FROM onboarding_items")
    onboarding_count = cursor.fetchone()[0]

    if onboarding_count == 0:
        onboarding_seed = [
            ("Complete Security Training", "Complete mandatory cybersecurity, data privacy, and phishing compliance modules.", None, 1),
            ("Submit HR and Tax Documents", "Upload identity proof, tax withholding forms, and direct deposit banking details.", None, 1),
            ("Set Up Company Email and Slack", "Configure 2FA security authentication and join essential team channels.", None, 2),
            ("Backend Technical Setup", "Clone repositories, configure Python virtualenv, and test database connectivity.", "Backend", 2),
            ("AI Model Sandbox Setup", "Verify Ollama installation, local model weights, and prompt evaluation tooling.", "AI", 2),
            ("Meet Team Lead and Buddy", "Schedule introductory 1-on-1 sessions with department manager and onboarding buddy.", None, 3),
            ("Review Production Architecture", "Read system architecture blueprints, security policies, and deployment runbooks.", "Backend", 4),
            ("First Sprint Planning", "Participate in sprint backlog grooming and assign first starter implementation task.", None, 5),
        ]

        cursor.executemany(
            "INSERT INTO onboarding_items (title, description, department, day_due) VALUES (?, ?, ?, ?)",
            onboarding_seed,
        )

    connection.commit()
    connection.close()


def get_onboarding_items(department: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve onboarding checklist items safely using parameterized queries.
    If department is provided, returns general items (department IS NULL)
    plus items tailored to that specific department.
    If department is None or 'all', returns all items.
    """
    connection = get_connection()
    cursor = connection.cursor()
    try:
        if department and department.strip().lower() not in ["all", "all_departments", "all_depts", "none", "null"]:
            clean_dept = department.strip().lower()
            cursor.execute(
                """
                SELECT id, title, description, department, day_due
                FROM onboarding_items
                WHERE department IS NULL OR LOWER(department) = ?
                ORDER BY day_due ASC, id ASC
                """,
                (clean_dept,),
            )
        else:
            cursor.execute(
                "SELECT id, title, description, department, day_due FROM onboarding_items ORDER BY day_due ASC, id ASC"
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def check_database_health() -> bool:
    """Check if the SQLite database is reachable and queryable."""
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        connection.close()
        return bool(result and result[0] == 1)
    except Exception:
        return False


def get_current_company_name() -> str:
    """Retrieve the current company name from SQLite, or fallback to 'the company'."""
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT value FROM company_info WHERE LOWER(key) = 'name' LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else "the company"
    finally:
        connection.close()


def get_all_department_names() -> List[str]:
    """Retrieve all distinct department names currently stored in SQLite."""
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT name FROM departments ORDER BY id ASC")
        return [row[0] for row in cursor.fetchall()]
    finally:
        connection.close()


def get_all_leave_types() -> List[str]:
    """Retrieve all distinct leave types currently stored in SQLite."""
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT leave_type FROM leave_policy ORDER BY id ASC")
        return [row[0] for row in cursor.fetchall()]
    finally:
        connection.close()


def get_company_info(key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve company information safely using parameterized queries.
    If key is None or an 'all' variant, returns all company info records.
    """
    connection = get_connection()
    cursor = connection.cursor()

    try:
        if key and key.strip().lower() not in ["all", "all_company", "company"]:
            clean_key = key.strip().lower()
            cursor.execute(
                "SELECT category, key, value FROM company_info WHERE LOWER(key) = ?",
                (clean_key,),
            )
            rows = cursor.fetchall()
            if not rows:
                # Try partial match if exact key match was not found
                cursor.execute(
                    "SELECT category, key, value FROM company_info WHERE LOWER(key) LIKE ?",
                    (f"%{clean_key}%",),
                )
                rows = cursor.fetchall()
        else:
            cursor.execute("SELECT category, key, value FROM company_info")
            rows = cursor.fetchall()

        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_department_employees(dept_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve department employee counts safely using parameterized queries.
    If dept_name is None or an 'all' variant, returns all departments.
    """
    connection = get_connection()
    cursor = connection.cursor()

    try:
        if dept_name and dept_name.strip().lower() not in ["all", "all_depts", "all departments", "all_departments"]:
            clean_dept = dept_name.strip().lower()
            cursor.execute(
                "SELECT name, employees FROM departments WHERE LOWER(name) = ?",
                (clean_dept,),
            )
            rows = cursor.fetchall()
            if not rows:
                # Try partial match (e.g. 'ai' or 'backend' in 'data science')
                cursor.execute(
                    "SELECT name, employees FROM departments WHERE LOWER(name) LIKE ?",
                    (f"%{clean_dept}%",),
                )
                rows = cursor.fetchall()
        else:
            cursor.execute("SELECT name, employees FROM departments")
            rows = cursor.fetchall()

        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_department_total_headcount() -> int:
    """Return total number of employees summed across all departments."""
    connection = get_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT SUM(employees) FROM departments")
        result = cursor.fetchone()
        return int(result[0]) if result and result[0] is not None else 0
    finally:
        connection.close()


def get_leave_policy(leave_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieve leave policy allowances safely using parameterized queries.
    If leave_type is None or an 'all' variant, returns all leave policies.
    """
    connection = get_connection()
    cursor = connection.cursor()

    try:
        if leave_type and leave_type.strip().lower() not in ["all", "all_leaves", "all leaves", "all_leave"]:
            clean_leave = leave_type.strip().lower()
            # Normalize common variants: 'annual' -> 'annual leave'
            if not clean_leave.endswith("leave"):
                clean_leave_with_suffix = f"{clean_leave} leave"
            else:
                clean_leave_with_suffix = clean_leave

            cursor.execute(
                "SELECT leave_type, days FROM leave_policy WHERE LOWER(leave_type) = ? OR LOWER(leave_type) = ?",
                (clean_leave, clean_leave_with_suffix),
            )
            rows = cursor.fetchall()

            if not rows:
                # Partial match fallback
                cursor.execute(
                    "SELECT leave_type, days FROM leave_policy WHERE LOWER(leave_type) LIKE ?",
                    (f"%{clean_leave.replace('leave', '').strip()}%",),
                )
                rows = cursor.fetchall()
        else:
            cursor.execute("SELECT leave_type, days FROM leave_policy")
            rows = cursor.fetchall()

        return [dict(row) for row in rows]
    finally:
        connection.close()


def get_full_knowledge() -> Dict[str, Any]:
    """
    Retrieve the entire currently loaded knowledge base from SQLite.
    Returns structured dictionary with 'company', 'departments', and 'leave_policy'.
    """
    connection = get_connection()
    cursor = connection.cursor()

    try:
        # 1. Company
        cursor.execute("SELECT key, value FROM company_info")
        company_rows = cursor.fetchall()
        company_dict: Dict[str, Any] = {}
        for row in company_rows:
            key = row["key"]
            val = row["value"]
            if key == "employees" and val.isdigit():
                company_dict[key] = int(val)
            else:
                company_dict[key] = val

        # 2. Departments
        cursor.execute("SELECT name, employees FROM departments ORDER BY id ASC")
        departments = [
            {"name": row["name"], "employees": row["employees"]}
            for row in cursor.fetchall()
        ]

        # 3. Leave Policy
        cursor.execute("SELECT leave_type, days FROM leave_policy ORDER BY id ASC")
        leave_policy = [
            {"leave_type": row["leave_type"], "days": row["days"]}
            for row in cursor.fetchall()
        ]

        # 4. Onboarding Items
        cursor.execute("SELECT id, title, description, department, day_due FROM onboarding_items ORDER BY day_due ASC, id ASC")
        onboarding_items = [
            {
                "id": row["id"],
                "title": row["title"],
                "description": row["description"],
                "department": row["department"],
                "day_due": row["day_due"],
            }
            for row in cursor.fetchall()
        ]

        return {
            "company": company_dict,
            "departments": departments,
            "leave_policy": leave_policy,
            "onboarding_items": onboarding_items,
        }
    finally:
        connection.close()


def clear_and_ingest_knowledge(
    company: Dict[str, Any],
    departments: List[Dict[str, Any]],
    leave_policy: List[Dict[str, Any]],
    onboarding_items: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Atomically clears existing knowledge tables and inserts validated knowledge records.
    Uses strictly parameterized queries; never executes raw/user-supplied SQL strings.
    If onboarding_items is provided, atomically replaces the checklist table as well.
    """
    connection = get_connection()
    try:
        with connection:
            cursor = connection.cursor()

            # 1. Wipe existing tables
            cursor.execute("DELETE FROM company_info")
            cursor.execute("DELETE FROM departments")
            cursor.execute("DELETE FROM leave_policy")
            if onboarding_items is not None:
                cursor.execute("DELETE FROM onboarding_items")

            # 2. Insert company_info
            company_data = [
                ("company", "name", str(company["name"]).strip()),
                ("company", "location", str(company["location"]).strip()),
                ("company", "employees", str(company["employees"]).strip()),
            ]
            cursor.executemany(
                "INSERT INTO company_info (category, key, value) VALUES (?, ?, ?)",
                company_data,
            )

            # 3. Insert departments
            dept_data = [
                (str(d["name"]).strip(), int(d["employees"]))
                for d in departments
            ]
            if dept_data:
                cursor.executemany(
                    "INSERT INTO departments (name, employees) VALUES (?, ?)",
                    dept_data,
                )

            # 4. Insert leave_policy
            leave_data = [
                (str(l["leave_type"]).strip(), int(l["days"]))
                for l in leave_policy
            ]
            if leave_data:
                cursor.executemany(
                    "INSERT INTO leave_policy (leave_type, days) VALUES (?, ?)",
                    leave_data,
                )

            # 5. Insert onboarding_items if provided
            if onboarding_items:
                item_data = [
                    (
                        str(item["title"]).strip(),
                        str(item["description"]).strip(),
                        str(item["department"]).strip() if item.get("department") else None,
                        int(item["day_due"]),
                    )
                    for item in onboarding_items
                ]
                cursor.executemany(
                    "INSERT INTO onboarding_items (title, description, department, day_due) VALUES (?, ?, ?, ?)",
                    item_data,
                )
    finally:
        connection.close()