# sql_agents.py
import os
import re
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from azure_client import get_client

# ---- Load .env from project folder ----
here = Path(__file__).resolve().parent
load_dotenv(dotenv_path=str(here / ".env"), override=True)

CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")

# ---- Local SQLite DB path (persistent) ----
DB_DIR = here / "data" / "sqlite"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "local.db"


# -------------------------------
# Helpers
# -------------------------------
def _ensure_db() -> None:
    """Create the SQLite DB file if it doesn't exist."""
    if not DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        conn.close()


def _extract_sql(text: str) -> str:
    """
    Extract SQL from model output. Prefer fenced blocks (```sql ... ``` or ``` ... ```).
    Fallback to the whole text if no fence found.
    """
    # ```sql ... ``` or ``` ... ```
    blocks = re.findall(r"```(?:sql)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if blocks:
        return blocks[0].strip()
    return text.strip()


def _ask_model_for_sql(user_query: str, dialect: str = "SQLite") -> str:
    """
    Ask Azure chat model to generate a single SQL statement.
    We keep the temperature implicit (default=1 on your deployment),
    because some deployments reject custom temperatures.
    """
    client = get_client()
    system = (
        f"You are an expert {dialect} SQL assistant. "
        f"Generate ONE {dialect} SQL statement that addresses the user's request. "
        f"If DDL is needed (CREATE/ALTER), produce it directly. "
        f"Do not include explanations unless asked; prefer a concise code block."
    )
    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_query}
        ],
        max_completion_tokens=800
    )
    return _extract_sql(resp.choices[0].message.content)


def _format_table(cursor: sqlite3.Cursor, rows: List[tuple], max_rows: int = 200) -> str:
    """
    Turn SELECT results into a readable, monospaced table.
    Truncates after max_rows to avoid giant outputs in the UI.
    """
    if cursor.description is None:
        return "No result set."
    headers = [d[0] for d in cursor.description]

    # Convert to strings & compute column widths
    str_rows = [[("" if v is None else str(v)) for v in r] for r in rows[:max_rows]]
    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    # Build table
    def fmt_row(vals):
        return " | ".join(val.ljust(widths[i]) for i, val in enumerate(vals))

    line = "-+-".join("-" * w for w in widths)
    out = []
    out.append("```\n")
    out.append(fmt_row(headers))
    out.append(line)
    for r in str_rows:
        out.append(fmt_row(r))
    if len(rows) > max_rows:
        out.append(f"... ({len(rows) - max_rows} more rows truncated)")
    out.append("\n```")
    return "\n".join(out)


def _is_select(sql: str) -> bool:
    return sql.strip().lower().startswith("select")


def _is_pragma(sql: str) -> bool:
    return sql.strip().lower().startswith("pragma")


def _disallow_dangerous(sql: str) -> Optional[str]:
    """
    Optionally block very dangerous commands.
    You can relax these if you trust your prompts and use case.
    """
    bad = ["drop database", "attach database", "vacuum into"]
    sql_low = sql.lower()
    for token in bad:
        if token in sql_low:
            return f"Blocked dangerous statement: '{token}'."
    return None


# -------------------------------
# Public entry points (used by UI)
# -------------------------------
def run_sqlite_agent(user_query: str) -> str:
    if not user_query or not user_query.strip():
        return "Type a query first."

    try:
        _ensure_db()
        sql = _ask_model_for_sql(user_query, dialect="SQLite")

        # DEBUG LOGGING
        print("MODEL GENERATED SQL:", sql)  # <-- ADD THIS

        if not sql.strip():
            return "The model returned empty SQL. Try rephrasing your query."

        # quick sanitization
        blocked = _disallow_dangerous(sql)
        if blocked:
            return f"[SQLite agent] {blocked}\n\nSQL proposed:\n```\n{sql}\n```"

        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        cur = conn.cursor()

        if _is_select(sql):
            cur.execute(sql)
            rows = cur.fetchall()
            out = _format_table(cur, rows)
            cur.close()
            conn.close()
            return f"{out}\n\nSQL executed:\n```\n{sql}\n```"

        elif _is_pragma(sql):
            # Allow safe PRAGMA reads like schema introspection
            cur.execute(sql)
            rows = cur.fetchall()
            out = _format_table(cur, rows)
            cur.close()
            conn.close()
            return f"{out}\n\nSQL executed:\n```\n{sql}\n```"

        else:
            cur.execute(sql)
            conn.commit()
            affected = cur.rowcount  # may be -1 for DDL
            cur.close()
            conn.close()
            return f"Executed successfully. Rows affected: {affected}\n\nSQL executed:\n```\n{sql}\n```"

    except Exception as e:
        # Return both the SQL and the error for easy debugging
        return f"[SQLite agent error]\nSQL proposed:\n```\n{sql if 'sql' in locals() else '<none>'}\n```\nError: {e}"


def run_postgres_agent(user_query: str) -> str:
    """
    1) Ask model for PostgreSQL SQL (no execution).
    2) Return the SQL to the UI.
    """
    if not user_query or not user_query.strip():
        return "Type a query first."

    try:
        sql = _ask_model_for_sql(user_query, dialect="PostgreSQL")
        return f"```sql\n{sql}\n```"
    except Exception as e:
        return f"[Postgres agent error] {e}"
