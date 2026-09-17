"""SQL safety guardrails for the AI assistant's controlled SQL tool
(src/ai/tools/sql_tool.py).

The LLM never gets raw Postgres credentials or an unrestricted connection
(AI_Customer_Intelligence_Claude_Code_Plan.md section 9) - every candidate
query is validated here first: single statement, no comments (a classic
way to smuggle a second statement past a naive check), SELECT-only, no
DDL/DML keywords, and restricted to an explicit table allowlist. This is
defense in depth alongside sql_tool.py's own read-only session and
statement timeout, not the only layer.
"""

from __future__ import annotations

import re

from src.ai.config import SQL_ALLOWED_TABLES


class SQLSafetyError(ValueError):
    """Raised when a candidate query fails a safety check. The message is
    safe to surface back to a caller (agent or user) - it never echoes
    credentials or connection details."""


_FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE",
    "GRANT", "REVOKE", "COPY", "CALL", "EXECUTE", "MERGE", "VACUUM",
    "REINDEX", "REFRESH", "LISTEN", "NOTIFY",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
_SELECT_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE
)


def validate_sql(query: str) -> str:
    """Validates `query`, returning it stripped of surrounding whitespace
    and any single trailing semicolon. Raises SQLSafetyError on the first
    violation found."""
    if not query or not query.strip():
        raise SQLSafetyError("empty query")

    stripped = query.strip()

    if "--" in stripped or "/*" in stripped:
        raise SQLSafetyError("SQL comments are not allowed (can hide a second statement)")

    # A single trailing semicolon is fine; one anywhere else means more
    # than one statement.
    body = stripped[:-1] if stripped.endswith(";") else stripped
    if ";" in body:
        raise SQLSafetyError("multiple SQL statements are not allowed")

    if not _SELECT_RE.match(body):
        raise SQLSafetyError("only SELECT statements are allowed")

    forbidden = _FORBIDDEN_RE.search(body)
    if forbidden:
        raise SQLSafetyError(f"forbidden keyword: {forbidden.group(1).upper()}")

    tables = {m.group(1).lower() for m in _TABLE_REF_RE.finditer(body)}
    if not tables:
        raise SQLSafetyError("could not identify a FROM/JOIN table reference")

    allowed = {t.lower() for t in SQL_ALLOWED_TABLES}
    disallowed = tables - allowed
    if disallowed:
        raise SQLSafetyError(
            f"query references table(s) not on the allowlist: {sorted(disallowed)} "
            f"(allowed: {sorted(allowed)})"
        )

    return body
