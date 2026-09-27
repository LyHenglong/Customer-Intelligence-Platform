"""SQL safety guardrails for the AI assistant's controlled SQL tool
(src/ai/tools/sql_tool.py).

The LLM never gets raw Postgres credentials or an unrestricted connection
(docs/ai_layer_build_plan.md section 9) - every candidate
query is validated here first: single statement, no comments (a classic
way to smuggle a second statement past a naive check), SELECT-only, no
DDL/DML keywords, no system/admin functions, and restricted to an
explicit table allowlist. This is defense in depth alongside sql_tool.py's
own read-only session, statement timeout and (when configured) a
least-privilege database role, not the only layer.

Table references are found by parsing the query with sqlglot, not by
regex: a regex over "FROM x"/"JOIN x" misses comma joins
("FROM allowed, other") and unqualified tables in subqueries
("(SELECT ... FROM pg_user)"), both of which used to pass.
"""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from src.ai.config import SQL_ALLOWED_TABLES


class SQLSafetyError(ValueError):
    """Raised when a candidate query fails a safety check. The message is
    safe to surface back to a caller (agent or user) - it never echoes
    credentials or connection details."""


_FORBIDDEN_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE",
    "COPY",
    "CALL",
    "EXECUTE",
    "MERGE",
    "VACUUM",
    "REINDEX",
    "REFRESH",
    "LISTEN",
    "NOTIFY",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
_SELECT_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)

# Functions that read server files, reach other databases, inspect or change
# server configuration, or render whole tables - none of which an analytics
# question needs, and several of which would sidestep the table allowlist.
_FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink")
_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "set_config",
        "current_setting",
        "version",
        "inet_server_addr",
        "inet_server_port",
        "query_to_xml",
        "query_to_xml_and_xmlschema",
        "table_to_xml",
        "cursor_to_xml",
        "schema_to_xml",
        "database_to_xml",
        "txid_current",
    }
)


def _function_names(tree: exp.Expression) -> set[str]:
    names = set()
    for func in tree.find_all(exp.Func):
        if isinstance(func, exp.Anonymous):
            names.add(func.name.lower())
        else:
            names.add(func.sql_name().lower())
    return names


def _table_references(tree: exp.Expression) -> set[str]:
    """Every table the query reads, schema-qualified and lowercased. A FROM
    source that is not a plain table (a table-valued function such as
    generate_series) is reported by its SQL text so the allowlist rejects
    it rather than silently ignoring it."""
    tables = set()
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            tables.add(table.sql(dialect="postgres").lower())
        elif table.db:
            tables.add(f"{table.db}.{table.name}".lower())
        else:
            tables.add(table.name.lower())
    return tables


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

    try:
        statements = sqlglot.parse(body, read="postgres")
    except ParseError as e:
        raise SQLSafetyError(f"could not parse query: {e.errors[0]['description'] if e.errors else e}") from e
    if len(statements) != 1 or statements[0] is None:
        raise SQLSafetyError("multiple SQL statements are not allowed")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise SQLSafetyError("only SELECT statements are allowed")

    for name in sorted(_function_names(tree)):
        if name in _FORBIDDEN_FUNCTIONS or name.startswith(_FORBIDDEN_FUNCTION_PREFIXES):
            raise SQLSafetyError(f"forbidden function: {name}")

    tables = _table_references(tree)
    if not tables:
        raise SQLSafetyError("could not identify a FROM/JOIN table reference")

    allowed = {t.lower() for t in SQL_ALLOWED_TABLES}
    disallowed = tables - allowed
    if disallowed:
        raise SQLSafetyError(
            f"query references table(s) not on the allowlist: {sorted(disallowed)} (allowed: {sorted(allowed)})"
        )

    return body
