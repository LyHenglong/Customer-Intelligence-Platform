"""
Shared Postgres access for everything that reads the warehouse (training,
the recommender, the dashboard).

Exists mainly for one reason: reading a large table out of Postgres with
psycopg2 is a memory trap, and the obvious fix does not work.

`pd.read_sql(query, conn, chunksize=N)` *looks* like it streams, but
psycopg2's default cursor is client-side: libpq fetches and buffers the
entire result set before pandas is handed a single row. The chunksize
argument then chunks an already-materialized buffer, so it bounds pandas'
peak but not the process's. At 1,000,000 rows this fails outright with
"out of memory for query result" - and before that, it was the hidden
cause of an Airflow retrain task being SIGKILLed.

`stream_query` below uses a *named* (server-side) cursor instead, which
keeps the result set on the Postgres side and pulls it down in batches.
That is what actually bounds memory.
"""

from __future__ import annotations

import os

import pandas as pd
import psycopg2


def get_pg_conn():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "warehouse"),
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
    )


def stream_query(
    query: str,
    columns: list[str],
    batch_rows: int = 50_000,
    transform=None,
    params=None,
) -> pd.DataFrame:
    """Runs `query` through a server-side cursor and assembles a DataFrame
    from batches of `batch_rows`.

    `columns` must match the SELECT list order - server-side cursors return
    plain tuples, so column names are supplied rather than inferred.

    `transform`, if given, is applied to each batch DataFrame before it is
    accumulated. Use it to downcast dtypes (e.g. object -> category) while
    only one batch is in the expensive representation at a time, rather
    than converting once at the end when every batch is already resident.

    `params` is passed straight to psycopg2's execute, so callers filtering
    on a runtime value (a batch name, a customer id) can use %s / %(name)s
    placeholders instead of interpolating into the SQL string themselves.
    """
    conn = get_pg_conn()
    try:
        # Named cursor => server-side. withhold=False is fine: the whole
        # read happens inside this one transaction.
        with conn.cursor(name="warehouse_stream") as cur:
            cur.itersize = batch_rows
            cur.execute(query, params)

            frames = []
            while True:
                rows = cur.fetchmany(batch_rows)
                if not rows:
                    break
                batch = pd.DataFrame(rows, columns=columns)
                if transform is not None:
                    batch = transform(batch)
                frames.append(batch)

        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True)
    finally:
        conn.close()


def scalar_query(query: str):
    """Single-value query (row counts and similar)."""
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()
