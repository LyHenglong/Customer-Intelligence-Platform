"""
Picks up the next unprocessed simulated batch file (data/raw/batch_NNN.csv),
loads and validates it with DuckDB, writes a raw pass-through table plus a
lightly-cleaned + feature-engineered table into Postgres, and records the
batch in ingestion_log so the same file is never reprocessed.

Usage:
    python -m src.ingest.batch_loader                 # next unprocessed batch
    python -m src.ingest.batch_loader --batch batch_001.csv   # explicit batch
    python -m src.ingest.batch_loader --all            # process all remaining batches
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
from pathlib import Path

import duckdb
import psycopg2
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("batch_loader")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

RAW_COLUMNS = [
    "customer_id", "signup_date", "age", "gender", "annual_income", "education",
    "marital_status", "dependents", "tenure", "contract", "payment_method",
    "paperless_billing", "senior_citizen", "monthlycharges", "totalcharges",
    "num_services", "has_phone_service", "has_internet_service",
    "has_online_security", "has_online_backup", "has_device_protection",
    "has_tech_support", "has_streaming_tv", "has_streaming_movies",
    "customer_satisfaction", "num_complaints", "num_service_calls",
    "late_payments", "avg_monthly_gb", "days_since_last_interaction",
    "credit_score", "churn",
]

CLEANED_COLUMNS = RAW_COLUMNS + [
    "tenure_years", "total_active_services", "avg_gb_per_service",
]

_SERVICE_FLAGS = [
    "has_phone_service", "has_internet_service", "has_online_security",
    "has_online_backup", "has_device_protection", "has_tech_support",
    "has_streaming_tv", "has_streaming_movies",
]


def get_pg_conn():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ.get("POSTGRES_DB", "warehouse"),
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
    )


def already_loaded_batches(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT batch_file FROM ingestion_log")
        return {r[0] for r in cur.fetchall()}


def next_unprocessed_batch(conn) -> Path | None:
    done = already_loaded_batches(conn)
    for path in sorted(RAW_DIR.glob("batch_*.csv")):
        if path.name not in done:
            return path
    return None


def _try_cast_select(alias: str = "") -> str:
    prefix = f"{alias}." if alias else ""
    return f"""
        TRY_CAST({prefix}customer_id AS VARCHAR)                    AS customer_id,
        TRY_CAST({prefix}signup_date AS TIMESTAMP)                  AS signup_date,
        TRY_CAST({prefix}age AS INTEGER)                            AS age,
        TRY_CAST({prefix}gender AS VARCHAR)                         AS gender,
        TRY_CAST({prefix}annual_income AS DOUBLE)                   AS annual_income,
        TRY_CAST({prefix}education AS VARCHAR)                      AS education,
        TRY_CAST({prefix}marital_status AS VARCHAR)                 AS marital_status,
        TRY_CAST({prefix}dependents AS INTEGER)                     AS dependents,
        TRY_CAST({prefix}tenure AS INTEGER)                         AS tenure,
        TRY_CAST({prefix}contract AS VARCHAR)                       AS contract,
        TRY_CAST({prefix}payment_method AS VARCHAR)                 AS payment_method,
        TRY_CAST({prefix}paperless_billing AS VARCHAR)              AS paperless_billing,
        TRY_CAST({prefix}senior_citizen AS INTEGER)                 AS senior_citizen,
        TRY_CAST({prefix}monthlycharges AS DOUBLE)                  AS monthlycharges,
        TRY_CAST({prefix}totalcharges AS DOUBLE)                    AS totalcharges,
        TRY_CAST({prefix}num_services AS INTEGER)                   AS num_services,
        TRY_CAST({prefix}has_phone_service AS INTEGER)              AS has_phone_service,
        TRY_CAST({prefix}has_internet_service AS INTEGER)           AS has_internet_service,
        TRY_CAST({prefix}has_online_security AS INTEGER)            AS has_online_security,
        TRY_CAST({prefix}has_online_backup AS INTEGER)              AS has_online_backup,
        TRY_CAST({prefix}has_device_protection AS INTEGER)          AS has_device_protection,
        TRY_CAST({prefix}has_tech_support AS INTEGER)                AS has_tech_support,
        TRY_CAST({prefix}has_streaming_tv AS INTEGER)                AS has_streaming_tv,
        TRY_CAST({prefix}has_streaming_movies AS INTEGER)            AS has_streaming_movies,
        TRY_CAST({prefix}customer_satisfaction AS DOUBLE)           AS customer_satisfaction,
        TRY_CAST({prefix}num_complaints AS DOUBLE)                  AS num_complaints,
        TRY_CAST({prefix}num_service_calls AS INTEGER)              AS num_service_calls,
        TRY_CAST({prefix}late_payments AS INTEGER)                  AS late_payments,
        TRY_CAST({prefix}avg_monthly_gb AS DOUBLE)                  AS avg_monthly_gb,
        TRY_CAST({prefix}days_since_last_interaction AS INTEGER)    AS days_since_last_interaction,
        TRY_CAST({prefix}credit_score AS DOUBLE)                    AS credit_score,
        TRY_CAST({prefix}churn AS INTEGER)                          AS churn
    """


def process_batch(batch_path: Path) -> tuple[list[tuple], list[tuple], dict]:
    """Load + validate + engineer features for one batch file using DuckDB.

    Returns (raw_rows, cleaned_rows, stats). Rows are materialized here
    (fetchall) before the in-memory DuckDB connection goes out of scope
    and closes - the underlying relations are not valid past that point.
    """
    con = duckdb.connect(database=":memory:")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE typed_batch AS
        SELECT {_try_cast_select()}
        FROM read_csv_auto(?, ALL_VARCHAR=TRUE, header=True)
        """,
        [str(batch_path)],
    )

    total_rows = con.execute("SELECT COUNT(*) FROM typed_batch").fetchone()[0]

    # Rows where required fields failed to cast, or churn is outside {0,1},
    # or tenure is negative, are dropped from the cleaned layer as a basic
    # data-quality gate (kept in the raw layer for auditability).
    invalid = con.execute(
        """
        SELECT COUNT(*) FROM typed_batch
        WHERE customer_id IS NULL
           OR churn NOT IN (0, 1)
           OR tenure < 0
        """
    ).fetchone()[0]

    raw_rows = con.sql(f"SELECT {', '.join(RAW_COLUMNS)} FROM typed_batch").fetchall()

    service_sum = " + ".join(f"COALESCE({c}, 0)" for c in _SERVICE_FLAGS)
    cleaned_rows = con.sql(
        f"""
        SELECT
            customer_id, signup_date, age, gender, annual_income, education,
            marital_status, dependents, tenure, contract, payment_method,
            (paperless_billing = 'Yes')  AS paperless_billing,
            (senior_citizen = 1)         AS senior_citizen,
            monthlycharges, totalcharges, num_services,
            (has_phone_service = 1)      AS has_phone_service,
            (has_internet_service = 1)   AS has_internet_service,
            (has_online_security = 1)    AS has_online_security,
            (has_online_backup = 1)      AS has_online_backup,
            (has_device_protection = 1)  AS has_device_protection,
            (has_tech_support = 1)       AS has_tech_support,
            (has_streaming_tv = 1)       AS has_streaming_tv,
            (has_streaming_movies = 1)   AS has_streaming_movies,
            customer_satisfaction, num_complaints, num_service_calls,
            late_payments, avg_monthly_gb, days_since_last_interaction,
            credit_score, churn,
            ROUND(tenure / 12.0, 2) AS tenure_years,
            ({service_sum}) AS total_active_services,
            CASE WHEN ({service_sum}) > 0
                 THEN ROUND(avg_monthly_gb / ({service_sum}), 2)
                 ELSE NULL END AS avg_gb_per_service
        FROM typed_batch
        WHERE customer_id IS NOT NULL
          AND churn IN (0, 1)
          AND tenure >= 0
        """
    ).fetchall()

    con.close()

    stats = {
        "total_rows": total_rows,
        "invalid_rows_dropped": invalid,
        "loaded_rows": total_rows - invalid,
    }
    return raw_rows, cleaned_rows, stats


def _copy_rows(conn, rows: list[tuple], table: str, columns: list[str], batch_name: str) -> None:
    """Streams materialized rows into Postgres via COPY FROM STDIN."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in rows:
        writer.writerow(["\\N" if v is None else v for v in row] + [batch_name])
    buf.seek(0)
    cols = columns + ["source_batch"]
    with conn.cursor() as cur:
        cur.copy_expert(
            f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')",
            buf,
        )


def load_batch(batch_path: Path, skip_raw: bool = False) -> None:
    log.info("Processing %s", batch_path.name)
    raw_rows, cleaned_rows, stats = process_batch(batch_path)
    log.info(
        "  total_rows=%d invalid_dropped=%d loaded_rows=%d",
        stats["total_rows"], stats["invalid_rows_dropped"], stats["loaded_rows"],
    )

    conn = get_pg_conn()
    try:
        if not skip_raw:
            _copy_rows(conn, raw_rows, "public.raw_customers", RAW_COLUMNS, batch_path.name)
        _copy_rows(conn, cleaned_rows, "public.customers_cleaned", CLEANED_COLUMNS, batch_path.name)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ingestion_log (batch_file, rows_loaded, status) VALUES (%s, %s, %s)",
                (batch_path.name, stats["loaded_rows"], "success"),
            )
        conn.commit()
        log.info("  committed batch %s to Postgres", batch_path.name)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", help="Explicit batch filename (e.g. batch_001.csv)")
    parser.add_argument("--all", action="store_true", help="Process all remaining unprocessed batches")
    parser.add_argument(
        "--skip-raw", action="store_true",
        help="Don't write the raw_customers pass-through table (only customers_cleaned). "
             "Useful for storage-capped hosts like Neon's free tier - raw_customers is an "
             "audit copy not read by training/RAG/dashboard code.",
    )
    args = parser.parse_args()

    conn = get_pg_conn()
    try:
        if args.batch:
            targets = [RAW_DIR / args.batch]
        elif args.all:
            done = already_loaded_batches(conn)
            targets = [p for p in sorted(RAW_DIR.glob("batch_*.csv")) if p.name not in done]
        else:
            nxt = next_unprocessed_batch(conn)
            targets = [nxt] if nxt else []
    finally:
        conn.close()

    if not targets:
        log.info("No new batches to process.")
        return

    for batch_path in targets:
        if not batch_path.exists():
            log.error("Batch file not found: %s", batch_path)
            sys.exit(1)
        load_batch(batch_path, skip_raw=args.skip_raw)


if __name__ == "__main__":
    main()
