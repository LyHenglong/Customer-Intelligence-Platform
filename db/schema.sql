-- Runs once on first Postgres container init (mounted into
-- /docker-entrypoint-initdb.d/). Creates the schemas dbt writes into, the
-- raw customer table ingestion writes into (schema confirmed by inspecting
-- the actual Kaggle CSV - see README for the source), and a tracking table
-- so the ingestion script knows which simulated batch files were loaded.

CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS intermediate;
CREATE SCHEMA IF NOT EXISTS marts;

CREATE TABLE IF NOT EXISTS public.ingestion_log (
    batch_file      TEXT PRIMARY KEY,
    rows_loaded     INTEGER NOT NULL,
    loaded_at       TIMESTAMP NOT NULL DEFAULT now(),
    status          TEXT NOT NULL DEFAULT 'success'
);

-- Raw layer: one row per customer per batch load, columns mirror the
-- source CSV as closely as possible (light typing only - real cleaning
-- happens in dbt staging models). source_batch/ingested_at are metadata
-- added by the ingestion script, not present in the source data.
CREATE TABLE IF NOT EXISTS public.raw_customers (
    customer_id                    TEXT PRIMARY KEY,
    signup_date                    TIMESTAMP,
    age                            INTEGER,
    gender                         TEXT,
    annual_income                  NUMERIC(14, 2),
    education                      TEXT,
    marital_status                 TEXT,
    dependents                     INTEGER,
    tenure                         INTEGER,
    contract                       TEXT,
    payment_method                 TEXT,
    paperless_billing              TEXT,
    senior_citizen                 INTEGER,
    monthlycharges                 NUMERIC(10, 2),
    totalcharges                   NUMERIC(14, 2),
    num_services                   INTEGER,
    has_phone_service              INTEGER,
    has_internet_service           INTEGER,
    has_online_security            INTEGER,
    has_online_backup              INTEGER,
    has_device_protection          INTEGER,
    has_tech_support               INTEGER,
    has_streaming_tv               INTEGER,
    has_streaming_movies           INTEGER,
    customer_satisfaction          NUMERIC(5, 2),
    num_complaints                 NUMERIC(6, 2),
    num_service_calls              INTEGER,
    late_payments                  INTEGER,
    avg_monthly_gb                 NUMERIC(10, 2),
    days_since_last_interaction    INTEGER,
    credit_score                   NUMERIC(6, 2),
    churn                          INTEGER,
    source_batch                   TEXT NOT NULL,
    ingested_at                    TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_raw_customers_source_batch ON public.raw_customers (source_batch);

-- Lightly-cleaned layer: type-validated (invalid values coerced to NULL via
-- TRY_CAST at ingestion, not silently kept as bad strings), categorical
-- flags cast to boolean, plus a few features cheap to compute per-batch at
-- ingestion time. Heavier joins/aggregations belong in dbt's intermediate
-- layer, not here - this table is dbt staging's direct input.
CREATE TABLE IF NOT EXISTS public.customers_cleaned (
    customer_id                    TEXT PRIMARY KEY,
    signup_date                    TIMESTAMP,
    age                            INTEGER,
    gender                         TEXT,
    annual_income                  NUMERIC(14, 2),
    education                      TEXT,
    marital_status                 TEXT,
    dependents                     INTEGER,
    tenure                         INTEGER,
    contract                       TEXT,
    payment_method                 TEXT,
    paperless_billing              BOOLEAN,
    senior_citizen                 BOOLEAN,
    monthlycharges                 NUMERIC(10, 2),
    totalcharges                   NUMERIC(14, 2),
    num_services                   INTEGER,
    has_phone_service              BOOLEAN,
    has_internet_service           BOOLEAN,
    has_online_security            BOOLEAN,
    has_online_backup              BOOLEAN,
    has_device_protection          BOOLEAN,
    has_tech_support               BOOLEAN,
    has_streaming_tv               BOOLEAN,
    has_streaming_movies           BOOLEAN,
    customer_satisfaction          NUMERIC(5, 2),
    num_complaints                 NUMERIC(6, 2),
    num_service_calls              INTEGER,
    late_payments                  INTEGER,
    avg_monthly_gb                 NUMERIC(10, 2),
    days_since_last_interaction    INTEGER,
    credit_score                   NUMERIC(6, 2),
    churn                          INTEGER,
    tenure_years                   NUMERIC(5, 2),
    total_active_services          INTEGER,
    avg_gb_per_service             NUMERIC(10, 2),
    source_batch                   TEXT NOT NULL,
    ingested_at                    TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_customers_cleaned_source_batch ON public.customers_cleaned (source_batch);
