-- Runs once on first Postgres container init (mounted into
-- /docker-entrypoint-initdb.d/). Creates the schemas dbt writes into, the
-- raw customer table ingestion writes into (schema confirmed by inspecting
-- the actual Kaggle CSV - see README for the source), and a tracking table
-- so the ingestion script knows which simulated batch files were loaded.

CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS intermediate;
CREATE SCHEMA IF NOT EXISTS marts;

-- Needed by public.rag_chunks below. Ships with the pgvector/pgvector
-- Docker image used by docker-compose.yml's postgres service, and is
-- available as an enable-able extension on Neon.
CREATE EXTENSION IF NOT EXISTS vector;

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

-- Per-feature PSI drift scores, one row per (reference batch, current
-- batch, feature). Written by src/monitoring/drift.py from the DAG's
-- detect_feature_drift task; read by the dashboard's Pipeline Status view.
-- Also created on demand by ensure_drift_table() so the module works
-- against a warehouse that predates this file.
CREATE TABLE IF NOT EXISTS public.feature_drift (
    reference_batch TEXT NOT NULL,
    current_batch   TEXT NOT NULL,
    feature         TEXT NOT NULL,
    feature_type    TEXT NOT NULL,
    psi             DOUBLE PRECISION NOT NULL,
    severity        TEXT NOT NULL,
    computed_at     TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (reference_batch, current_batch, feature)
);

-- AI agent layer: cached LLM-generated content (explanation_agent /
-- outreach_agent), keyed by customer + agent type + the churn model
-- version whose SHAP output grounded it. Written/read by
-- src/agents/cache.py. Also created on demand by ensure_tables() so the
-- module works against a warehouse that predates this file.
CREATE TABLE IF NOT EXISTS public.llm_explanations (
    customer_id        TEXT NOT NULL,
    agent_type          TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    content             TEXT NOT NULL,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    created_at          TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (customer_id, agent_type, model_version)
);

-- One row per completed retrain, written by the DAG's summarize_retrain
-- task (retrain_summary_agent). The plain-English narrative counterpart
-- to models_store/*.json's structured metrics.
CREATE TABLE IF NOT EXISTS public.retrain_summaries (
    churn_model_version  TEXT PRIMARY KEY,
    previous_version     TEXT,
    summary_text         TEXT NOT NULL,
    prompt_tokens        INTEGER,
    completion_tokens    INTEGER,
    created_at           TIMESTAMP NOT NULL DEFAULT now()
);

-- RAG knowledge corpus chunks (src/ai/rag/), one row per chunk produced by
-- src/ai/rag/ingest.py from the documents under knowledge/. embedding is
-- 384-dim to match the default local embedding model
-- (sentence-transformers/all-MiniLM-L6-v2 - see src/ai/rag/embeddings.py).
-- Also created on demand by vector_store.ensure_table() so the module
-- works against a warehouse that predates this file.
CREATE TABLE IF NOT EXISTS public.rag_chunks (
    chunk_id      TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL,
    title         TEXT NOT NULL,
    source        TEXT NOT NULL,
    section       TEXT,
    page          INTEGER,
    text          TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding     VECTOR(384),
    created_at    TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id ON public.rag_chunks (document_id);

-- One row per AI assistant request (src/ai/graph.py's run_query, written
-- by src/ai/observability/tracing.py). Deliberately excludes the
-- generated answer text and full evidence values - routing/timing/
-- token/cost metadata only, never the customer data a response's
-- evidence carried. Also created on demand by tracing.ensure_table() so
-- the module works against a warehouse that predates this file.
CREATE TABLE IF NOT EXISTS public.ai_traces (
    trace_id             TEXT PRIMARY KEY,
    request_timestamp    TIMESTAMP NOT NULL DEFAULT now(),
    user_query           TEXT NOT NULL,
    route                TEXT,
    tools_used           JSONB NOT NULL DEFAULT '[]'::jsonb,
    tool_latency_ms      JSONB NOT NULL DEFAULT '{}'::jsonb,
    sql_query_hash       TEXT,
    retrieval_latency_ms DOUBLE PRECISION,
    retrieved_documents  JSONB NOT NULL DEFAULT '[]'::jsonb,
    reranker_latency_ms  DOUBLE PRECISION,
    llm_model            TEXT,
    input_tokens         INTEGER,
    output_tokens        INTEGER,
    estimated_cost_usd   DOUBLE PRECISION,
    total_latency_ms     DOUBLE PRECISION,
    validation_result    TEXT,
    fallback_status      BOOLEAN NOT NULL DEFAULT false,
    error                TEXT
);
