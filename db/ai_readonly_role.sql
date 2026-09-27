-- Least-privilege role for the AI assistant's SQL tool (src/ai/tools/sql_tool.py).
--
-- The tool validates every query (src/ai/guardrails/sql_safety.py) and runs it
-- in a read-only transaction, but both of those are checks this codebase
-- performs on itself. Connecting as this role makes Postgres enforce the
-- table allowlist too: anything outside the GRANTs below is a permission error,
-- whatever the query text looks like.
--
-- Run once as the warehouse owner, after dbt has built marts.customer_360:
--   psql "$DATABASE_URL" -v ai_sql_password="'<a strong password>'" -f db/ai_readonly_role.sql
-- then set AI_SQL_POSTGRES_USER=ai_sql_reader and AI_SQL_POSTGRES_PASSWORD for the API.
--
-- Keep the GRANTs in sync with SQL_ALLOWED_TABLES in src/ai/config.py.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ai_sql_reader') THEN
        CREATE ROLE ai_sql_reader LOGIN;
    END IF;
END
$$;

ALTER ROLE ai_sql_reader WITH PASSWORD :ai_sql_password;
ALTER ROLE ai_sql_reader SET default_transaction_read_only = on;
ALTER ROLE ai_sql_reader SET statement_timeout = '5s';

REVOKE ALL ON SCHEMA public FROM ai_sql_reader;
GRANT USAGE ON SCHEMA marts, public TO ai_sql_reader;

GRANT SELECT ON marts.customer_360 TO ai_sql_reader;
GRANT SELECT ON public.feature_drift TO ai_sql_reader;
GRANT SELECT ON public.ingestion_log TO ai_sql_reader;
GRANT SELECT ON public.retrain_summaries TO ai_sql_reader;
