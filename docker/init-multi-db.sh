#!/bin/bash
# Runs once, on first Postgres container init, to create the extra
# databases used exclusively for a single component's own metadata,
# separate from the warehouse database that dbt/ingestion/ML code read
# and write: Airflow's own metadata, and MLflow's tracking/registry
# backend store (mlflow server's own schema migrations should stay
# isolated from both warehouse and airflow_meta).
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE "${AIRFLOW_DB}"'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${AIRFLOW_DB}')\gexec
    SELECT 'CREATE DATABASE "${MLFLOW_DB}"'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${MLFLOW_DB}')\gexec
EOSQL
