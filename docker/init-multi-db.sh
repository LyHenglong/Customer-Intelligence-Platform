#!/bin/bash
# Runs once, on first Postgres container init, to create the second
# database used exclusively for Airflow's own metadata (separate from
# the warehouse database that dbt/ingestion/ML code read and write).
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE "${AIRFLOW_DB}"'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${AIRFLOW_DB}')\gexec
EOSQL
