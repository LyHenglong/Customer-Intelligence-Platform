#!/usr/bin/env bash
# Migrates only what the deployed dashboard actually reads - marts.customer_360
# plus the small operational tables (ingestion_log, feature_drift,
# retrain_summaries, llm_explanations) - from the local Docker Postgres into
# a hosted Postgres (Neon or otherwise). Deliberately does NOT replicate
# raw_customers/customers_cleaned/staging/intermediate: those are pipeline
# internals the dashboard never queries, and skipping them cuts the migrated
# size roughly in half.
#
# Usage:
#   ./scripts/migrate_to_neon.sh "postgresql://user:pass@host/dbname?sslmode=require"
#
# Requires the local docker-compose postgres service to be running
# (docker compose up -d postgres). Does NOT require psql on the host - both
# the dump (from local) and the restore (to the target) run through the
# postgres container's own psql/pg_dump, piped to/from the target's real
# hostname. This matters on Windows/Git Bash hosts in particular, which
# don't ship a psql binary at all (confirmed absent when this was first run
# for real) - the container has one, and Docker containers have outbound
# internet access by default, so it can reach an external target like Neon
# just as well as it reaches the local "postgres" service.
set -euo pipefail

TARGET_DSN="${1:?Usage: $0 <target-postgres-connection-string>}"
DUMP_FILE="$(dirname "$0")/../neon_migration.sql"

echo "==> Dumping marts.customer_360 + operational tables from local Postgres..."
docker compose exec -T postgres pg_dump \
  -U "${POSTGRES_USER:?set POSTGRES_USER (or run via .env)}" \
  -d "${POSTGRES_DB:?set POSTGRES_DB}" \
  --no-owner --no-privileges --clean --if-exists \
  -t marts.customer_360 \
  -t public.ingestion_log \
  -t public.feature_drift \
  -t public.retrain_summaries \
  -t public.llm_explanations \
  > "$DUMP_FILE"

echo "==> Dump written: $DUMP_FILE ($(du -h "$DUMP_FILE" | cut -f1))"

echo "==> Ensuring the marts schema exists on the target (pg_dump -t doesn't emit CREATE SCHEMA)..."
docker compose exec -T postgres psql "$TARGET_DSN" -c "CREATE SCHEMA IF NOT EXISTS marts;"

echo "==> Restoring into target (a 213MB dump over a real network - this can take several minutes, see the README's note on Neon latency)..."
docker compose exec -T postgres psql "$TARGET_DSN" -v ON_ERROR_STOP=1 -f /dev/stdin < "$DUMP_FILE"

echo "==> Verifying row counts on the target..."
docker compose exec -T postgres psql "$TARGET_DSN" -t -A -c "
SELECT 'customer_360', COUNT(*) FROM marts.customer_360
UNION ALL SELECT 'ingestion_log', COUNT(*) FROM public.ingestion_log
UNION ALL SELECT 'feature_drift', COUNT(*) FROM public.feature_drift;
"

echo "==> Done. Delete $DUMP_FILE once you've confirmed the target looks right (it's gitignored, but it's a full data copy)."
