"""Scheduled Postgres backup for self-hosted deployments.

Only relevant when Postgres is self-hosted (the local docker-compose
`postgres` service). A hosted provider like Neon (the platform this
project's own README already recommends for the "public demo" deploy
path) provides automated backups/point-in-time recovery natively - this
DAG is not needed there and produces redundant, unwatched local files if
left running against a database that already has real ones. Guarded by
POSTGRES_HOST at the top: skips itself entirely against anything that
looks like a hosted provider rather than "postgres" (the docker-compose
service name).

Unlike churn_pipeline (schedule=None, manually triggered to simulate one
batch "arriving" per run), a backup genuinely needs to run on a real
wall-clock cadence, so this uses schedule="@daily".

Retention: keeps the last BACKUP_RETENTION_DAYS (default 14) of dumps in
./backups/ (mounted below) and prunes older ones - unbounded local
accumulation would eventually fill the host's disk. This is local-disk
retention only; copying dumps off-host (e.g. to object storage) is not
implemented here - see README's "Backups" section for why (kept as
documentation/config rather than a new cloud dependency nobody asked to
provision, same reasoning as this repo's existing "avoid unnecessary
infrastructure" choices elsewhere).
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.utils.dates import days_ago

from src.monitoring.alerting import send_slack_alert

log = logging.getLogger(__name__)

BACKUP_DIR = "/opt/airflow/backups"
BACKUP_RETENTION_DAYS = int(os.environ.get("BACKUP_RETENTION_DAYS", "14"))


def _alert_on_task_failure(context: dict) -> None:
    task_instance = context["task_instance"]
    send_slack_alert(
        f":x: *Warehouse backup failed* - `{task_instance.task_id}` "
        f"(run {context['run_id']}). Log: {task_instance.log_url}"
    )


default_args = {
    "owner": "telecom_churn_platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": _alert_on_task_failure,
}


@dag(
    dag_id="backup_warehouse",
    description="Daily pg_dump of the self-hosted warehouse database, with local retention pruning",
    default_args=default_args,
    schedule="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["telecom-churn", "backup"],
)
def backup_warehouse():

    @task
    def dump_and_prune() -> str:
        import datetime
        import subprocess
        from pathlib import Path

        host = os.environ.get("POSTGRES_HOST", "")
        if host != "postgres":
            raise AirflowSkipException(
                f"POSTGRES_HOST={host!r} doesn't look like the self-hosted docker-compose "
                "service - skipping (a hosted provider like Neon already backs itself up)."
            )

        backup_dir = Path(BACKUP_DIR)
        backup_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dump_path = backup_dir / f"warehouse_{stamp}.sql.gz"

        db = os.environ["POSTGRES_DB"]
        user = os.environ["POSTGRES_USER"]
        env = {**os.environ, "PGPASSWORD": os.environ["POSTGRES_PASSWORD"]}

        dump_cmd = f"pg_dump -h {host} -U {user} -d {db} | gzip > {dump_path}"
        result = subprocess.run(dump_cmd, shell=True, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"pg_dump failed: {result.stderr}")
        log.info("Wrote %s (%d bytes)", dump_path, dump_path.stat().st_size)

        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=BACKUP_RETENTION_DAYS)
        pruned = 0
        for existing in backup_dir.glob("warehouse_*.sql.gz"):
            mtime = datetime.datetime.fromtimestamp(existing.stat().st_mtime, tz=datetime.timezone.utc)
            if mtime < cutoff:
                existing.unlink()
                pruned += 1
        if pruned:
            log.info("Pruned %d backup(s) older than %d days", pruned, BACKUP_RETENTION_DAYS)

        return str(dump_path)

    dump_and_prune()


backup_warehouse()
