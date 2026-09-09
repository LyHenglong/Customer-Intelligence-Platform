"""
Orchestrates one simulated batch arrival: ingest the next unprocessed
batch file (DuckDB cleaning/typing/feature engineering + Postgres load,
all inside src/ingest/batch_loader.py) -> dbt run -> dbt test -> (every
Nth batch) retrain the churn model on the now-larger customer_360.

Manually triggered (schedule=None), not time-based: each DAG run represents
one simulated weekly batch "arriving", which happens whenever a user or a
script triggers it - a wall-clock schedule would misrepresent the
simulation. See README's "How to run" section for how to trigger runs.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

DBT_DIR = "/opt/airflow/dbt"
RETRAIN_EVERY_N_BATCHES = int(os.environ.get("RETRAIN_EVERY_N_BATCHES", "3"))

default_args = {
    "owner": "telecom_churn_platform",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


@dag(
    dag_id="churn_pipeline",
    description="Ingest next simulated batch -> dbt run -> dbt test",
    default_args=default_args,
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["telecom-churn"],
)
def churn_pipeline():

    @task
    def ingest_next_batch() -> str:
        """Picks up the next unprocessed data/raw/batch_*.csv file (if any),
        processes it with DuckDB, and loads raw + cleaned tables into
        Postgres. Skips the DAG run cleanly if every batch has already
        been ingested."""
        from src.ingest.batch_loader import get_pg_conn, load_batch, next_unprocessed_batch

        conn = get_pg_conn()
        try:
            batch_path = next_unprocessed_batch(conn)
        finally:
            conn.close()

        if batch_path is None:
            raise AirflowSkipException("No new batch files to ingest.")

        load_batch(batch_path)
        return batch_path.name

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=f"cd {DBT_DIR} && dbt run",
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=f"cd {DBT_DIR} && dbt test",
    )

    @task.branch
    def check_retrain_needed() -> str:
        """Counts batches ingested so far (ingestion_log) and branches to
        retraining only every RETRAIN_EVERY_N_BATCHES-th batch - a simple
        batch-count trigger, not drift-based (see README Limitations)."""
        from src.ingest.batch_loader import get_pg_conn

        conn = get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM ingestion_log")
                n_batches = cur.fetchone()[0]
        finally:
            conn.close()

        log.info("Batches ingested so far: %d (retrain every %d)", n_batches, RETRAIN_EVERY_N_BATCHES)
        if n_batches > 0 and n_batches % RETRAIN_EVERY_N_BATCHES == 0:
            return "retrain_churn_model"
        return "skip_retrain"

    @task
    def retrain_churn_model() -> dict:
        from src.model.train_churn import train_and_save

        metadata = train_and_save()
        log.info("Retrained churn model: version=%s f1_churn=%.4f", metadata["version"], metadata["f1_churn"])
        return metadata

    skip_retrain = EmptyOperator(task_id="skip_retrain")

    pipeline_done = EmptyOperator(task_id="pipeline_done", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    branch = check_retrain_needed()
    ingest_next_batch() >> dbt_run >> dbt_test >> branch
    branch >> retrain_churn_model() >> pipeline_done
    branch >> skip_retrain >> pipeline_done


churn_pipeline()
