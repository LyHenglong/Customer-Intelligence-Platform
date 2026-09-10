"""
Orchestrates one simulated batch arrival: ingest the next unprocessed
batch file (DuckDB cleaning/typing/feature engineering + Postgres load,
all inside src/ingest/batch_loader.py) -> dbt run -> dbt test -> PSI drift
check -> retrain the churn model on the now-larger customer_360, either
every Nth batch or whenever the drift check finds a shifted feature.

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

    @task
    def detect_feature_drift(batch_file: str) -> dict:
        """Scores the just-ingested batch against the baseline batch with
        PSI and persists the per-feature results to public.feature_drift.

        Runs after dbt_test rather than before, so a batch that fails data
        quality never contributes a drift verdict."""
        from src.monitoring.drift import detect_drift_for_batch

        summary = detect_drift_for_batch(batch_file)
        log.info(
            "Drift check for %s vs %s: max PSI %.4f, %d feature(s) beyond the significant band",
            summary["current_batch"],
            summary["reference_batch"],
            summary["max_psi"],
            len(summary["drifted_features"]),
        )
        return summary

    @task.branch
    def check_retrain_needed(drift_summary: dict) -> str:
        """Retrains on either of two independent triggers.

        1. Cadence: every RETRAIN_EVERY_N_BATCHES-th batch, so the model
           never goes stale just because nothing tripped an alarm.
        2. Evidence: any feature whose PSI crossed the significant band
           (>= 0.25), which means the incoming distribution no longer looks
           like the one the model was fitted on.

        Cadence alone was the original behaviour; the drift trigger is what
        makes a retrain responsive to the data rather than only to a
        counter."""
        from src.ingest.batch_loader import get_pg_conn

        conn = get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM ingestion_log")
                n_batches = cur.fetchone()[0]
        finally:
            conn.close()

        cadence_due = n_batches > 0 and n_batches % RETRAIN_EVERY_N_BATCHES == 0
        drift_detected = bool(drift_summary.get("drift_detected"))

        log.info(
            "Retrain check - batches: %d (every %d -> due=%s); drift: max PSI %.4f -> detected=%s %s",
            n_batches,
            RETRAIN_EVERY_N_BATCHES,
            cadence_due,
            drift_summary.get("max_psi", 0.0),
            drift_detected,
            drift_summary.get("drifted_features", []),
        )

        if cadence_due or drift_detected:
            return "retrain_churn_model"
        return "skip_retrain"

    @task
    def retrain_churn_model() -> dict:
        from src.model.train_churn import train_and_save

        metadata = train_and_save()
        log.info("Retrained churn model: version=%s f1_churn=%.4f", metadata["version"], metadata["f1_churn"])
        return metadata

    @task
    def retrain_recommender() -> dict:
        """Rebuilds the recommender's k-NN reference set on the same trigger
        as the churn model.

        Retraining these together is deliberate: the recommender's reference
        set is a snapshot of customer_360, so leaving it out meant it aged
        indefinitely while the churn model refreshed - the recommender
        artifact was 9 hours older than the churn artifact and built from
        2 batches' worth of customers rather than 13."""
        from src.model.train_recommender import train_and_save

        metadata = train_and_save()
        log.info(
            "Retrained recommender: version=%s reference_profiles=%d",
            metadata["version"], metadata["n_customers"],
        )
        return metadata

    skip_retrain = EmptyOperator(task_id="skip_retrain")

    pipeline_done = EmptyOperator(task_id="pipeline_done", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    # ingest returns the batch filename, which the drift check needs, so the
    # two are wired by data dependency rather than only by ordering.
    ingested_batch = ingest_next_batch()
    drift_summary = detect_feature_drift(ingested_batch)
    branch = check_retrain_needed(drift_summary)

    ingested_batch >> dbt_run >> dbt_test >> drift_summary >> branch
    # Sequential, not parallel: both retrains run in-process under
    # LocalExecutor on a 3.8GB VM, and running them concurrently is what
    # the memory ceiling cannot absorb (see docker-compose mem_limit notes).
    branch >> retrain_churn_model() >> retrain_recommender() >> pipeline_done
    branch >> skip_retrain >> pipeline_done


churn_pipeline()
