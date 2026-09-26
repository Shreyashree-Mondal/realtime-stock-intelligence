"""
AIRFLOW DAG — the daily batch layer, every weekday after US market close.

  ingest_daily_prices -> pyspark_risk_job -> retrain_models -> ab_test_report
                                          +-> data_quality_report

Airflow gives us scheduling, retries, dependency order, logs and alerts for these jobs.
"""
import os
from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator

from airflow import DAG

PROJECT = os.getenv("PROJECT_DIR", "/opt/project")
run = lambda script: f"cd {PROJECT} && python {script}"  # noqa: E731

with DAG(
    dag_id="stock_daily_batch",
    start_date=datetime(2026, 1, 1),
    schedule="30 21 * * 1-5",          # 21:30 UTC, Mon-Fri
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["stocks", "portfolio"],
) as dag:
    ingest = BashOperator(task_id="ingest_daily_prices", bash_command=run("batch/daily_ingest.py"))
    risk = BashOperator(task_id="pyspark_risk_job", bash_command=run("batch/daily_risk_job.py"))
    retrain = BashOperator(task_id="retrain_models", bash_command=run("ml/train.py"))
    ab = BashOperator(task_id="ab_test_report", bash_command=run("ml/ab_test.py"))
    dq = BashOperator(task_id="data_quality_report", bash_command=run("batch/dq_report.py"))

    ingest >> risk >> [retrain, dq]
    retrain >> ab
