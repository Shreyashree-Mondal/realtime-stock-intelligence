# Resume bullets

Use these **after** you have run the project and can explain every part (docs/WHY.md).
Replace every `[X]` with your own measured numbers; never guess them.

**Real-Time Stock Market Intelligence Platform** | Kafka, PySpark, Airflow, MLflow, XGBoost, FastAPI, Docker, Databricks

- Built an end-to-end streaming pipeline ingesting live trades via WebSocket API into Kafka and
  processing them with PySpark Structured Streaming (event-time tumbling and sliding windows,
  watermarking), sustaining [X] trades/sec with ~[X]s end-to-end latency.
- Engineered 12 time-series features and trained Ridge and XGBoost models to forecast
  5-minute volatility, reducing MAE by [X]% vs a naive baseline; tracked experiments in MLflow.
- Designed an online A/B test (hash-based 50/50 assignment, Welch's t-test, bootstrap CI,
  minimum-sample rule) to validate the challenger model on live data before promotion.
- Implemented anomaly detection (Isolation Forest) and FinBERT news-sentiment scoring on
  streaming data, surfacing unusual moves and news impact in a live FastAPI dashboard.
- Built a daily Airflow batch layer computing VaR, Sharpe, max drawdown and beta for US and
  Indian (NSE) equities with PySpark window functions; automated 6 data-quality checks with
  quarantine of invalid records.
- Built a RAG-based AI market assistant: an LLM agent routes each question to live SQL data
  and/or vector search (pgvector, bge-small embeddings) over docs, news and auto-generated
  reports, answering with citations; retrieval scored [X]% hit rate@3 on an evaluation set;
  guardrails include SQL validation, a read-only role and prompt-injection defences.
- Containerised 11 services with Docker and set up GitHub Actions CI/CD
  (24 unit tests, image publishing to GHCR); ported the pipeline to Databricks Delta
  (medallion architecture).

**Skills this adds:** Kafka, PySpark, Spark Structured Streaming, Airflow, Docker, CI/CD,
MLflow, A/B testing, time-series forecasting, Databricks/Delta Lake, AWS S3, Snowflake,
FinBERT/Transformers, RAG, embeddings, pgvector, LLM agents, text-to-SQL, FastAPI, financial risk metrics.

**Pick 3–4 bullets per application:** data engineering roles → streaming, Airflow, data quality,
Databricks. Data science roles → forecasting, A/B test, anomaly, NLP. Finance/risk roles →
risk metrics, data quality, A/B rigor. GenAI roles → RAG assistant + NLP.
