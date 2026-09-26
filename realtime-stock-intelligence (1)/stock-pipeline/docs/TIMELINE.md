# Timeline (about 2–3 hours per day)

## Part 1: Build and deploy (~3 weeks)
| Week | Goal | Done when |
|---|---|---|
| 1 | Core running: Docker, Kafka, PySpark streaming, API, dashboard. Push to GitHub, CI green | dashboard shows live candles; Actions tab green |
| 2 | ML: let data collect, `make train`, scorer running, `make abtest`, MLflow screenshots; batch layer `make daily` | A/B panel and risk table filled |
| 3 | Extras: NLP sentiment, GenAI assistant (free LLM key), Airflow, Databricks, Power BI, README results + demo video, LinkedIn post | everything in the DEPLOY checklist ticked |

Full-time (6–8 h/day): about 1–1.5 weeks.

## Part 2: Learn it for interviews (~3–4 weeks)
| Week | Topic | How |
|---|---|---|
| 1 | Kafka + PySpark basics | read `producer.py`, `transforms.py`; change a window size and watch the result |
| 2 | Streaming concepts + SQL window functions | watermarks, output modes; rewrite 2 queries from `analytics.sql` yourself |
| 3 | ML + A/B testing | rerun `train.py` with a new feature; explain p-value and confidence interval out loud |
| 4 | Airflow, Docker, CI/CD, GenAI, Databricks | trigger the DAG manually; break a test and watch CI fail |

Tip: practise answering every question in `docs/WHY.md` in your own words, out loud.
