# Shortcuts. Run `make help` to list them.
.PHONY: help up down logs train abtest daily rageval test lint mlflow

help:      ## list commands
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'
up:        ## start the core pipeline (Kafka, Postgres, producer, Spark, scorer, API)
	docker compose up -d --build
down:      ## stop everything
	docker compose --profile nlp --profile airflow down
logs:      ## follow logs
	docker compose logs -f --tail=50
train:     ## train forecasting + anomaly models (needs ~45 min of data first)
	docker compose run --rm scorer python ml/train.py
abtest:    ## analyse the live A/B test between model A and model B
	docker compose run --rm scorer python ml/ab_test.py
daily:     ## run the batch layer once: daily prices -> PySpark risk job -> data-quality report
	docker compose run --rm scorer sh -c "python batch/daily_ingest.py && python batch/daily_risk_job.py && python batch/dq_report.py"
rageval:   ## re-index the knowledge base and measure RAG retrieval quality
	docker compose run --rm scorer sh -c "python genai/rag.py --once && python genai/eval_rag.py"
test:      ## run unit tests
	pytest -q
lint:      ## check code style
	ruff check .
mlflow:    ## open the MLflow experiment UI on http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
