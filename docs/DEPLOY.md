# Deploying and showcasing

## 1. GitHub (same as your other projects)
1. Create a new public repo, e.g. `realtime-stock-intelligence`.
2. Upload all files (keep the folder structure, including `.github/` and `.devcontainer/`).
   Do **not** upload `.env`.
3. Open the **Actions** tab: the CI workflow should turn green. After that, CD publishes the
   Docker image under **Packages** on your profile.
4. Add a CI badge to the README top:
   `![CI](https://github.com/<you>/<repo>/actions/workflows/ci.yml/badge.svg)`

## 2. Live demo with GitHub Codespaces (free)
Reviewers click **Code → Codespaces → Create codespace** and run `make up`; the dashboard
opens in the browser. Free accounts get a monthly allowance of Codespaces hours. Stop the
codespace when you're done so hours aren't wasted.
Add to your README: `[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/<you>/<repo>)`

## 3. Databricks Free Edition
1. Sign up for Databricks Free Edition.
2. Import `databricks/stock_pipeline_databricks.py` as a notebook.
3. Easiest source: upload `data/bronze` Parquet files to a Volume and change the bronze step to
   read files. For true streaming, use a managed Kafka trial (e.g. Confluent Cloud credits).
4. Schedule it as a Job (Workflows) and build an AI/BI dashboard on the gold tables.
5. Screenshot the Delta tables, Job runs and dashboard for your README.

## 4. Power BI
1. Run `sql/powerbi_views.sql` once:
   `docker compose exec -T postgres psql -U stocks -d stocks < sql/powerbi_views.sql`
2. Power BI Desktop → Get Data → PostgreSQL → server `localhost`, database `stocks`,
   user `analyst_ro` / `analyst_ro`.
3. Build three pages: Intraday (`v_intraday`), Model performance (`v_model_performance`),
   Risk US vs India (`v_risk`). Save screenshots to `docs/`.

## 5. AWS S3 + Snowflake (optional, free tiers/trials)
`python cloud/s3_upload.py` then run `cloud/snowflake_load.sql` in a Snowflake worksheet.

## 6. Showcase checklist
- [ ] Dashboard screenshot + 60-second demo video/GIF in README
- [ ] Results table in README filled with your real numbers
- [ ] MLflow, Airflow, Kafka UI, Spark UI, Power BI, Databricks screenshots in `docs/`
- [ ] LinkedIn post with the architecture diagram
