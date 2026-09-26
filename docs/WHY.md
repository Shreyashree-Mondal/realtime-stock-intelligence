# Why each part exists (beginner guide + interview answers)

Each section follows the same pattern: **what it is → why we used it → how it helps a
business → a likely interview question with a short answer.**

---

## 1. Kafka (message broker)
**What:** A "post office" for data. Producers drop messages in a *topic*; consumers read them.
**Why:** Trades arrive thousands per second. Kafka buffers them, so if Spark restarts, no data is lost.
Several consumers (Spark, sentiment model) can read the same stream independently.
**Helps the business:** decouples systems; one broken service doesn't take down the others.
**Q: Why key messages by stock symbol?**
Kafka keeps order only inside a partition. The same key always goes to the same partition,
so each stock's trades stay in order.

## 2. PySpark Structured Streaming
**What:** Spark processing data continuously in small "micro-batches".
**Why:** Turns raw trades into 1-minute candles and 5-minute volatility as data arrives,
and scales to much bigger volumes by adding machines.
**Q: Tumbling vs sliding window?**
Tumbling windows don't overlap (10:00–10:01, 10:01–10:02) → candles.
Sliding windows overlap (10:00–10:05, 10:01–10:06) → smooth rolling statistics.
**Q: What is a watermark?**
How long to wait for late data. We accept trades up to 2 minutes late, then close the window
and free memory.
**Q: Event time vs processing time?**
Event time = when the trade happened; processing time = when we received it. We group by
event time so network delays don't put trades in the wrong minute.

## 3. Data quality
**What:** Every record is checked (missing symbol, negative price, future timestamp…).
Bad records go to a `dq_rejects` table instead of disappearing. A daily report checks
freshness, nulls, OHLC logic and duplicates, and fails the Airflow run if something is wrong.
**Why:** Wrong data leads to wrong decisions; silent data loss is worse than a visible error.
**Q: Why quarantine instead of drop?** So you can count, investigate and replay bad data.

## 4. SQL window functions
**What:** Calculations across related rows without collapsing them (`OVER (PARTITION BY … ORDER BY …)`).
**Used for:** moving averages (`AVG OVER ROWS BETWEEN`), returns (`LAG`), top movers (`RANK`),
z-score alerts (`STDDEV OVER`), crossovers, running max for drawdown.
**Q: Difference between GROUP BY and a window function?** GROUP BY returns one row per group;
a window function keeps every row and adds the calculated value to it.

## 5. Machine learning: volatility forecasting
**What:** Predict how much a stock will move in the next 5 minutes.
**Why volatility, not price?** Price direction is close to random. Volatility "clusters":
calm periods follow calm ones, so it is predictable, and it is used for risk limits,
position sizing and option pricing.
**Models:** a naive **baseline** (next 5 min = last 5 min), **Ridge** (simple linear, the
champion) and **XGBoost** (trees, the challenger).
**Q: Why a baseline?** A model is only useful if it beats the simplest possible guess.
**Q: Why a time-based train/test split?** A random split would train on the future and test
on the past, giving unrealistically good scores (data leakage).
**Q: How do you avoid training/serving skew?** Training and live scoring use the same
`build_features()` function, and a unit test checks that features never use future data.

## 6. MLflow
**What:** Records every training run: parameters, metrics, model files.
**Why:** You can compare runs and reproduce any model later.

## 7. Anomaly detection (Isolation Forest)
**What:** An unsupervised model that flags bars that look unusual (big move + volume spike).
**Why:** No labels exist for "abnormal"; Isolation Forest isolates rare points quickly.
Same idea as fraud detection.
**Q: Why unsupervised?** We don't have labelled examples of anomalies.

## 8. A/B testing
**What:** Each live bar is randomly served by model A or model B (50/50). Once the actual
volatility is known, we compare their errors.
**How we decide:** Welch's t-test (p < 0.05), a bootstrap 95% confidence interval for the
difference in error, and a minimum sample size of 100 per group before deciding.
**Why:** Offline test scores can be misleading; A/B testing proves which model is better on
live data before switching.
**Q: Why a minimum sample size?** Checking results too early ("peeking") finds false winners.
**Q: Why keep A if there's no significant difference?** The simpler model is cheaper and
easier to explain.
**Q: Why hash the ID to assign variants?** It's random across bars but reproducible, the same
way websites hash a user ID so a user always sees the same version.

## 9. NLP: FinBERT news sentiment
**What:** A BERT model trained on financial text scores each headline from −1 to +1.
**Why FinBERT, not general sentiment?** Finance wording is tricky ("losses narrowed" is good news).
**Helps the business:** connects news to price moves; analysts see *why* a stock moved.

## 10. GenAI assistant (text-to-SQL)
**What:** Ask "Which stock was most volatile in the last hour?" → the LLM writes SQL → it runs
→ the LLM summarises the rows.
**Safety (the important part):** only single SELECT statements pass a guard; queries run as a
**read-only** database user with a 5-second timeout; the answer uses only returned data.
**Q: Why not trust the prompt alone?** LLMs can ignore instructions; the database permissions
are what actually enforce safety.

## 11. Batch layer (PySpark + Airflow)
**What:** Once a day: download 2 years of US **and Indian** (NSE) prices → PySpark computes
risk metrics → retrain models → A/B report → data-quality report.
**Risk metrics:** Sharpe ratio (return per unit of risk), 1-day 95% VaR (a loss exceeded only
5% of days), max drawdown (worst fall from a peak), beta (sensitivity to the market index).
**Why Airflow?** Scheduling, run order, retries, logs and alerts, instead of cron scripts.
**Q: Why both streaming and batch?** Streaming gives fast, approximate answers; batch gives
complete, corrected history. Most real platforms have both.

## 12. FastAPI + dashboard
Serves every result as a REST API (auto docs at `/docs`); the dashboard calls the API.

## 13. Docker, CI/CD
**Docker:** every service runs the same on any machine with one command.
**CI (GitHub Actions):** each push runs linting + 24 unit tests (features, A/B logic, SQL guard, RAG helpers,
PySpark transforms) and builds the Docker image.
**CD:** when CI passes on `main`, the image is published to GitHub Container Registry, versioned
by commit.

## 14. Databricks, S3, Snowflake, Power BI
**Databricks:** the same pipeline as Bronze (raw) → Silver (clean) → Gold (aggregated) Delta tables.
**S3 + Snowflake:** the data lake copied to cloud storage and loaded into a warehouse.
**Power BI:** business-friendly dashboards on read-only SQL views.

## 15. RAG AI assistant (retrieval-augmented generation)
**What:** An assistant that answers from *your* platform's knowledge, not just the LLM's memory.
It indexes three kinds of text: project docs, news headlines with sentiment, and auto-written
reports (risk, A/B test, data quality, model results). Each chunk becomes an **embedding**
(a list of 384 numbers that captures meaning), stored in Postgres with **pgvector**.
**How a question is answered:**
1. A small **agent** (router) decides: live numbers → SQL tool; explanations/news → RAG tool; or both.
2. RAG embeds the question and fetches the 5 most similar chunks (cosine similarity).
3. The LLM answers using only that context and cites sources like [1], [2].
**Why RAG instead of fine-tuning?** Our data changes every few minutes. RAG just re-indexes;
fine-tuning would mean retraining constantly, and it can't cite sources.
**Why pgvector instead of a separate vector database?** One database to run; enough for thousands
of chunks. At much larger scale you'd consider a dedicated vector DB.
**Evaluation:** `genai/eval_rag.py` asks 10 questions whose correct section is known and reports
hit rate@3 and MRR, so changes to chunking or the embedding model are measured, not guessed.
**Safety:** news text is untrusted, so the prompt says to treat context as data (prompt-injection
defence); SQL still runs read-only; no investment advice.
**Q: How do you reduce hallucination?** Answer only from retrieved context, require citations,
and say "not enough information" when retrieval finds nothing relevant.
**Q: How did you choose chunk size?** Split by headings so each chunk is one topic, capped at
~1,200 characters; then checked hit rate with the eval set.

---
### Honest limitations (say these before an interviewer does)
- Free data is delayed or limited; Indian stocks are daily only (no free real-time NSE feed).
- The simulator is used for demos; results on real data will differ.
- Volatility forecasts are useful for risk, not a trading signal.
- One machine; in production Kafka and Spark would run as clusters.
- The RAG eval set is small (10 questions); a real system would use a larger labelled set.
