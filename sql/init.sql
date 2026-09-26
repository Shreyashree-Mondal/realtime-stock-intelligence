-- ============================================================
-- Schema for the serving layer. Loaded automatically by Postgres
-- the first time the container starts.
-- ============================================================

-- ---------- streaming gold tables (written by Spark) ----------
CREATE TABLE IF NOT EXISTS ohlc_1m (
    symbol TEXT NOT NULL, window_start TIMESTAMP NOT NULL, window_end TIMESTAMP NOT NULL,
    open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION, close DOUBLE PRECISION,
    volume DOUBLE PRECISION, vwap DOUBLE PRECISION, trade_count BIGINT,
    PRIMARY KEY (symbol, window_start)
);

CREATE TABLE IF NOT EXISTS rolling_5m (
    symbol TEXT NOT NULL, window_start TIMESTAMP NOT NULL, window_end TIMESTAMP NOT NULL,
    avg_price DOUBLE PRECISION, volatility DOUBLE PRECISION, volume DOUBLE PRECISION, trade_count BIGINT,
    PRIMARY KEY (symbol, window_start)
);

-- ---------- data quality ----------
CREATE TABLE IF NOT EXISTS dq_rejects (          -- bad records quarantined by Spark
    id BIGSERIAL PRIMARY KEY, rejected_at TIMESTAMP DEFAULT (now() AT TIME ZONE 'UTC'),
    reason TEXT, raw_value TEXT
);

CREATE TABLE IF NOT EXISTS dq_report (           -- daily checks from batch/dq_report.py
    run_at TIMESTAMP DEFAULT (now() AT TIME ZONE 'UTC'), check_name TEXT, status TEXT, detail TEXT
);

-- ---------- machine learning ----------
CREATE TABLE IF NOT EXISTS predictions (         -- online forecasts + A/B assignment
    symbol TEXT NOT NULL, window_start TIMESTAMP NOT NULL,
    variant CHAR(1) NOT NULL,                    -- 'A' or 'B' (which model was served)
    served_pred DOUBLE PRECISION,                -- prediction from the assigned model
    shadow_pred DOUBLE PRECISION,                -- prediction from the other model (logged, not served)
    baseline_pred DOUBLE PRECISION,              -- naive baseline: current volatility
    actual DOUBLE PRECISION,                     -- filled in ~5 minutes later
    created_at TIMESTAMP DEFAULT (now() AT TIME ZONE 'UTC'),
    PRIMARY KEY (symbol, window_start)
);

CREATE TABLE IF NOT EXISTS anomalies (
    symbol TEXT NOT NULL, window_start TIMESTAMP NOT NULL,
    anomaly_score DOUBLE PRECISION, ret_pct DOUBLE PRECISION, volume_ratio DOUBLE PRECISION,
    PRIMARY KEY (symbol, window_start)
);

CREATE TABLE IF NOT EXISTS ab_results (
    run_at TIMESTAMP DEFAULT (now() AT TIME ZONE 'UTC'),
    n_a INT, n_b INT, mae_a DOUBLE PRECISION, mae_b DOUBLE PRECISION, mae_baseline DOUBLE PRECISION,
    diff_ci_low DOUBLE PRECISION, diff_ci_high DOUBLE PRECISION, p_value DOUBLE PRECISION, decision TEXT
);

-- ---------- NLP ----------
CREATE TABLE IF NOT EXISTS news_sentiment (
    news_id TEXT PRIMARY KEY, symbol TEXT, published_at TIMESTAMP, headline TEXT, source TEXT,
    label TEXT, score DOUBLE PRECISION          -- score = P(positive) - P(negative), range -1..1
);

-- ---------- batch layer (daily, written by PySpark job) ----------
CREATE TABLE IF NOT EXISTS daily_metrics (
    symbol TEXT, market TEXT, date DATE, close DOUBLE PRECISION, daily_return DOUBLE PRECISION,
    vol_20d_ann DOUBLE PRECISION, drawdown DOUBLE PRECISION,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS risk_summary (
    symbol TEXT PRIMARY KEY, market TEXT, as_of DATE, ann_return DOUBLE PRECISION,
    ann_volatility DOUBLE PRECISION, sharpe DOUBLE PRECISION, var_95_1d DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION, beta DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_ohlc_time ON ohlc_1m (window_start);
CREATE INDEX IF NOT EXISTS idx_pred_actual ON predictions (actual) WHERE actual IS NULL;

-- ---------- RAG knowledge base (pgvector) ----------
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id    TEXT PRIMARY KEY,
    source_type TEXT,                 -- 'doc' | 'news' | 'report'
    source_id   TEXT,
    symbol      TEXT,                 -- NULL when not about one stock
    title       TEXT,
    content     TEXT,
    created_at  TIMESTAMP DEFAULT (now() AT TIME ZONE 'UTC'),
    embedding   vector(384)           -- BAAI/bge-small-en-v1.5 embeddings
);
CREATE INDEX IF NOT EXISTS idx_rag_hnsw ON rag_chunks USING hnsw (embedding vector_cosine_ops);

-- ---------- read-only role for the GenAI assistant ----------
DO $$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'analyst_ro') THEN
        CREATE ROLE analyst_ro LOGIN PASSWORD 'analyst_ro';
    END IF;
END $$;
GRANT CONNECT ON DATABASE stocks TO analyst_ro;
GRANT USAGE ON SCHEMA public TO analyst_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst_ro;
ALTER ROLE analyst_ro SET statement_timeout = '5s';
