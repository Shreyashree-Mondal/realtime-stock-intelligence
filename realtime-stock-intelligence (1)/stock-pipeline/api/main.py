"""
Serving layer: REST API over the gold tables + the live dashboard.

  uvicorn api.main:app --reload --port 8000
  open http://localhost:8000          (dashboard)
  open http://localhost:8000/docs     (Swagger UI)
"""
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

load_dotenv()
PG_DSN = os.getenv("PG_DSN", "dbname=stocks user=stocks password=stocks host=localhost port=5432")
NOW = "(now() AT TIME ZONE 'UTC')"

app = FastAPI(title="Real-Time Stock Analytics API", version="2.0")


def query(sql: str, params: tuple = ()) -> list[dict]:
    try:
        conn = psycopg2.connect(PG_DSN)
    except psycopg2.OperationalError as e:
        raise HTTPException(503, f"Database unavailable: {e}")
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/health")
def health():
    row = query(f"SELECT MAX(window_end) AS last_bar, {NOW} AS now_utc FROM ohlc_1m")[0]
    return {"status": "ok", **row}


@app.get("/api/symbols")
def symbols():
    return [r["symbol"] for r in query("SELECT DISTINCT symbol FROM ohlc_1m ORDER BY symbol")]


@app.get("/api/candles/{symbol}")
def candles(symbol: str, minutes: int = Query(60, ge=5, le=1440)):
    """1-min OHLC bars with SMA-5, SMA-20 and 1-min return (window functions)."""
    sql = f"""
    WITH bars AS (
        SELECT * FROM ohlc_1m
        WHERE symbol = %s AND window_start >= {NOW} - make_interval(mins => %s + 20)
    )
    SELECT * FROM (
        SELECT symbol, window_start, open, high, low, close, volume, vwap, trade_count,
               AVG(close) OVER w5  AS sma_5,
               AVG(close) OVER w20 AS sma_20,
               close / LAG(close) OVER (ORDER BY window_start) - 1 AS ret_1m
        FROM bars
        WINDOW w5  AS (ORDER BY window_start ROWS BETWEEN 4  PRECEDING AND CURRENT ROW),
               w20 AS (ORDER BY window_start ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
    ) x
    WHERE window_start >= {NOW} - make_interval(mins => %s)
    ORDER BY window_start
    """
    return query(sql, (symbol, minutes, minutes))


@app.get("/api/movers")
def movers(minutes: int = Query(15, ge=1, le=240)):
    """Percent change per symbol over the last N minutes, ranked."""
    sql = f"""
    WITH recent AS (
        SELECT symbol, window_start,
               FIRST_VALUE(open) OVER full_p AS first_open,
               LAST_VALUE(close) OVER full_p AS last_close,
               SUM(volume)       OVER (PARTITION BY symbol) AS total_volume,
               ROW_NUMBER()      OVER (PARTITION BY symbol ORDER BY window_start DESC) AS rn
        FROM ohlc_1m
        WHERE window_start >= {NOW} - make_interval(mins => %s)
        WINDOW full_p AS (PARTITION BY symbol ORDER BY window_start
                          ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
    )
    SELECT symbol, first_open, last_close, total_volume,
           (last_close / NULLIF(first_open, 0) - 1) * 100 AS pct_change,
           RANK() OVER (ORDER BY last_close / NULLIF(first_open, 0) DESC) AS gain_rank
    FROM recent WHERE rn = 1
    ORDER BY gain_rank
    """
    return query(sql, (minutes,))


@app.get("/api/alerts")
def alerts(z: float = Query(3.0, ge=1.0), minutes: int = Query(60, ge=5, le=1440)):
    """1-min returns whose z-score vs the previous 20 bars exceeds the threshold."""
    sql = f"""
    WITH r AS (
        SELECT symbol, window_start, close,
               close / LAG(close) OVER (PARTITION BY symbol ORDER BY window_start) - 1 AS ret
        FROM ohlc_1m
        WHERE window_start >= {NOW} - make_interval(mins => %s + 25)
    ),
    s AS (
        SELECT *, AVG(ret) OVER t AS mu, STDDEV_SAMP(ret) OVER t AS sigma
        FROM r
        WINDOW t AS (PARTITION BY symbol ORDER BY window_start ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
    )
    SELECT symbol, window_start, close, ret * 100 AS pct_move,
           (ret - mu) / NULLIF(sigma, 0) AS z_score
    FROM s
    WHERE window_start >= {NOW} - make_interval(mins => %s)
      AND ABS((ret - mu) / NULLIF(sigma, 0)) >= %s
    ORDER BY window_start DESC
    LIMIT 25
    """
    return query(sql, (minutes, minutes, z))


@app.get("/api/volatility")
def volatility():
    """Latest 5-minute sliding-window volatility per symbol."""
    sql = f"""
    SELECT DISTINCT ON (symbol)
           symbol, window_start, window_end, avg_price, volatility,
           volatility / NULLIF(avg_price, 0) * 100 AS volatility_pct, volume, trade_count
    FROM rolling_5m
    WHERE window_start >= {NOW} - INTERVAL '30 minutes'
    ORDER BY symbol, ABS(EXTRACT(EPOCH FROM window_end - {NOW}))
    """
    return query(sql)


# ------------------------------------------------------------------ ML, A/B test, NLP, risk, DQ
@app.get("/api/model/ab")
def ab_summary():
    """Live A/B comparison of model A (Ridge) vs model B (XGBoost) + latest formal test result."""
    live = query("""
        SELECT variant, COUNT(*) AS n,
               AVG(ABS(served_pred - actual))   AS mae,
               AVG(ABS(baseline_pred - actual)) AS baseline_mae
        FROM predictions WHERE actual IS NOT NULL
        GROUP BY variant ORDER BY variant""")
    last_test = query("SELECT * FROM ab_results ORDER BY run_at DESC LIMIT 1")
    return {"live": live, "last_test": last_test[0] if last_test else None}


@app.get("/api/model/predictions/{symbol}")
def predictions(symbol: str, minutes: int = Query(60, ge=5, le=1440)):
    return query(f"""
        SELECT window_start, variant, served_pred, baseline_pred, actual FROM predictions
        WHERE symbol = %s AND window_start >= {NOW} - make_interval(mins => %s)
        ORDER BY window_start""", (symbol, minutes))


@app.get("/api/anomalies")
def anomalies(minutes: int = Query(60, ge=5, le=1440)):
    return query(f"""
        SELECT * FROM anomalies WHERE window_start >= {NOW} - make_interval(mins => %s)
        ORDER BY window_start DESC LIMIT 25""", (minutes,))


@app.get("/api/sentiment")
def sentiment(hours: int = Query(24, ge=1, le=168)):
    """Latest headlines + average sentiment per symbol."""
    latest = query(f"""
        SELECT symbol, published_at, headline, label, score FROM news_sentiment
        WHERE published_at >= {NOW} - make_interval(hours => %s)
        ORDER BY published_at DESC LIMIT 20""", (hours,))
    by_symbol = query(f"""
        SELECT symbol, COUNT(*) AS n, AVG(score) AS avg_score FROM news_sentiment
        WHERE published_at >= {NOW} - make_interval(hours => %s)
        GROUP BY symbol ORDER BY avg_score DESC""", (hours,))
    return {"latest": latest, "by_symbol": by_symbol}


@app.get("/api/risk")
def risk():
    return query("SELECT * FROM risk_summary ORDER BY market, sharpe DESC")


@app.get("/api/dq")
def data_quality():
    return query("SELECT DISTINCT ON (check_name) * FROM dq_report ORDER BY check_name, run_at DESC")


class Question(BaseModel):
    question: str


@app.post("/api/ask")
def ask_market(q: Question):
    """AI assistant: routes to live SQL data and/or RAG (docs, news, reports) and answers with citations."""
    from genai.agent import answer
    if not 3 <= len(q.question) <= 300:
        raise HTTPException(400, "Question must be 3-300 characters.")
    try:
        return answer(q.question)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
