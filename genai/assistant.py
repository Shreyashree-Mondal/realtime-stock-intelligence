"""
GENAI "ASK THE MARKET" ASSISTANT — plain-English questions answered from live data.

  1. The LLM turns the question into a SQL query (text-to-SQL), given the table schema.
  2. A safety guard checks the SQL (read-only, single SELECT statement).
  3. The query runs as a READ-ONLY database user with a 5-second timeout.
  4. The LLM writes a short answer based only on the returned rows.

Safety is layered on purpose: prompt instructions can be ignored by a model,
so the guard and the read-only database role are what actually enforce it.
Works with any OpenAI-compatible API (Groq free tier, OpenAI, local Ollama, ...).
"""
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

SCHEMA = """
ohlc_1m(symbol, window_start, window_end, open, high, low, close, volume, vwap, trade_count)  -- 1-min candles, UTC
rolling_5m(symbol, window_start, window_end, avg_price, volatility, volume, trade_count)       -- 5-min sliding windows
predictions(symbol, window_start, variant, served_pred, shadow_pred, baseline_pred, actual)   -- volatility forecasts
anomalies(symbol, window_start, anomaly_score, ret_pct, volume_ratio)
news_sentiment(news_id, symbol, published_at, headline, source, label, score)                 -- score -1..1
risk_summary(symbol, market, as_of, ann_return, ann_volatility, sharpe, var_95_1d, max_drawdown, beta)
daily_metrics(symbol, market, date, close, daily_return, vol_20d_ann, drawdown)
Current UTC time in SQL: (now() AT TIME ZONE 'UTC')
"""

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|"
                       r"call|do|execute|vacuum|pg_sleep|set|reset)\b", re.IGNORECASE)


def is_safe_sql(sql: str) -> bool:
    s = sql.strip().rstrip(";").strip()
    if not s or ";" in s:                                   # exactly one statement
        return False
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):  # read queries only
        return False
    return FORBIDDEN.search(s) is None


def _llm(messages):
    from openai import OpenAI
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise RuntimeError("Set LLM_API_KEY in .env to enable the assistant.")
    client = OpenAI(base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"), api_key=key)
    resp = client.chat.completions.create(model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
                                          messages=messages, temperature=0)
    return resp.choices[0].message.content.strip()


def generate_sql(question: str) -> str:
    sql = _llm([
        {"role": "system", "content": "You write one PostgreSQL SELECT query answering the user's question "
                                      "using only these tables:\n" + SCHEMA +
                                      "\nReturn only the SQL, no explanation, no markdown."},
        {"role": "user", "content": question},
    ])
    return re.sub(r"^```(sql)?|```$", "", sql.strip(), flags=re.IGNORECASE).strip().rstrip(";")


def run_sql(sql: str) -> list[dict]:
    """Runs a guarded query as the read-only user. Raises ValueError if the SQL is unsafe."""
    from common.db import read_sql
    if not is_safe_sql(sql):
        raise ValueError("Query blocked by the safety check.")
    df = read_sql(f"SELECT * FROM ({sql}) q LIMIT 100",
                  dsn=os.getenv("PG_RO_DSN", "dbname=stocks user=analyst_ro password=analyst_ro host=localhost"))
    return json.loads(df.to_json(orient="records", date_format="iso"))


def ask(question: str) -> dict:
    """SQL-only answer (the RAG agent in genai/agent.py builds on this)."""
    sql = generate_sql(question)
    try:
        rows = run_sql(sql)
    except ValueError as e:
        return {"question": question, "sql": sql, "rows": [], "answer": str(e)}
    answer = _llm([
        {"role": "system", "content": "Answer the question in 2-3 sentences using ONLY the data given. "
                                      "If the data is empty, say there is not enough data yet. No investment advice."},
        {"role": "user", "content": f"Question: {question}\nData (first rows): {json.dumps(rows[:30])}"},
    ])
    return {"question": question, "sql": sql, "rows": rows, "answer": answer}
