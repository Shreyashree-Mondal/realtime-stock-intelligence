"""
RAG KNOWLEDGE BASE — Retrieval-Augmented Generation.

The LLM doesn't know our project, today's news or our model results. RAG fixes that:
  1. INDEX:    split text into chunks, turn each into an embedding (a 384-number vector
               that captures meaning), store it in Postgres with pgvector.
  2. RETRIEVE: embed the question and find the most similar chunks (cosine similarity).
  3. GENERATE: give those chunks to the LLM as context and ask it to answer with citations.

What gets indexed (refreshed every 5 minutes by this script, run as the `rag-indexer` service):
  doc     project documentation (README, docs/*.md): "how does this work, and why?"
  news    headlines with FinBERT sentiment:           "what news came out about NVDA?"
  report  risk metrics, A/B results, data-quality checks, model metrics, written as text

Run:  python genai/rag.py            (loop)      python genai/rag.py --once
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.db import read_sql, write_rows  # noqa: E402

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_model = None


# ---------------------------------------------------------------- embeddings
def embed(texts: list[str]) -> list[list[float]]:
    """Local embedding model (runs on CPU, ~130 MB download on first use, no API key)."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(EMBED_MODEL)
    return [[float(x) for x in v] for v in _model.embed(texts)]


def to_pgvector(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


# ---------------------------------------------------------------- chunking
def chunk_markdown(text: str, default_title: str, max_chars: int = 1200) -> list[tuple[str, str]]:
    """Split on headings, then pack paragraphs into chunks of at most ~max_chars.
    Each chunk keeps its section title, which improves retrieval and citations."""
    chunks = []
    for section in re.split(r"\n(?=#{1,3} )", "\n" + text):
        section = section.strip()
        if not section:
            continue
        first = section.splitlines()[0]
        title = first.lstrip("#").strip() if first.startswith("#") else default_title
        buf = ""
        for para in re.split(r"\n\s*\n", section):
            if buf and len(buf) + len(para) > max_chars:
                chunks.append((title, buf.strip()))
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            chunks.append((title, buf.strip()))
    return chunks


def _chunk_id(*parts) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()


def upsert_chunks(rows: list[dict], embedder=embed) -> int:
    """rows: dicts with source_type, source_id, symbol, title, content."""
    if not rows:
        return 0
    vectors = embedder([f"{r['title']}\n{r['content']}" for r in rows])
    write_rows(
        """INSERT INTO rag_chunks (chunk_id, source_type, source_id, symbol, title, content, embedding)
           VALUES %s ON CONFLICT (chunk_id) DO UPDATE
           SET content = EXCLUDED.content, title = EXCLUDED.title, embedding = EXCLUDED.embedding,
               created_at = now() AT TIME ZONE 'UTC'""",
        [(_chunk_id(r["source_type"], r["source_id"], i), r["source_type"], r["source_id"], r.get("symbol"),
          r["title"], r["content"], to_pgvector(v)) for i, (r, v) in enumerate(zip(rows, vectors))],
        template="(%s, %s, %s, %s, %s, %s, %s::vector)",
    )
    return len(rows)


# ---------------------------------------------------------------- sources
def index_docs() -> int:
    rows = []
    for path in [Path("README.md"), *sorted(Path("docs").glob("*.md"))]:
        if path.exists():
            for title, content in chunk_markdown(path.read_text(encoding="utf-8"), path.stem):
                rows.append({"source_type": "doc", "source_id": path.name, "title": f"{path.name}: {title}",
                             "content": content})
    # one source_id per file + position keeps ids stable across re-runs
    for i, r in enumerate(rows):
        r["source_id"] = f"{r['source_id']}#{i}"
    return upsert_chunks(rows)


def index_news() -> int:
    news = read_sql("""
        SELECT n.news_id, n.symbol, n.published_at, n.headline, n.source, n.label, n.score
        FROM news_sentiment n
        LEFT JOIN rag_chunks c ON c.source_type = 'news' AND c.source_id = n.news_id
        WHERE c.chunk_id IS NULL ORDER BY n.published_at DESC LIMIT 500""")
    rows = [{"source_type": "news", "source_id": r.news_id, "symbol": r.symbol,
             "title": f"News {r.symbol} {r.published_at:%Y-%m-%d %H:%M} UTC",
             "content": f"{r.headline} (source: {r.source}; FinBERT sentiment {r.label}, score {r.score:+.2f})"}
            for r in news.itertuples()]
    return upsert_chunks(rows)


def _f(v, spec: str) -> str:
    """Format a number, or 'n/a' if missing."""
    try:
        return format(float(v), spec) if v is not None and v == v else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def index_reports() -> int:
    """Turn tables into short natural-language 'reports' so the assistant can explain them."""
    rows = []
    for r in read_sql("SELECT * FROM risk_summary").itertuples():
        rows.append({"source_type": "report", "source_id": f"risk-{r.symbol}", "symbol": r.symbol,
                     "title": f"Risk report {r.symbol} ({r.market}) as of {r.as_of}",
                     "content": (f"{r.symbol} over 2 years: annualised return {_f(r.ann_return, '.1%')}, "
                                 f"annualised volatility {_f(r.ann_volatility, '.1%')}, Sharpe ratio {_f(r.sharpe, '.2f')}, "
                                 f"1-day 95% VaR {_f(r.var_95_1d, '.2%')}, max drawdown {_f(r.max_drawdown, '.1%')}, "
                                 f"beta vs index {_f(r.beta, '.2f')}.")})
    ab = read_sql("SELECT * FROM ab_results ORDER BY run_at DESC LIMIT 1")
    if not ab.empty:
        a = ab.iloc[0]
        rows.append({"source_type": "report", "source_id": "ab-latest", "title": f"Latest A/B test ({a.run_at:%Y-%m-%d %H:%M})",
                     "content": (f"Model A (Ridge) MAE {a.mae_a:.6f} on {a.n_a} bars; model B (XGBoost) MAE {a.mae_b:.6f} "
                                 f"on {a.n_b} bars; naive baseline MAE {a.mae_baseline:.6f}. 95% CI for MAE_B - MAE_A: "
                                 f"[{a.diff_ci_low:.6f}, {a.diff_ci_high:.6f}], p-value {a.p_value:.4f}. "
                                 f"Decision: {a.decision}.")})
    dq = read_sql("SELECT DISTINCT ON (check_name) check_name, status, detail, run_at FROM dq_report "
                  "ORDER BY check_name, run_at DESC")
    if not dq.empty:
        rows.append({"source_type": "report", "source_id": "dq-latest", "title": "Latest data-quality report",
                     "content": " ".join(f"{r.check_name}: {r.status} ({r.detail})." for r in dq.itertuples())})
    meta = Path("models/metadata.json")
    if meta.exists():
        m = json.loads(meta.read_text())["metrics"]
        rows.append({"source_type": "report", "source_id": "model-metrics", "title": "Model training results (test set)",
                     "content": " ".join(f"{k}: MAE {v['mae']:.6f}, RMSE {v['rmse']:.6f}, R2 {v['r2']:.3f}."
                                         for k, v in m.items())})
    return upsert_chunks(rows)


# ---------------------------------------------------------------- retrieval
def known_symbols() -> list[str]:
    return read_sql("SELECT DISTINCT symbol FROM rag_chunks WHERE symbol IS NOT NULL")["symbol"].tolist()


def detect_symbol(question: str, symbols: list[str]) -> str | None:
    q = question.upper()
    for s in sorted(symbols, key=len, reverse=True):
        short = s.split(":")[-1].replace(".NS", "")
        if re.search(rf"\b{re.escape(short)}\b", q):
            return s
    return None


def retrieve(question: str, k: int = 5, embedder=embed) -> list[dict]:
    """Top-k most similar chunks. If the question names a stock, prefer chunks about it."""
    qv = to_pgvector(embedder([question])[0])
    symbol = detect_symbol(question, known_symbols())
    where = "WHERE symbol = %s OR symbol IS NULL" if symbol else ""
    params = ((qv, symbol, qv, k) if symbol else (qv, qv, k))
    df = read_sql(f"""
        SELECT source_type, source_id, symbol, title, content,
               1 - (embedding <=> %s::vector) AS similarity
        FROM rag_chunks {where}
        ORDER BY embedding <=> %s::vector LIMIT %s""", params)
    return df.to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    print(f"indexed {index_docs()} doc chunks")
    while True:
        try:
            print(f"indexed {index_news()} news, {index_reports()} report chunks")
        except Exception as e:           # tables may be empty on a fresh start
            print(f"indexing skipped: {e}")
        if args.once:
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
