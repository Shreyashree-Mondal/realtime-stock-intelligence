"""
RAG EVALUATION — does retrieval find the right chunk?

For each test question we know which section of the docs should answer it.
  Hit rate@3 : share of questions where that section is in the top 3 results
  MRR        : mean reciprocal rank (1.0 = always ranked first)
Re-run after changing chunk size or the embedding model to see if retrieval improved.

Run (after the indexer has run once):  python genai/eval_rag.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genai.rag import retrieve  # noqa: E402

EVAL_SET = [
    ("Why are Kafka messages keyed by the stock symbol?", "Kafka"),
    ("What is a watermark and why do we need it?", "Structured Streaming"),
    ("What happens to bad records in the pipeline?", "Data quality"),
    ("Why do we forecast volatility instead of price?", "volatility forecasting"),
    ("Why is the train test split based on time?", "volatility forecasting"),
    ("How does the A/B test decide which model wins?", "A/B testing"),
    ("Why use FinBERT for news sentiment?", "FinBERT"),
    ("How is the text-to-SQL assistant kept safe?", "GenAI assistant"),
    ("What does value at risk mean?", "Batch layer"),
    ("How does retrieval augmented generation work here?", "RAG"),
]


def main(k: int = 3):
    hits, rr = 0, 0.0
    for question, expected in EVAL_SET:
        results = retrieve(question, k=5)
        titles = [r["title"] for r in results]
        rank = next((i for i, t in enumerate(titles, 1) if expected.lower() in t.lower()), None)
        hits += bool(rank and rank <= k)
        rr += 1 / rank if rank else 0
        print(f"{'HIT ' if rank and rank <= k else 'MISS'} rank={rank}  {question}")
    n = len(EVAL_SET)
    print(f"\nHit rate@{k}: {hits / n:.0%}   MRR: {rr / n:.2f}   ({n} questions)")


if __name__ == "__main__":
    main()
