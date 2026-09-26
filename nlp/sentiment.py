"""
NEWS SENTIMENT (NLP) — Kafka consumer that scores each headline with FinBERT,
a BERT model fine-tuned on financial text, and stores the result in Postgres.

score = P(positive) - P(negative), from -1 (very negative) to +1 (very positive).

Why FinBERT instead of a general sentiment model? Finance language is different:
"shares fell less than expected" is good news; general models often get this wrong.
"""
import json
import os
import sys
from datetime import datetime, timezone

from confluent_kafka import Consumer
from transformers import pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.db import write_rows  # noqa: E402

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)


def score(headlines: list[str]) -> list[tuple[str, float]]:
    out = []
    for probs in clf(headlines, truncation=True):
        p = {d["label"].lower(): d["score"] for d in probs}
        label = max(p, key=p.get)
        out.append((label, p.get("positive", 0) - p.get("negative", 0)))
    return out


def main():
    consumer = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": "sentiment-scorer",
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})
    consumer.subscribe(["news.raw"])
    print("sentiment scorer running")
    while True:
        msgs = consumer.consume(num_messages=32, timeout=5)
        items = [json.loads(m.value()) for m in msgs if m.error() is None]
        if not items:
            continue
        results = score([it["headline"] for it in items])
        rows = [(it["news_id"], it["symbol"],
                 datetime.fromtimestamp(it["published_ts"], tz=timezone.utc).replace(tzinfo=None),
                 it["headline"], it["source"], label, float(sc))
                for it, (label, sc) in zip(items, results)]
        write_rows("INSERT INTO news_sentiment VALUES %s ON CONFLICT (news_id) DO NOTHING", rows)
        consumer.commit(asynchronous=False)   # commit only after the DB write succeeded
        print(f"scored {len(rows)} headlines")


if __name__ == "__main__":
    main()
