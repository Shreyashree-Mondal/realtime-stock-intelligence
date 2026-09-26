"""
NEWS INGESTION — company news headlines into Kafka topic `news.raw`.

  --mode finnhub   polls Finnhub's company-news REST API every 2 minutes (free key)
  --mode simulate  emits sample headlines, for demos
"""
import argparse
import json
import os
import random
import time
from datetime import date, timedelta

import requests
from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = "news.raw"
# company news exists for stocks only (not crypto)
SYMBOLS = [s for s in os.getenv("SYMBOLS", "AAPL,MSFT,NVDA,TSLA").split(",") if ":" not in s]
producer = Producer({"bootstrap.servers": BOOTSTRAP})

SAMPLE = [
    "{s} beats quarterly earnings expectations as revenue climbs",
    "{s} shares slide after weak guidance for next quarter",
    "Analysts upgrade {s} citing strong demand",
    "{s} faces regulatory probe over data practices",
    "{s} announces share buyback program",
    "{s} misses revenue estimates amid supply constraints",
    "{s} unveils new product line at annual event",
]


def send(item: dict):
    producer.produce(TOPIC, key=item["symbol"], value=json.dumps(item))
    producer.poll(0)


def run_finnhub(token: str):
    seen = set()
    while True:
        today = date.today()
        for s in SYMBOLS:
            r = requests.get("https://finnhub.io/api/v1/company-news", timeout=10, params={
                "symbol": s, "from": str(today - timedelta(days=1)), "to": str(today), "token": token})
            if r.status_code != 200:
                print(f"{s}: HTTP {r.status_code}")
                continue
            for n in r.json():
                nid = str(n.get("id"))
                if nid in seen:
                    continue
                seen.add(nid)
                send({"news_id": nid, "symbol": s, "headline": n.get("headline", ""),
                      "source": n.get("source", ""), "published_ts": int(n.get("datetime", 0))})
            time.sleep(1.1)  # stay under the free-tier rate limit
        producer.flush(5)
        print(f"news polled, {len(seen)} unique headlines so far")
        time.sleep(120)


def run_simulator():
    i = 0
    while True:
        s = random.choice(SYMBOLS)
        send({"news_id": f"sim-{i}", "symbol": s, "headline": random.choice(SAMPLE).format(s=s),
              "source": "simulator", "published_ts": int(time.time())})
        i += 1
        time.sleep(random.uniform(10, 40))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["finnhub", "simulate"], default="simulate")
    args = ap.parse_args()
    run_finnhub(os.environ["FINNHUB_TOKEN"]) if args.mode == "finnhub" else run_simulator()
