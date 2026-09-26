"""
Ingestion layer: streams trades into Kafka topic `trades.raw`.

  python producer/producer.py --mode finnhub     # real trades (free Finnhub key)
  python producer/producer.py --mode simulate    # synthetic random-walk trades, works any time

Each message is keyed by symbol so all trades for one ticker land on the same
partition (preserves per-symbol ordering).
"""
import argparse
import json
import math
import os
import random
import time

import websocket
from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.getenv("RAW_TOPIC", "trades.raw")
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "AAPL,MSFT,NVDA,TSLA,BINANCE:BTCUSDT").split(",")]

producer = Producer({
    "bootstrap.servers": BOOTSTRAP,
    "acks": "all",
    "enable.idempotence": True,   # no duplicates on producer retries
    "linger.ms": 50,              # small batching for throughput
    "compression.type": "lz4",
})
_sent = 0


def _on_delivery(err, msg):
    if err is not None:
        print(f"delivery failed: {err}")


def publish(trade: dict) -> None:
    global _sent
    producer.produce(TOPIC, key=trade["symbol"], value=json.dumps(trade), on_delivery=_on_delivery)
    producer.poll(0)
    _sent += 1
    if _sent % 500 == 0:
        print(f"sent {_sent} trades")


# ---------------------------------------------------------------- Finnhub (real data)
def run_finnhub(token: str) -> None:
    def on_open(ws):
        for s in SYMBOLS:
            ws.send(json.dumps({"type": "subscribe", "symbol": s}))
        print(f"subscribed: {SYMBOLS}")

    def on_message(ws, message):
        msg = json.loads(message)
        if msg.get("type") != "trade":          # skip pings
            return
        for d in msg.get("data", []):
            publish({
                "symbol": d["s"],
                "price": float(d["p"]),
                "volume": float(d.get("v") or 0),
                "trade_ts": int(d["t"]),        # epoch millis, event time
                "source": "finnhub",
            })

    def on_error(ws, error):
        print(f"websocket error: {error}")

    def on_close(ws, code, reason):
        print(f"websocket closed ({code}): {reason}")

    while True:  # auto-reconnect
        ws = websocket.WebSocketApp(
            f"wss://ws.finnhub.io?token={token}",
            on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close,
        )
        ws.run_forever(ping_interval=30, ping_timeout=10)
        print("reconnecting in 5s...")
        time.sleep(5)


# ---------------------------------------------------------------- Simulator (demo / off-hours)
def run_simulator(rate: float) -> None:
    """Random-walk prices with *volatility clustering*: calm and turbulent periods
    alternate, like real markets. This gives the ML model something real to learn."""
    prices = {s: random.uniform(80, 500) for s in SYMBOLS}
    base_vol = 0.0001          # per-trade volatility (realistic minute-level moves)
    vol = {s: base_vol for s in SYMBOLS}
    print(f"simulating {rate} trades/sec for {SYMBOLS}")
    while True:
        s = random.choice(SYMBOLS)
        # volatility drifts slowly and is pulled back toward its long-run level
        vol[s] = min(max(vol[s] * math.exp(random.gauss(0, 0.03)) + 0.01 * (base_vol - vol[s]),
                         base_vol / 4), base_vol * 6)
        prices[s] *= math.exp(random.gauss(0, vol[s]))
        if random.random() < 0.002:                               # rare jump so anomaly alerts fire
            prices[s] *= random.choice([1.025, 0.975])
            vol[s] *= 2                                           # shocks raise volatility
        publish({
            "symbol": s,
            "price": round(prices[s], 4),
            "volume": round(random.expovariate(1 / 50) * (vol[s] / base_vol), 2) + 1,
            "trade_ts": int(time.time() * 1000),
            "source": "simulator",
        })
        time.sleep(1 / rate)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["finnhub", "simulate"], default="simulate")
    ap.add_argument("--rate", type=float, default=20, help="simulator trades per second")
    args = ap.parse_args()
    try:
        if args.mode == "finnhub":
            token = os.getenv("FINNHUB_TOKEN")
            if not token or token == "your_key_here":
                raise SystemExit("Set FINNHUB_TOKEN in .env (free at finnhub.io) or use --mode simulate")
            run_finnhub(token)
        else:
            run_simulator(args.rate)
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush(10)
        print(f"flushed, total sent {_sent}")
