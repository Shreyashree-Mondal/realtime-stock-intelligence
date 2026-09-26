"""
ONLINE SCORING SERVICE (model serving + A/B test assignment)

Every 30 seconds:
  1. reads recent 1-min bars from Postgres
  2. for each symbol's latest *completed* bar:
       - randomly (but reproducibly) assigns it to model A or B   <- the A/B split
       - stores the served prediction, the other model's "shadow" prediction and the baseline
       - scores it with the anomaly model and stores anomalies
  3. back-fills the actual volatility for predictions made 5+ minutes ago
     so the A/B test can compare predictions against reality.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.db import read_sql, write_rows  # noqa: E402
from ml.features import FEATURES, TARGET, build_features  # noqa: E402

MODEL_DIR = Path("models")
LOOP_SECONDS = 30


def assign_variant(symbol: str, window_start) -> str:
    """Deterministic 50/50 split: same input always gets the same variant (like hashing
    a user id in a web A/B test), but across many bars the split is effectively random."""
    h = hashlib.md5(f"{symbol}|{window_start}".encode()).hexdigest()
    return "A" if int(h, 16) % 2 == 0 else "B"


def load_models():
    while not (MODEL_DIR / "metadata.json").exists():
        print("No trained models yet. Waiting... (run `make train` after ~45 min of data)")
        time.sleep(60)
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    return (joblib.load(MODEL_DIR / "A_ridge.joblib"),
            joblib.load(MODEL_DIR / "B_xgboost.joblib"),
            joblib.load(MODEL_DIR / "anomaly_iforest.joblib"),
            meta["anomaly_features"])


def score_once(model_a, model_b, iso, anomaly_features):
    bars = read_sql("""
        SELECT symbol, window_start, open, high, low, close, volume, trade_count FROM ohlc_1m
        WHERE window_start >= (now() AT TIME ZONE 'UTC') - INTERVAL '90 minutes'""")
    if bars.empty:
        return
    feats = build_features(bars)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    completed = feats[feats["window_start"] <= now - pd.Timedelta(minutes=2)]  # allow late trades
    latest = completed.dropna(subset=FEATURES).groupby("symbol").tail(1)

    pred_rows, anomaly_rows = [], []
    if not latest.empty:
        X = latest[FEATURES]
        pa, pb = model_a.predict(X), model_b.predict(X)
        is_anom = iso.predict(latest[anomaly_features]) == -1
        scores = iso.score_samples(latest[anomaly_features])
        for i, row in enumerate(latest.itertuples()):
            v = assign_variant(row.symbol, row.window_start)
            served, shadow = (pa[i], pb[i]) if v == "A" else (pb[i], pa[i])
            pred_rows.append((row.symbol, row.window_start.to_pydatetime(), v,
                              float(served), float(shadow), float(row.vol_5)))
            if is_anom[i]:
                anomaly_rows.append((row.symbol, row.window_start.to_pydatetime(), float(-scores[i]),
                                     float(row.ret_1 * 100), float(row.volume_ratio)))

    write_rows("""INSERT INTO predictions (symbol, window_start, variant, served_pred, shadow_pred, baseline_pred)
                  VALUES %s ON CONFLICT (symbol, window_start) DO NOTHING""", pred_rows)
    write_rows("INSERT INTO anomalies VALUES %s ON CONFLICT DO NOTHING", anomaly_rows)

    # back-fill actuals: target is known once the next 5 bars exist
    actuals = feats.dropna(subset=[TARGET])[["symbol", "window_start", TARGET]]
    write_rows("""UPDATE predictions p SET actual = v.actual
                   FROM (VALUES %s) AS v(symbol, window_start, actual)
                   WHERE p.symbol = v.symbol AND p.window_start = v.window_start::timestamp
                     AND p.actual IS NULL""",
               [(r.symbol, r.window_start.to_pydatetime(), float(getattr(r, TARGET)))
                for r in actuals.itertuples()])
    print(f"scored {len(pred_rows)} bars, {len(anomaly_rows)} anomalies")


def main():
    model_a, model_b, iso, anomaly_features = load_models()
    print("scorer running")
    while True:
        try:
            score_once(model_a, model_b, iso, anomaly_features)
        except Exception as e:  # keep the service alive; log and retry
            print(f"scoring error: {e}")
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()
