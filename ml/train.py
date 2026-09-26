"""
MODEL TRAINING — volatility forecasting + anomaly detection, tracked in MLflow.

Models compared (all predict next-5-minute volatility):
  baseline  "tomorrow looks like today": predict the current 5-min volatility
  A: Ridge  simple, fast, explainable linear model        (the current "champion")
  B: XGBoost gradient-boosted trees, captures non-linearity (the "challenger")
Both A and B are then served live and compared in an A/B test (ml/ab_test.py).

Plus an Isolation Forest that flags unusual bars (sudden moves / volume spikes).

Run:  python ml/train.py            then   make mlflow  (UI at http://localhost:5000)
"""
import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.db import read_sql  # noqa: E402
from ml.features import FEATURES, TARGET, build_features  # noqa: E402

MODEL_DIR = Path("models")
ANOMALY_FEATURES = ["abs_ret_1", "range_pct", "volume_ratio", "trade_count_ratio", "vol_ratio_5_15"]
MIN_ROWS = 300


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    bars = read_sql("SELECT symbol, window_start, open, high, low, close, volume, trade_count "
                    "FROM ohlc_1m ORDER BY symbol, window_start")
    df = build_features(bars).dropna(subset=FEATURES + [TARGET])
    if len(df) < MIN_ROWS:
        raise SystemExit(f"Only {len(df)} usable rows. Let the pipeline run longer "
                         f"(~45+ minutes with 5 symbols) and try again.")

    # TIME-BASED split: train on the past, test on the most recent 20%.
    # A random split would leak future information into training.
    cutoff = df["window_start"].quantile(0.8)
    train, test = df[df["window_start"] <= cutoff], df[df["window_start"] > cutoff]
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    print(f"train rows: {len(train)}, test rows: {len(test)}")

    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")   # local MLflow tracking database
    mlflow.set_experiment("volatility-forecast")
    MODEL_DIR.mkdir(exist_ok=True)
    results = {}

    with mlflow.start_run(run_name="baseline"):
        m = evaluate(y_te, test["vol_5"])
        mlflow.log_metrics(m)
        results["baseline"] = m

    models = {
        "A_ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "B_xgboost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42),
    }
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_tr, y_tr)
            m = evaluate(y_te, model.predict(X_te))
            mlflow.log_params({"model": name, "n_features": len(FEATURES), "train_rows": len(train)})
            mlflow.log_metrics(m)
            joblib.dump(model, MODEL_DIR / f"{name}.joblib")
            mlflow.log_artifact(str(MODEL_DIR / f"{name}.joblib"))   # model file saved with the run
            results[name] = m

    if hasattr(models["B_xgboost"], "feature_importances_"):
        imp = dict(zip(FEATURES, models["B_xgboost"].feature_importances_.round(4).tolist()))
        print("XGBoost feature importance:", dict(sorted(imp.items(), key=lambda kv: -kv[1])))

    # ---- anomaly detection (unsupervised) ----
    with mlflow.start_run(run_name="isolation_forest"):
        iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
        iso.fit(train[ANOMALY_FEATURES])
        rate = float((iso.predict(test[ANOMALY_FEATURES]) == -1).mean())
        mlflow.log_metric("test_anomaly_rate", rate)
        joblib.dump(iso, MODEL_DIR / "anomaly_iforest.joblib")

    (MODEL_DIR / "metadata.json").write_text(json.dumps(
        {"features": FEATURES, "anomaly_features": ANOMALY_FEATURES, "metrics": results}, indent=2))

    print("\nTest-set results (lower MAE/RMSE is better):")
    for name, m in results.items():
        print(f"  {name:10s} MAE={m['mae']:.6f}  RMSE={m['rmse']:.6f}  R2={m['r2']:.3f}")
    print(f"anomaly rate on test: {rate:.2%}\nModels saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
