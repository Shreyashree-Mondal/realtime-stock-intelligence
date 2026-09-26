"""
FEATURE ENGINEERING — one function used for BOTH training and live scoring,
so the model sees features computed exactly the same way (no training/serving skew).

Input:  1-minute bars (symbol, window_start, open, high, low, close, volume, trade_count)
Output: one row per bar with features + the target.

Target = realised volatility over the NEXT 5 minutes
       = standard deviation of the next five 1-minute returns.
Why volatility and not price? Price direction is close to unpredictable; volatility
"clusters" (calm follows calm, turbulent follows turbulent), so it is forecastable and
useful: it drives risk limits, position sizing and option pricing.
"""
import numpy as np
import pandas as pd

HORIZON = 5  # minutes ahead

FEATURES = [
    "ret_1", "ret_2", "ret_3",          # last three 1-min returns (momentum)
    "abs_ret_1",                        # size of the last move
    "vol_5", "vol_15",                  # recent realised volatility (short and longer)
    "vol_ratio_5_15",                   # is volatility rising or falling?
    "range_pct",                        # (high - low) / close of the last bar
    "volume_ratio",                     # volume vs its 15-min average
    "trade_count_ratio",                # activity vs its 15-min average
    "sma_gap_20",                       # distance from the 20-min moving average
    "rsi_14",                           # relative strength index
]
TARGET = "target_vol_5"


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("window_start").copy()
    r = g["close"].pct_change()
    g["ret_1"], g["ret_2"], g["ret_3"] = r, r.shift(1), r.shift(2)
    g["abs_ret_1"] = r.abs()
    g["vol_5"] = r.rolling(5).std()
    g["vol_15"] = r.rolling(15).std()
    g["vol_ratio_5_15"] = g["vol_5"] / g["vol_15"]
    g["range_pct"] = (g["high"] - g["low"]) / g["close"]
    g["volume_ratio"] = g["volume"] / g["volume"].rolling(15).mean()
    g["trade_count_ratio"] = g["trade_count"] / g["trade_count"].rolling(15).mean()
    g["sma_gap_20"] = g["close"] / g["close"].rolling(20).mean() - 1
    g["rsi_14"] = _rsi(g["close"])
    # target: std of the NEXT 5 returns (uses only future data -> never used as a feature)
    future = pd.concat([r.shift(-k) for k in range(1, HORIZON + 1)], axis=1)
    g[TARGET] = future.std(axis=1, skipna=False)
    return g


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    bars["window_start"] = pd.to_datetime(bars["window_start"])
    out = bars.groupby("symbol", group_keys=False).apply(_per_symbol)
    return out.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
