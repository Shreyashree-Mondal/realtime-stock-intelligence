"""Feature engineering must be correct and must NOT look into the future."""
import numpy as np
import pandas as pd

from ml.features import FEATURES, TARGET, build_features


def make_bars(n=60, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    return pd.DataFrame({
        "symbol": "TEST",
        "window_start": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
        "volume": rng.uniform(100, 200, n), "trade_count": rng.integers(10, 50, n),
    })


def test_all_feature_columns_exist():
    out = build_features(make_bars())
    for col in FEATURES + [TARGET]:
        assert col in out.columns


def test_no_future_leakage_in_features():
    """Changing FUTURE prices must not change features at time t."""
    bars = make_bars()
    t = 40
    changed = bars.copy()
    changed.loc[t + 1:, ["close", "high", "low"]] *= 1.5
    a = build_features(bars).iloc[t][FEATURES]
    b = build_features(changed).iloc[t][FEATURES]
    pd.testing.assert_series_equal(a, b)


def test_target_uses_future_and_is_missing_at_the_end():
    out = build_features(make_bars())
    assert out[TARGET].iloc[-5:].isna().all()        # no future data -> no label
    assert out[TARGET].iloc[20:50].notna().all()


def test_target_matches_manual_calculation():
    bars = make_bars()
    out = build_features(bars)
    r = bars["close"].pct_change()
    expected = r.iloc[31:36].std()
    assert np.isclose(out[TARGET].iloc[30], expected)
