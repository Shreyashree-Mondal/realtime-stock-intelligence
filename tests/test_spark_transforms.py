"""PySpark transformations tested on a tiny batch DataFrame (no Kafka needed)."""
import json
import time
from datetime import datetime

import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession  # noqa: E402

from streaming.transforms import ohlc_1m, parse_trades  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    s = (SparkSession.builder.master("local[1]").appName("tests")
         .config("spark.sql.session.timeZone", "UTC").config("spark.ui.enabled", "false").getOrCreate())
    yield s
    s.stop()


def kafka_like(spark, records):
    return spark.createDataFrame([(json.dumps(r).encode(),) for r in records], ["value"])


def test_bad_records_get_a_reason(spark):
    now_ms = int(time.time() * 1000)
    rows = [
        {"symbol": "AAPL", "price": 190.0, "volume": 5, "trade_ts": now_ms},
        {"symbol": None, "price": 190.0, "volume": 5, "trade_ts": now_ms},
        {"symbol": "AAPL", "price": -1.0, "volume": 5, "trade_ts": now_ms},
        {"symbol": "AAPL", "price": 190.0, "volume": 5, "trade_ts": now_ms + 3_600_000},
    ]
    out = {r["dq_reason"] for r in parse_trades(kafka_like(spark, rows)).collect()}
    assert out == {None, "missing_symbol", "bad_price", "timestamp_in_future"}


def test_ohlc_open_high_low_close_vwap(spark):
    base = int(datetime(2026, 1, 1, 10, 0).timestamp() * 1000)
    trades = spark.createDataFrame(
        [("AAPL", 100.0, 1.0, base + 1000), ("AAPL", 105.0, 1.0, base + 20000),
         ("AAPL", 95.0, 2.0, base + 40000), ("AAPL", 102.0, 1.0, base + 59000)],
        ["symbol", "price", "volume", "trade_ts"],
    ).selectExpr("*", "CAST(trade_ts / 1000 AS TIMESTAMP) AS event_time")
    bar = ohlc_1m(trades).collect()[0]
    assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 105.0, 95.0, 102.0)
    assert bar.volume == 5.0 and bar.trade_count == 4
    assert abs(bar.vwap - (100 + 105 + 190 + 102) / 5) < 1e-9
