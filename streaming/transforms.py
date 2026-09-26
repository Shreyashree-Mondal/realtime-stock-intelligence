"""
Pure PySpark transformations, kept separate from the streaming job so they can be
unit-tested on small batch DataFrames (see tests/test_spark_transforms.py).
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

TRADE_SCHEMA = T.StructType([
    T.StructField("symbol", T.StringType()),
    T.StructField("price", T.DoubleType()),
    T.StructField("volume", T.DoubleType()),
    T.StructField("trade_ts", T.LongType()),
    T.StructField("source", T.StringType()),
])


def parse_trades(raw: DataFrame) -> DataFrame:
    """Kafka rows (binary `value`) -> typed trade columns + event_time + dq_reason.
    dq_reason is NULL for good rows and explains why a row is bad otherwise."""
    parsed = (
        raw.select(F.col("value").cast("string").alias("raw_value"))
        .withColumn("t", F.from_json("raw_value", TRADE_SCHEMA))
        .select("raw_value", "t.*")
        .withColumn("event_time", (F.col("trade_ts") / 1000).cast("timestamp"))
    )
    now = F.current_timestamp()
    reason = (
        F.when(F.col("symbol").isNull(), "missing_symbol")
        .when(F.col("price").isNull() | (F.col("price") <= 0), "bad_price")
        .when(F.col("volume").isNull() | (F.col("volume") < 0), "bad_volume")
        .when(F.col("event_time").isNull(), "bad_timestamp")
        .when(F.col("event_time") > now + F.expr("INTERVAL 5 MINUTES"), "timestamp_in_future")
        .when(F.col("event_time") < now - F.expr("INTERVAL 1 DAY"), "timestamp_too_old")
    )
    return parsed.withColumn("dq_reason", reason)


def ohlc_1m(trades: DataFrame) -> DataFrame:
    """1-minute TUMBLING window candles with VWAP."""
    return (
        trades.groupBy(F.window("event_time", "1 minute"), "symbol")
        .agg(
            F.min_by("price", "event_time").alias("open"),
            F.max("price").alias("high"),
            F.min("price").alias("low"),
            F.max_by("price", "event_time").alias("close"),
            F.sum("volume").alias("volume"),
            F.sum(F.col("price") * F.col("volume")).alias("pv"),
            F.count("*").alias("trade_count"),
        )
        .select(
            "symbol",
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "open", "high", "low", "close", "volume",
            F.when(F.col("volume") > 0, F.col("pv") / F.col("volume")).alias("vwap"),
            "trade_count",
        )
    )


def rolling_5m(trades: DataFrame) -> DataFrame:
    """5-minute SLIDING window (moves every 1 minute): average price and volatility."""
    return (
        trades.groupBy(F.window("event_time", "5 minutes", "1 minute"), "symbol")
        .agg(
            F.avg("price").alias("avg_price"),
            F.stddev("price").alias("volatility"),
            F.sum("volume").alias("volume"),
            F.count("*").alias("trade_count"),
        )
        .select(
            "symbol",
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "avg_price", "volatility", "volume", "trade_count",
        )
    )
