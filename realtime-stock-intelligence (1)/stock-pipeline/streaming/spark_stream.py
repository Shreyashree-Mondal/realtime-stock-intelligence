"""
STREAM PROCESSING (PySpark Structured Streaming)

Reads trades from Kafka and runs four streaming queries:
  1. bronze     raw good trades -> Parquet files (full history, can be replayed)
  2. dq_rejects bad records     -> Postgres table (quarantined, never silently dropped)
  3. ohlc_1m    1-min tumbling window candles     -> Postgres
  4. rolling_5m 5-min sliding window volatility   -> Postgres

Run:  python streaming/spark_stream.py     (Spark UI at http://localhost:4040)
"""
import os
import sys

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streaming.transforms import ohlc_1m, parse_trades, rolling_5m  # noqa: E402

load_dotenv()
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.getenv("RAW_TOPIC", "trades.raw")
PG_DSN = os.getenv("PG_DSN", "dbname=stocks user=stocks password=stocks host=localhost port=5432")
CHECKPOINT = "data/checkpoints"

spark = (
    SparkSession.builder.appName("stock-market-stream")
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

raw = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", BOOTSTRAP)
    .option("subscribe", TOPIC)
    .option("startingOffsets", "latest")
    .option("failOnDataLoss", "false")
    .load()
)
checked = parse_trades(raw)
good = checked.filter(F.col("dq_reason").isNull()).drop("dq_reason", "raw_value")
bad = checked.filter(F.col("dq_reason").isNotNull()).select(F.col("dq_reason").alias("reason"), "raw_value")

# watermark: accept trades up to 2 minutes late, then finalise the window
good_wm = good.withWatermark("event_time", "2 minutes")


def to_postgres(table, cols, keys=None):
    """foreachBatch sink. With keys -> idempotent UPSERT; without keys -> plain INSERT."""
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES %s"
    if keys:
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in keys)
        sql += f" ON CONFLICT ({', '.join(keys)}) DO UPDATE SET {updates}"

    def _write(batch_df, batch_id):
        rows = [tuple(r) for r in batch_df.select(*cols).collect()]
        if not rows:
            return
        conn = psycopg2.connect(PG_DSN)
        try:
            with conn, conn.cursor() as cur:
                execute_values(cur, sql, rows)
        finally:
            conn.close()
        print(f"[{table}] batch {batch_id}: {len(rows)} rows")

    return _write


queries = [
    good.withColumn("date", F.to_date("event_time"))
        .writeStream.format("parquet").option("path", "data/bronze/trades")
        .option("checkpointLocation", f"{CHECKPOINT}/bronze").partitionBy("date")
        .trigger(processingTime="30 seconds").start(),

    bad.writeStream.foreachBatch(to_postgres("dq_rejects", ["reason", "raw_value"]))
        .option("checkpointLocation", f"{CHECKPOINT}/dq_rejects")
        .trigger(processingTime="30 seconds").start(),

    ohlc_1m(good_wm).writeStream.outputMode("update")
        .foreachBatch(to_postgres("ohlc_1m",
                                  ["symbol", "window_start", "window_end", "open", "high", "low",
                                   "close", "volume", "vwap", "trade_count"],
                                  ["symbol", "window_start"]))
        .option("checkpointLocation", f"{CHECKPOINT}/ohlc_1m")
        .trigger(processingTime="5 seconds").start(),

    rolling_5m(good_wm).writeStream.outputMode("update")
        .foreachBatch(to_postgres("rolling_5m",
                                  ["symbol", "window_start", "window_end", "avg_price",
                                   "volatility", "volume", "trade_count"],
                                  ["symbol", "window_start"]))
        .option("checkpointLocation", f"{CHECKPOINT}/rolling_5m")
        .trigger(processingTime="10 seconds").start(),
]

print("4 streaming queries running (Spark UI: http://localhost:4040)")
spark.streams.awaitAnyTermination()
