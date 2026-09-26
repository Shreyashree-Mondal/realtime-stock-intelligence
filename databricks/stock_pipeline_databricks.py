# Databricks notebook source
# MAGIC %md
# MAGIC # Stock market streaming pipeline on Databricks (Delta Lake, medallion architecture)
# MAGIC
# MAGIC Same logic as `streaming/spark_stream.py`, but landing in Delta tables:
# MAGIC
# MAGIC | Layer  | Table                          | What it holds                                   |
# MAGIC |--------|--------------------------------|-------------------------------------------------|
# MAGIC | Bronze | `stocks.bronze_trades`         | Raw Kafka messages, untouched (replayable)      |
# MAGIC | Silver | `stocks.silver_trades`         | Parsed, typed, de-duplicated trades             |
# MAGIC | Gold   | `stocks.gold_ohlc_1m`          | 1-min tumbling OHLC + VWAP                      |
# MAGIC | Gold   | `stocks.gold_rolling_5m`       | 5-min sliding avg price and volatility          |
# MAGIC
# MAGIC **Kafka source:** Databricks can't reach `localhost:9092` on your laptop, so point the local
# MAGIC producer at a managed Kafka cluster (e.g. a Confluent Cloud free-tier cluster) by setting
# MAGIC `KAFKA_BOOTSTRAP` plus SASL settings, then enter the same credentials in the widgets below.
# MAGIC
# MAGIC **Compute:** Serverless / Free Edition compute runs streams with `trigger(availableNow=True)`:
# MAGIC each run processes everything new and stops. Schedule this notebook as a Job every few minutes
# MAGIC for near-real-time. On a classic cluster you can switch to `processingTime` triggers.

# COMMAND ----------

dbutils.widgets.text("bootstrap", "", "Kafka bootstrap servers")
dbutils.widgets.text("api_key", "", "Kafka API key")
dbutils.widgets.text("api_secret", "", "Kafka API secret")
dbutils.widgets.text("topic", "trades.raw", "Topic")
dbutils.widgets.text("catalog", "workspace", "Catalog")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = "stocks"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.checkpoints")
CKPT = f"/Volumes/{CATALOG}/{SCHEMA}/checkpoints"
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"

# Better practice: store the key/secret with dbutils.secrets and read them here instead of widgets
kafka_options = {
    "kafka.bootstrap.servers": dbutils.widgets.get("bootstrap"),
    "kafka.security.protocol": "SASL_SSL",
    "kafka.sasl.mechanism": "PLAIN",
    "kafka.sasl.jaas.config": (
        "kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule required "
        f'username="{dbutils.widgets.get("api_key")}" password="{dbutils.widgets.get("api_secret")}";'
    ),
    "subscribe": dbutils.widgets.get("topic"),
    "startingOffsets": "earliest",
    "failOnDataLoss": "false",
}

# COMMAND ----------

# MAGIC %md ## Bronze: raw Kafka records

# COMMAND ----------

from pyspark.sql import functions as F, types as Ty

bronze = (
    spark.readStream.format("kafka").options(**kafka_options).load()
    .select(
        F.col("key").cast("string").alias("key"),
        F.col("value").cast("string").alias("value"),
        "topic", "partition", "offset",
        F.col("timestamp").alias("kafka_ts"),
        F.current_timestamp().alias("ingested_at"),
    )
)

(bronze.writeStream
    .option("checkpointLocation", f"{CKPT}/bronze")
    .trigger(availableNow=True)
    .toTable(T("bronze_trades"))
    .awaitTermination())

# COMMAND ----------

# MAGIC %md ## Silver: parse, validate, de-duplicate

# COMMAND ----------

trade_schema = Ty.StructType([
    Ty.StructField("symbol", Ty.StringType()),
    Ty.StructField("price", Ty.DoubleType()),
    Ty.StructField("volume", Ty.DoubleType()),
    Ty.StructField("trade_ts", Ty.LongType()),
    Ty.StructField("source", Ty.StringType()),
])

silver = (
    spark.readStream.table(T("bronze_trades"))
    .select(F.from_json("value", trade_schema).alias("t"), "kafka_ts")
    .select("t.*", "kafka_ts")
    .withColumn("event_time", (F.col("trade_ts") / 1000).cast("timestamp"))
    .filter(F.col("symbol").isNotNull() & (F.col("price") > 0) & (F.col("volume") >= 0))
    .withWatermark("event_time", "2 minutes")
    .dropDuplicatesWithinWatermark(["symbol", "trade_ts", "price", "volume"])
)

(silver.writeStream
    .option("checkpointLocation", f"{CKPT}/silver")
    .trigger(availableNow=True)
    .toTable(T("silver_trades"))
    .awaitTermination())

# COMMAND ----------

# MAGIC %md ## Gold: windowed aggregations (tumbling + sliding)

# COMMAND ----------

trades = spark.readStream.table(T("silver_trades")).withWatermark("event_time", "2 minutes")

ohlc = (
    trades.groupBy(F.window("event_time", "1 minute"), "symbol")
    .agg(
        F.min_by("price", "event_time").alias("open"),
        F.max("price").alias("high"),
        F.min("price").alias("low"),
        F.max_by("price", "event_time").alias("close"),
        F.sum("volume").alias("volume"),
        (F.sum(F.col("price") * F.col("volume")) / F.nullif(F.sum("volume"), F.lit(0))).alias("vwap"),
        F.count("*").alias("trade_count"),
    )
    .select("symbol", F.col("window.start").alias("window_start"), F.col("window.end").alias("window_end"),
            "open", "high", "low", "close", "volume", "vwap", "trade_count")
)

rolling = (
    trades.groupBy(F.window("event_time", "5 minutes", "1 minute"), "symbol")
    .agg(F.avg("price").alias("avg_price"), F.stddev("price").alias("volatility"),
         F.sum("volume").alias("volume"), F.count("*").alias("trade_count"))
    .select("symbol", F.col("window.start").alias("window_start"), F.col("window.end").alias("window_end"),
            "avg_price", "volatility", "volume", "trade_count")
)

# Append mode emits each window once, after the watermark passes it -> clean, final rows in Delta
for name, df in [("gold_ohlc_1m", ohlc), ("gold_rolling_5m", rolling)]:
    (df.writeStream.outputMode("append")
       .option("checkpointLocation", f"{CKPT}/{name}")
       .trigger(availableNow=True)
       .toTable(T(name))
       .awaitTermination())

# COMMAND ----------

# MAGIC %md ## Analytics with SQL window functions

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Moving averages, returns and z-score anomalies in one pass
# MAGIC WITH bars AS (
# MAGIC   SELECT symbol, window_start, close, volume,
# MAGIC          AVG(close) OVER (PARTITION BY symbol ORDER BY window_start ROWS BETWEEN 4  PRECEDING AND CURRENT ROW) AS sma_5,
# MAGIC          AVG(close) OVER (PARTITION BY symbol ORDER BY window_start ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20,
# MAGIC          close / LAG(close) OVER (PARTITION BY symbol ORDER BY window_start) - 1 AS ret
# MAGIC   FROM workspace.stocks.gold_ohlc_1m
# MAGIC ),
# MAGIC scored AS (
# MAGIC   SELECT *,
# MAGIC          (ret - AVG(ret) OVER w) / NULLIF(STDDEV(ret) OVER w, 0) AS z_score
# MAGIC   FROM bars
# MAGIC   WINDOW w AS (PARTITION BY symbol ORDER BY window_start ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
# MAGIC )
# MAGIC SELECT * FROM scored ORDER BY symbol, window_start DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Daily summary per symbol: rank symbols by return, volume share of the day
# MAGIC WITH daily AS (
# MAGIC   SELECT symbol, DATE(window_start) AS day,
# MAGIC          MIN_BY(open, window_start) AS day_open, MAX_BY(close, window_start) AS day_close,
# MAGIC          MAX(high) AS day_high, MIN(low) AS day_low, SUM(volume) AS day_volume
# MAGIC   FROM workspace.stocks.gold_ohlc_1m
# MAGIC   GROUP BY symbol, DATE(window_start)
# MAGIC )
# MAGIC SELECT *,
# MAGIC        ROUND((day_close / day_open - 1) * 100, 3) AS pct_return,
# MAGIC        RANK() OVER (PARTITION BY day ORDER BY day_close / day_open DESC) AS return_rank,
# MAGIC        ROUND(day_volume / SUM(day_volume) OVER (PARTITION BY day) * 100, 2) AS volume_share_pct
# MAGIC FROM daily
# MAGIC ORDER BY day DESC, return_rank

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC - Build a Databricks SQL / AI/BI dashboard on the gold tables.
# MAGIC - Schedule this notebook as a Job (every 5 minutes) for near-real-time refresh.
# MAGIC - Port the three layers to a Lakeflow Declarative Pipeline (DLT) with data-quality expectations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: train the volatility model on Databricks with MLflow (built in)

# COMMAND ----------

# Uses the same feature code as the local project (upload ml/features.py to the workspace next to this notebook)
# import mlflow
# from features import build_features, FEATURES, TARGET
# pdf = build_features(spark.table(T("gold_ohlc_1m")).toPandas()).dropna(subset=FEATURES + [TARGET])
# with mlflow.start_run(run_name="xgboost_databricks"):
#     from xgboost import XGBRegressor
#     model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05).fit(pdf[FEATURES], pdf[TARGET])
#     mlflow.log_metric("train_rows", len(pdf))
