"""
BATCH RISK JOB (PySpark) — daily risk metrics using Spark Window functions.

Per stock:
  daily return        close / previous close - 1                 (lag)
  20-day volatility   rolling std of returns * sqrt(252)          (rowsBetween)
  drawdown            close / running max close - 1               (unbounded preceding)
  Sharpe ratio        annual return / annual volatility
  1-day 95% VaR       5th percentile of daily returns (historical method)
  beta vs index       cov(stock, index) / var(index)

Why these? They are the standard numbers a risk or portfolio team looks at every day.
"""
import os
import sys

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.db import execute, write_rows  # noqa: E402

spark = (SparkSession.builder.appName("daily-risk").config("spark.sql.session.timeZone", "UTC")
         .config("spark.sql.shuffle.partitions", "4").getOrCreate())
spark.sparkContext.setLogLevel("WARN")

prices = spark.read.parquet("data/daily/prices.parquet")

by_sym = Window.partitionBy("symbol").orderBy("date")
df = (prices
      .withColumn("daily_return", F.col("close") / F.lag("close").over(by_sym) - 1)
      .withColumn("vol_20d_ann", F.stddev("daily_return").over(by_sym.rowsBetween(-19, 0)) * F.sqrt(F.lit(252)))
      .withColumn("running_max", F.max("close").over(by_sym.rowsBetween(Window.unboundedPreceding, 0)))
      .withColumn("drawdown", F.col("close") / F.col("running_max") - 1))

# join each stock's return with its market index return on the same day (for beta)
idx = (df.filter("is_index").select("market", "date", F.col("daily_return").alias("index_return")))
stocks = df.filter(~F.col("is_index")).join(idx, ["market", "date"], "left")

summary = (stocks.groupBy("symbol", "market").agg(
    F.max("date").alias("as_of"),
    (F.avg("daily_return") * 252).alias("ann_return"),
    (F.stddev("daily_return") * F.sqrt(F.lit(252))).alias("ann_volatility"),
    (-F.percentile_approx("daily_return", 0.05)).alias("var_95_1d"),
    F.min("drawdown").alias("max_drawdown"),
    (F.covar_samp("daily_return", "index_return") / F.var_samp("index_return")).alias("beta"),
).withColumn("sharpe", F.col("ann_return") / F.col("ann_volatility")))

# small results -> collect and write to Postgres (full refresh = safe to re-run)
daily_rows = [tuple(r) for r in stocks.select("symbol", "market", "date", "close", "daily_return",
                                                "vol_20d_ann", "drawdown").collect()]
summary_rows = [tuple(r) for r in summary.select("symbol", "market", "as_of", "ann_return", "ann_volatility",
                                                  "sharpe", "var_95_1d", "max_drawdown", "beta").collect()]
execute("TRUNCATE daily_metrics, risk_summary")
write_rows("INSERT INTO daily_metrics VALUES %s", daily_rows)
write_rows("""INSERT INTO risk_summary (symbol, market, as_of, ann_return, ann_volatility, sharpe,
              var_95_1d, max_drawdown, beta) VALUES %s""", summary_rows)
summary.orderBy("market", F.desc("sharpe")).show(truncate=False)
spark.stop()
