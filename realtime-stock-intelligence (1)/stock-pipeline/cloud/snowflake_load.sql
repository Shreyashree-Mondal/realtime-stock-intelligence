-- Optional: load the S3 data lake into Snowflake (free trial account works).
-- Replace the placeholders; in real projects use a STORAGE INTEGRATION instead of keys.
CREATE DATABASE IF NOT EXISTS STOCKS;
CREATE SCHEMA IF NOT EXISTS STOCKS.RAW;
USE SCHEMA STOCKS.RAW;

CREATE OR REPLACE FILE FORMAT parquet_fmt TYPE = PARQUET;

CREATE OR REPLACE STAGE lake_stage
  URL = 's3://<your-bucket>/stock-pipeline/'
  CREDENTIALS = (AWS_KEY_ID = '<key>' AWS_SECRET_KEY = '<secret>')
  FILE_FORMAT = parquet_fmt;

CREATE OR REPLACE TABLE daily_prices (date DATE, close FLOAT, volume FLOAT, symbol STRING,
                                      market STRING, is_index BOOLEAN);

COPY INTO daily_prices FROM @lake_stage/data/daily/
  MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- Same window-function analytics, now in Snowflake:
SELECT symbol, date, close,
       close / LAG(close) OVER (PARTITION BY symbol ORDER BY date) - 1 AS daily_return,
       AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS sma_50
FROM daily_prices
WHERE NOT is_index
QUALIFY ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) <= 5;
