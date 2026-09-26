-- Friendly views for Power BI / Tableau (connect with the Postgres connector,
-- host localhost, db stocks, user analyst_ro). Run once: psql -f sql/powerbi_views.sql
CREATE OR REPLACE VIEW v_intraday AS
SELECT symbol, window_start AS bar_time, open, high, low, close, volume, vwap,
       AVG(close) OVER (PARTITION BY symbol ORDER BY window_start ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20
FROM ohlc_1m;

CREATE OR REPLACE VIEW v_model_performance AS
SELECT date_trunc('hour', window_start) AS hour, variant,
       COUNT(*) AS n, AVG(ABS(served_pred - actual)) AS mae, AVG(ABS(baseline_pred - actual)) AS baseline_mae
FROM predictions WHERE actual IS NOT NULL
GROUP BY 1, 2;

CREATE OR REPLACE VIEW v_risk AS SELECT * FROM risk_summary;

GRANT SELECT ON v_intraday, v_model_performance, v_risk TO analyst_ro;
