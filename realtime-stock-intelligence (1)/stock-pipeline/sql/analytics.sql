-- =====================================================================
-- Real-time analytics on the gold tables using SQL window functions.
-- The API runs parameterised versions of these; run them directly in
-- psql to explore:  docker exec -it postgres psql -U stocks -d stocks
-- =====================================================================

-- 1) Candles + moving averages + 1-minute return
--    AVG() OVER a ROWS frame = simple moving average; LAG() = previous bar
WITH bars AS (
    SELECT *
    FROM ohlc_1m
    WHERE symbol = 'AAPL'
      AND window_start >= (now() AT TIME ZONE 'UTC') - INTERVAL '80 minutes'  -- 60 + 20 warm-up
)
SELECT *
FROM (
    SELECT symbol, window_start, open, high, low, close, volume, vwap,
           AVG(close) OVER w5  AS sma_5,
           AVG(close) OVER w20 AS sma_20,
           close / LAG(close) OVER (PARTITION BY symbol ORDER BY window_start) - 1 AS ret_1m,
           SUM(volume) OVER (PARTITION BY symbol ORDER BY window_start) AS cum_volume
    FROM bars
    WINDOW w5  AS (PARTITION BY symbol ORDER BY window_start ROWS BETWEEN 4  PRECEDING AND CURRENT ROW),
           w20 AS (PARTITION BY symbol ORDER BY window_start ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
) x
WHERE window_start >= (now() AT TIME ZONE 'UTC') - INTERVAL '60 minutes'
ORDER BY window_start;


-- 2) Top movers over the last 15 minutes
--    FIRST_VALUE / LAST_VALUE over the full partition, ROW_NUMBER to keep one row, RANK to order
WITH recent AS (
    SELECT symbol, window_start, close,
           FIRST_VALUE(open)  OVER full_p AS first_open,
           LAST_VALUE(close)  OVER full_p AS last_close,
           SUM(volume)        OVER (PARTITION BY symbol) AS total_volume,
           ROW_NUMBER()       OVER (PARTITION BY symbol ORDER BY window_start DESC) AS rn
    FROM ohlc_1m
    WHERE window_start >= (now() AT TIME ZONE 'UTC') - INTERVAL '15 minutes'
    WINDOW full_p AS (PARTITION BY symbol ORDER BY window_start
                      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
)
SELECT symbol, first_open, last_close, total_volume,
       (last_close / first_open - 1) * 100 AS pct_change,
       RANK() OVER (ORDER BY last_close / first_open DESC) AS gain_rank
FROM recent
WHERE rn = 1
ORDER BY gain_rank;


-- 3) Anomaly alerts: z-score of each 1-min return against the previous 20 returns
--    (the frame ends at 1 PRECEDING so the current bar doesn't dilute its own baseline)
WITH r AS (
    SELECT symbol, window_start, close,
           close / LAG(close) OVER (PARTITION BY symbol ORDER BY window_start) - 1 AS ret
    FROM ohlc_1m
    WHERE window_start >= (now() AT TIME ZONE 'UTC') - INTERVAL '3 hours'
),
s AS (
    SELECT *,
           AVG(ret)         OVER t AS mu,
           STDDEV_SAMP(ret) OVER t AS sigma
    FROM r
    WINDOW t AS (PARTITION BY symbol ORDER BY window_start ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
)
SELECT symbol, window_start, close, ret * 100 AS pct_move,
       (ret - mu) / NULLIF(sigma, 0) AS z_score
FROM s
WHERE ABS((ret - mu) / NULLIF(sigma, 0)) >= 3
ORDER BY window_start DESC
LIMIT 20;


-- 4) Moving-average crossover signals (SMA5 crossing SMA20)
--    LAG on a derived column to detect a sign change between consecutive bars
WITH ma AS (
    SELECT symbol, window_start, close,
           AVG(close) OVER (PARTITION BY symbol ORDER BY window_start ROWS BETWEEN 4  PRECEDING AND CURRENT ROW)
         - AVG(close) OVER (PARTITION BY symbol ORDER BY window_start ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS spread,
           COUNT(*)   OVER (PARTITION BY symbol ORDER BY window_start) AS n
    FROM ohlc_1m
    WHERE window_start >= (now() AT TIME ZONE 'UTC') - INTERVAL '3 hours'
),
flagged AS (
    SELECT *, LAG(spread) OVER (PARTITION BY symbol ORDER BY window_start) AS prev_spread
    FROM ma
)
SELECT symbol, window_start, close,
       CASE WHEN prev_spread <= 0 AND spread > 0 THEN 'bullish cross'
            WHEN prev_spread >= 0 AND spread < 0 THEN 'bearish cross' END AS signal
FROM flagged
WHERE n >= 20
  AND ((prev_spread <= 0 AND spread > 0) OR (prev_spread >= 0 AND spread < 0))
ORDER BY window_start DESC;


-- 5) Current 5-minute volatility per symbol (from the sliding-window table)
--    DISTINCT ON picks the sliding window whose end is closest to now
SELECT DISTINCT ON (symbol)
       symbol, window_start, window_end, avg_price, volatility,
       volatility / NULLIF(avg_price, 0) * 100 AS volatility_pct, volume
FROM rolling_5m
ORDER BY symbol, ABS(EXTRACT(EPOCH FROM window_end - (now() AT TIME ZONE 'UTC')));
