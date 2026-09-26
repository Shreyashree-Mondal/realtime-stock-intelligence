"""The GenAI assistant must only ever run single read-only queries."""
import pytest

from genai.assistant import is_safe_sql


@pytest.mark.parametrize("sql", [
    "SELECT * FROM ohlc_1m",
    "WITH x AS (SELECT 1) SELECT * FROM x",
    "select symbol, avg(close) from ohlc_1m group by symbol;",
])
def test_allows_read_queries(sql):
    assert is_safe_sql(sql)


@pytest.mark.parametrize("sql", [
    "DROP TABLE ohlc_1m",
    "SELECT 1; DELETE FROM predictions",
    "UPDATE predictions SET actual = 0",
    "WITH d AS (DELETE FROM anomalies RETURNING *) SELECT * FROM d",
    "SELECT pg_sleep(100)",
    "",
])
def test_blocks_everything_else(sql):
    assert not is_safe_sql(sql)
