"""
DATA-QUALITY REPORT — runs daily (Airflow) and fails loudly if data looks wrong.

Checks: freshness, completeness (nulls), validity (OHLC logic), uniqueness,
quarantined records, and prediction back-fill. Results go to the dq_report table.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.db import read_sql, write_rows  # noqa: E402

CHECKS = {
    # name: (sql returning one number, rule, description)
    "freshness_minutes": ("SELECT EXTRACT(EPOCH FROM (now() AT TIME ZONE 'UTC') - MAX(window_end)) / 60 FROM ohlc_1m",
                          lambda v: v is not None and v < 10, "latest bar is less than 10 min old"),
    "null_prices": ("SELECT COUNT(*) FROM ohlc_1m WHERE open IS NULL OR close IS NULL",
                    lambda v: v == 0, "no bars with missing prices"),
    "invalid_ohlc": ("SELECT COUNT(*) FROM ohlc_1m WHERE high < low OR close > high OR close < low",
                     lambda v: v == 0, "high >= close >= low for every bar"),
    "duplicate_bars": ("SELECT COUNT(*) - COUNT(DISTINCT (symbol, window_start)) FROM ohlc_1m",
                       lambda v: v == 0, "one bar per symbol per minute"),
    "rejects_last_24h": ("SELECT COUNT(*) FROM dq_rejects WHERE rejected_at > (now() AT TIME ZONE 'UTC') - INTERVAL '1 day'",
                         lambda v: v < 1000, "fewer than 1000 quarantined records per day"),
    "pending_actuals": ("SELECT COUNT(*) FROM predictions WHERE actual IS NULL AND "
                        "window_start < (now() AT TIME ZONE 'UTC') - INTERVAL '30 minutes'",
                        lambda v: v < 50, "predictions get their actual outcome back-filled"),
}


def main():
    rows, failed = [], []
    for name, (sql, rule, desc) in CHECKS.items():
        value = read_sql(sql).iloc[0, 0]
        value = None if value is None else float(value)
        status = "PASS" if rule(value) else "FAIL"
        rows.append((name, status, f"{desc} (value={value})"))
        if status == "FAIL":
            failed.append(name)
        print(f"{status}  {name:20s} {desc}  value={value}")
    write_rows("INSERT INTO dq_report (check_name, status, detail) VALUES %s", rows)
    if failed:
        raise SystemExit(f"Data-quality checks failed: {failed}")


if __name__ == "__main__":
    main()
