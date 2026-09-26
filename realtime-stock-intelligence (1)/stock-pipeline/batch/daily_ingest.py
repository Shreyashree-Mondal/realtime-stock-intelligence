"""
BATCH INGESTION — two years of daily prices for US and Indian stocks (yfinance),
plus each market's index (S&P 500, NIFTY 50) for beta calculation.
Output: data/daily/prices.parquet   (read by the PySpark risk job)
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

UNIVERSE = {
    "US": ["AAPL", "MSFT", "NVDA", "TSLA", "JPM"],
    "IN": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
}
INDEX = {"US": "^GSPC", "IN": "^NSEI"}


def main():
    frames = []
    for market, tickers in UNIVERSE.items():
        for t in tickers + [INDEX[market]]:
            df = yf.download(t, period="2y", interval="1d", auto_adjust=True, progress=False)
            if df.empty:
                print(f"no data for {t}")
                continue
            if isinstance(df.columns, pd.MultiIndex):       # newer yfinance returns MultiIndex
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()[["Date", "Close", "Volume"]]
            df.columns = ["date", "close", "volume"]
            df["symbol"], df["market"], df["is_index"] = t, market, t == INDEX[market]
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    Path("data/daily").mkdir(parents=True, exist_ok=True)
    out.to_parquet("data/daily/prices.parquet", index=False)
    print(f"saved {len(out)} rows for {out.symbol.nunique()} tickers")


if __name__ == "__main__":
    main()
