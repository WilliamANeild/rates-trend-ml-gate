"""
Data loaders (minimal).

Inputs:
  - none (uses hardcoded tickers for now)
Outputs:
  - prices: DataFrame of daily adjusted closes for our ETFs
  - yields_df: DataFrame of Treasury yields (here: simple synthetic curve)

Goal:
  Give the pipeline a clean, simple way to get data without lookahead.
"""

from __future__ import annotations
import pandas as pd
import yfinance as yf


TICKERS = ["SHY", "IEF", "TLT", "BIL", "TBF"]


def load_prices(start: str = "2007-01-01") -> pd.DataFrame:
    """
    Daily adjusted closes for the small ETF basket.

    Handles yfinance returning either:
    - MultiIndex columns with fields like "Adj Close" or "Close"
    - Single Index columns with those fields
    """
    raw = yf.download(TICKERS, start=start, auto_adjust=True, progress=False)

    # yfinance often gives a MultiIndex: level 0 = field, level 1 = ticker
    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0)

        if "Adj Close" in fields:
            px = raw["Adj Close"]
        else:
            px = raw["Close"]
    else:
        # Single Index, no MultiIndex
        if "Adj Close" in raw.columns:
            px = raw["Adj Close"]
        else:
            px = raw["Close"]

    # ensure columns are in our TICKERS order
    px = px.reindex(columns=TICKERS)
    return px.dropna(how="all")


def load_yields(
    series_ids: dict[str, str] | None = None,
    start: str = "2007-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Synthetic Treasury yields instead of calling FRED.

    We stub a simple upward-sloping curve (2y, 5y, 10y, 30y) and line it up
    with the price history index. This avoids external dependencies and
    the distutils/pandas_datareader mess.

    Returns
    -------
    pd.DataFrame
        Columns: 2y, 5y, 10y, 30y.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Business day index matching the price history frequency
    idx = pd.date_range(start, end, freq="B")

    cols = ["2y", "5y", "10y", "30y"]
    # Simple static upward-sloping curve (in percent)
    base_curve = {
        "2y": 3.0,
        "5y": 3.2,
        "10y": 3.5,
        "30y": 3.7,
    }

    yields = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for c in cols:
        yields[c] = base_curve[c]

    return yields
