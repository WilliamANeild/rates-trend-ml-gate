"""
Data loaders (minimal).

Inputs:
  - none (uses hardcoded tickers for now)
Outputs:
  - prices: DataFrame of daily adjusted closes for our ETFs
  - yields_df: DataFrame of Treasury yields if available, else empty DataFrame

Goal:
  Give the pipeline a clean, simple way to get data without lookahead.
"""

from __future__ import annotations
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web


TICKERS = ["SHY","IEF","TLT","BIL","TBF"]

def load_prices(start="2007-01-01") -> pd.DataFrame:
    """Daily adjusted closes for the small ETF basket."""
    df = yf.download(TICKERS, start=start, auto_adjust=True, progress=False)["Adj Close"]
    # ensure columns are in our TICKERS order
    df = df.reindex(columns=TICKERS)
    return df.dropna(how="all")

def load_yields(series_ids: dict[str, str] | None = None,
    start="2007-01-01",
    end=None
) -> pd.DataFrame:
    """
    Download Treasury yields from FRED.
    Default series: 2y, 5y, 10y, 30y (in percent).
    
    Parameters
    ----------
    series_ids : dict[str, str], optional
        Mapping from column name -> FRED series ID.
        Defaults to common Treasury yield curve series.
    start, end : str or datetime, optional
        Date range for data.
    
    Returns
    -------
    pd.DataFrame
        Columns: 2y, 5y, 10y, 30y (or custom).
        Forward-filled, business-day aligned.
    """
    if series_ids is None:
        series_ids = {
            "2y": "DGS2",
            "5y": "DGS5",
            "10y": "DGS10",
            "30y": "DGS30",
        }

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    yields = web.DataReader(list(series_ids.values()), "fred", start, end)
    yields.columns = list(series_ids.keys())
    yields = yields.ffill()
    return yields
