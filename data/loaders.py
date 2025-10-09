cat > data/loaders.py << 'EOF'
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

TICKERS = ["SHY","IEF","TLT","BIL","TBF"]

def load_prices(start="2007-01-01") -> pd.DataFrame:
    """Daily adjusted closes for the small ETF basket."""
    df = yf.download(TICKERS, start=start, auto_adjust=True, progress=False)["Adj Close"]
    # ensure columns are in our TICKERS order
    df = df.reindex(columns=TICKERS)
    return df.dropna(how="all")

def load_yields() -> pd.DataFrame:
    """
    Placeholder for FRED yields. Returns empty DataFrame for now so the rest of the
    pipeline can run. Later we will add 2y, 10y, 30y, etc.
    """
    return pd.DataFrame()
EOF
