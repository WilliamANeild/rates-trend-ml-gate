cat > features/signals.py << 'EOF'
"""
Signals module.
Input: 
  - prices: pandas DataFrame of daily adjusted closes, columns = tickers
  - yields_df: pandas DataFrame of Treasury yields if available (optional)
Output:
  - momentum_df: z-scored multi-window momentum per ticker
  - carry_df: simple carry proxy per ticker (zeros if yields_df is None)
  - score_df: blended pre-gate score per ticker
Goal:
  Build clean, no-lookahead features for the allocator.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

def _zscore(df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    mu = df.rolling(lookback, min_periods=lookback//4).mean()
    sd = df.rolling(lookback, min_periods=lookback//4).std()
    return (df - mu) / sd

def momentum_features(prices: pd.DataFrame, windows=(5,20,60,120)) -> pd.DataFrame:
    """
    For each window, compute sum of daily returns over that window.
    Return a single DataFrame = average of z-scored windows.
    No lookahead: uses past data only.
    """
    rets = prices.pct_change()
    feats = []
    for w in windows:
        feat = rets.rolling(w, min_periods=max(2, w//2)).sum()
        feats.append(_zscore(feat))
    mom = pd.concat(feats, axis=1).groupby(level=0, axis=1).mean()
    mom = mom.dropna(how="all")
    return mom

def carry_proxy(yields_df: pd.DataFrame | None, tickers: list[str]) -> pd.DataFrame:
    """
    Very simple carry stand-in.
    If yields_df provided, map ETF duration buckets to yield levels.
    If None, return zeros so the pipeline still runs.
    Mapping defaults:
      SHY -> 2y, IEF -> 10y, TLT -> 30y, BIL -> cash ~ 3m, TBF -> negative TLT carry.
    """
    idx = None
    if yields_df is not None and not yields_df.empty:
        idx = yields_df.index
        tenors = {
            "SHY": "DGS2",
            "IEF": "DGS10",
            "TLT": "DGS30",
            "BIL": "DGS3MO",   # will be NaN if not present
            "TBF": "DGS30",    # inverse exposure, sign flipped later
        }
        out = {}
        for t in tickers:
            k = tenors.get(t, None)
            if k is None or k not in yields_df.columns:
                out[t] = pd.Series(0.0, index=idx)
            else:
                s = yields_df[k].astype(float)
                if t == "TBF":
                    s = -s
                out[t] = s
        carry = pd.DataFrame(out).reindex(idx).ffill()
        carry = _zscore(carry)
    else:
        # zeros with the same index as prices will be set by caller
        carry = pd.DataFrame(columns=tickers)
    return carry

def pre_gate_score(momentum_df: pd.DataFrame, carry_df: pd.DataFrame, w_mom=0.7, w_carry=0.3) -> pd.DataFrame:
    """
    Blend momentum and carry into a single score per ticker.
    Align on the intersection of available dates and tickers.
    """
    common_index = momentum_df.index
    common_cols = [c for c in momentum_df.columns if c in carry_df.columns] if not carry_df.empty else momentum_df.columns
    mom = momentum_df[common_cols]
    if carry_df.empty:
        car = pd.DataFrame(0.0, index=common_index, columns=common_cols)
    else:
        car = carry_df.reindex(common_index)[common_cols].fillna(0.0)
    score = w_mom * mom + w_carry * car
    return score.dropna(how="all")

EOF
