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
    """
    Time series z-score by column, using a rolling window.
    """
    mu = df.rolling(lookback, min_periods=lookback // 4).mean()
    sd = df.rolling(lookback, min_periods=lookback // 4).std()
    return (df - mu) / sd


def momentum_features(
    prices: pd.DataFrame,
    windows: tuple[int, ...] = (20, 60, 120),
) -> pd.DataFrame:
    """
    Multi-window momentum based on rolling Sharpe of log returns.

    For each window w:
      - compute daily log returns
      - compute rolling mean and std over past w days
      - Sharpe_w = mean_w / std_w
      - time z-score Sharpe_w over a long window

    Then average the z-scored Sharpe values across windows.
    No lookahead: all operations only use past data.
    """
    # log returns are more stable for long histories
    log_prices = np.log(prices)
    rets = log_prices.diff()

    feats_z = []
    for w in windows:
        roll_mean = rets.rolling(w, min_periods=max(5, w // 2)).mean()
        roll_std = rets.rolling(w, min_periods=max(5, w // 2)).std()

        sharpe = roll_mean / roll_std
        sharpe_z = _zscore(sharpe)
        feats_z.append(sharpe_z)

    # simple average of the z-scored windows
    mom = None
    for f in feats_z:
        if mom is None:
            mom = f.copy()
        else:
            mom = mom.add(f, fill_value=0.0)
    mom = mom / len(feats_z)

    mom = mom.dropna(how="all")
    return mom


def carry_proxy(yields_df: pd.DataFrame | None, tickers: list[str]) -> pd.DataFrame:
    """
    Very simple carry stand in.

    If yields_df provided, map ETF duration buckets to yield levels.
    If None, return zeros so the pipeline still runs.

    Mapping defaults:
      SHY -> 2y, IEF -> 10y, TLT -> 30y, BIL -> cash about 3m, TBF -> negative TLT carry.
    """
    if yields_df is None or yields_df.empty:
        # caller will reindex to prices index
        return pd.DataFrame(columns=tickers)

    idx = yields_df.index
    tenors = {
        "SHY": "DGS2",
        "IEF": "DGS10",
        "TLT": "DGS30",
        "BIL": "DGS3MO",  # may be NaN if not present
        "TBF": "DGS30",   # inverse exposure, sign flipped below
    }

    out: dict[str, pd.Series] = {}
    for t in tickers:
        k = tenors.get(t)
        if k is None or k not in yields_df.columns:
            out[t] = pd.Series(0.0, index=idx)
        else:
            s = yields_df[k].astype(float)
            if t == "TBF":
                s = -s
            out[t] = s

    carry = pd.DataFrame(out).reindex(idx).ffill()
    carry = _zscore(carry)
    carry = carry.dropna(how="all")
    return carry


def pre_gate_score(
    momentum_df: pd.DataFrame,
    carry_df: pd.DataFrame,
    w_mom: float = 0.8,
    w_carry: float = 0.2,
) -> pd.DataFrame:
    """
    Blend momentum and carry into a single score per ticker.

    Steps:
      - align on common dates and tickers
      - if carry_df is empty, use zero carry
      - compute weighted sum: score = w_mom * mom + w_carry * carry
      - drop all NaN rows
    """
    common_index = momentum_df.index
    if carry_df.empty:
        common_cols = momentum_df.columns
        car = pd.DataFrame(0.0, index=common_index, columns=common_cols)
    else:
        common_cols = [c for c in momentum_df.columns if c in carry_df.columns]
        if not common_cols:
            # fall back to momentum only
            common_cols = momentum_df.columns
            car = pd.DataFrame(0.0, index=common_index, columns=common_cols)
        else:
            car = carry_df.reindex(common_index)[common_cols].fillna(0.0)

    mom = momentum_df[common_cols]

    score = w_mom * mom + w_carry * car
    score = score.dropna(how="all")
    return score
