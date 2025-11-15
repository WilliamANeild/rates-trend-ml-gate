import numpy as np
import pandas as pd


def estimate_vol(
    returns: pd.DataFrame | pd.Series,
    window: int | None = None,
    lookback: int = 63,
    annualization: int = 252,
) -> pd.DataFrame | pd.Series:
    """
    Rolling volatility estimate.

    Parameters
    ----------
    returns : DataFrame or Series
        Daily returns (index = dates, columns = assets).
    window : int, optional
        Alias for lookback. If provided, overrides lookback.
    lookback : int
        Rolling window length in days.
    annualization : int
        Trading days per year for annualizing vol.

    Returns
    -------
    rolling_vol : DataFrame or Series
        Rolling annualized volatility with the same shape as `returns`.
    """
    if window is not None:
        lookback = window

    rolling_std = returns.rolling(window=lookback, min_periods=lookback).std()
    rolling_vol = rolling_std * np.sqrt(annualization)
    return rolling_vol


def risk_parity_scale(
    raw_exposure: pd.DataFrame,
    vol_est: pd.DataFrame,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Turn raw signals into risk-balanced weights using inverse-vol scaling.

    Parameters
    ----------
    raw_exposure : DataFrame
        Signals or tilts for each asset (e.g. 0/1, -1/1, or scores).
    vol_est : DataFrame
        Volatility estimates for each asset (same shape as raw_exposure).
    eps : float
        Small number to avoid division by zero.

    Returns
    -------
    weights : DataFrame
        Row-wise weights that sum to 1 (or 0 if the row is all NaN).
    """
    # Align vol to signals
    vol_est = vol_est.reindex_like(raw_exposure)

    inv_vol = 1.0 / (vol_est + eps)
    inv_vol = inv_vol.fillna(1.0)

    risk_scores = raw_exposure * inv_vol

    abs_sum = risk_scores.abs().sum(axis=1).replace(0, np.nan)
    weights = risk_scores.div(abs_sum, axis=0)

    weights = weights.fillna(0.0)
    return weights


def apply_gate(
    base_weights: pd.DataFrame,
    gated_signal: pd.DataFrame,
    alpha: float = 0.7,
) -> pd.DataFrame:
    """
    Tilt the portfolio toward the strongest ML-gated signals.

    Logic:
      - Use gated_signal as a strength measure (only care about positives).
      - Each day, build an "overlay" portfolio that invests only in
        assets with positive signals, proportional to their signal size.
      - Final weights = (1 - alpha) * base_weights + alpha * overlay
        on days with positive signals; otherwise just base_weights.

    Parameters
    ----------
    base_weights : DataFrame
        Baseline portfolio weights (risk-parity from scores).
    gated_signal : DataFrame
        Score * ML multiplier, same shape as base_weights.
    alpha : float
        Fraction of capital shifted into the overlay (0 = no tilt, 1 = all overlay).

    Returns
    -------
    tilted_weights : DataFrame
        New weights after tilting toward strongest signals.
    """
    # Align shapes
    bw, gs = base_weights.align(gated_signal, join="inner")

    # Only use positive signals for allocating extra capital
    pos = gs.clip(lower=0.0)

    # Row-wise sum of positive signals
    row_sum = pos.sum(axis=1)

    # Overlay weights: normalize positive signals to sum to 1 where possible
    overlay = pos.div(row_sum.replace(0.0, np.nan), axis=0)

    # Mask of dates with any positive signal
    has_pos = row_sum > 0.0

    # Start from base weights
    tilted = bw.copy()

    # Blend base and overlay where we have positive signals
    idx = tilted.index[has_pos]
    tilted.loc[idx] = (1.0 - alpha) * bw.loc[idx] + alpha * overlay.loc[idx]

    # Replace remaining NaNs (e.g. all-zero rows) with 0
    tilted = tilted.fillna(0.0)

    return tilted


def apply_constraints(
    weights: pd.DataFrame,
    caps: dict[str, float] | None = None,
    cash_floor: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply portfolio constraints:

    - Per-asset caps from `caps` (in absolute value)
    - Enforce a minimum allocation to the cash sleeve (BIL / *cash*)
    - Cap gross leverage at 1.0 (sum of abs weights)
    - Top up any leftover into cash
    - Return risk flags where constraints were active

    Parameters
    ----------
    weights : DataFrame
        Raw weights (rows = dates, columns = assets).
    caps : dict[str, float], optional
        Per-asset max abs weights, e.g. {"TLT": 0.25, "TBF": 0.10}.
        If None, defaults to 0.35 for every column.
    cash_floor : float
        Minimum allocation to the cash sleeve.

    Returns
    -------
    constrained : DataFrame
        Constrained and re-normalized weights.
    risk_flags : DataFrame
        Boolean flags with columns:
        ["capped_asset", "cash_floor", "leverage_capped"].
    """
    constrained = weights.copy()
    tickers = list(constrained.columns)

    # Default caps if none provided
    if caps is None:
        caps = {t: 0.35 for t in tickers}

    # Identify cash column (prefer BIL, otherwise anything with 'cash')
    cash_col = None
    for t in tickers:
        if t.upper() == "BIL":
            cash_col = t
            break
    if cash_col is None:
        for t in tickers:
            if "cash" in t.lower():
                cash_col = t
                break

    # Risk flags table
    risk_flags = pd.DataFrame(
        index=constrained.index,
        columns=["capped_asset", "cash_floor", "leverage_capped"],
        data=False,
    )

    # 1) Per-asset caps (long/short, by abs value)
    for t in tickers:
        cap = caps.get(t, 0.35)
        before = constrained[t]
        after = before.clip(lower=-cap, upper=cap)
        capped_mask = before.ne(after)
        if capped_mask.any():
            risk_flags.loc[capped_mask, "capped_asset"] = True
        constrained[t] = after

    # 2) Enforce cash floor if we have a cash column
    if cash_col is not None:
        before = constrained[cash_col]
        after = before.clip(lower=cash_floor)
        cash_mask = before.ne(after)
        if cash_mask.any():
            risk_flags.loc[cash_mask, "cash_floor"] = True
        constrained[cash_col] = after

    # 3) Cap gross leverage (sum of |weights|) at 1.0
    gross = constrained.abs().sum(axis=1)
    leverage_mask = gross > 1.0
    scale = 1.0 / gross.clip(lower=1.0)
    constrained = constrained.mul(scale, axis=0)
    if leverage_mask.any():
        risk_flags.loc[leverage_mask, "leverage_capped"] = True

    # 4) If leverage < 1 and we have cash, put leftover into cash
    if cash_col is not None:
        row_sum = constrained.sum(axis=1)
        add_to_cash = (1.0 - row_sum).clip(lower=0.0)
        constrained[cash_col] += add_to_cash

    return constrained, risk_flags
