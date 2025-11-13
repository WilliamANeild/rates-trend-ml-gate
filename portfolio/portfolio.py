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
) -> pd.DataFrame:
    """
    Apply ML gate to base weights.

    Parameters
    ----------
    base_weights : DataFrame
        Baseline portfolio weights (risk-parity scaled from scores).
    gated_signal : DataFrame
        Score * ML multiplier, same shape as base_weights.

    Returns
    -------
    gated_weights : DataFrame
        Base weights scaled by the relative strength of the gated signal.
    """
    # Align shapes
    base_weights, gated_signal = base_weights.align(gated_signal, join="inner")

    # Use the absolute gated signal as an intensity measure per asset
    gate_abs = gated_signal.abs()

    # Normalize per row so the strongest signal in a row gets multiplier 1
    max_abs = gate_abs.max(axis=1).replace(0, np.nan)
    scale = gate_abs.div(max_abs, axis=0).fillna(0.0)

    gated_weights = base_weights * scale
    return gated_weights


def apply_constraints(
    weights: pd.DataFrame,
    caps: dict[str, float] | None = None,
    cash_floor: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply simple portfolio constraints:
    - per-asset caps from `caps` dict
    - enforce a minimum cash allocation if there is a 'BIL' or 'cash' sleeve
    - re-normalize rows to sum to 1
    - return a risk_flags DataFrame where caps were hit.

    Parameters
    ----------
    weights : DataFrame
        Raw weights (rows = dates, columns = assets).
    caps : dict[str, float], optional
        Per-asset max weights, e.g. {"TLT": 0.25, "TBF": 0.10}.
    cash_floor : float
        Minimum allocation to the cash sleeve.

    Returns
    -------
    constrained : DataFrame
        Constrained and re-normalized weights.
    risk_flags : DataFrame
        Boolean flags where raw weights wanted to exceed the caps.
    """
    constrained = weights.copy()
    risk_flags = pd.DataFrame(False, index=weights.index, columns=weights.columns)

    # Treat 'BIL' or any column containing 'cash' as cash
    cash_cols = [c for c in constrained.columns if "cash" in c.lower()]
    if "BIL" in constrained.columns and "BIL" not in cash_cols:
        cash_cols.append("BIL")
    cash_col = cash_cols[0] if cash_cols else None

    # Apply per-asset caps
    if caps is not None:
        for col, cap in caps.items():
            if col in constrained.columns:
                original = constrained[col].copy()
                constrained[col] = constrained[col].clip(upper=cap, lower=-cap)
                # Flag where we wanted to exceed the cap
                risk_flags[col] = (original.abs() > cap + 1e-9)

    # Enforce a floor on cash if a cash column exists
    if cash_col is not None:
        constrained[cash_col] = constrained[cash_col].clip(lower=cash_floor)

    # Row-wise renormalization to sum to 1
    row_sum = constrained.sum(axis=1).replace(0, np.nan)
    constrained = constrained.div(row_sum, axis=0).fillna(0.0)

    return constrained, risk_flags
