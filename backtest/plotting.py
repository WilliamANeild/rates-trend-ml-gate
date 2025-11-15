"""
Plotting utilities for the rates ML-gated backtest.

Builds equity curves for:
  - ML-gated strategy
  - Buy & hold IEF, TLT, BIL

Window: 2020-01-01 through end of data.
"""

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data.loaders as loaders


def _equity_curve(returns: pd.Series, start_value: float = 20_000.0) -> pd.Series:
    """Convert daily returns into a dollar equity curve."""
    return start_value * (1.0 + returns).cumprod()


def make_charts(metrics_dict: Dict, out_dir: str = "reports") -> None:
    """
    Entry point used by backtest/run.py.

    We ignore metrics_dict and rebuild what we need from:
      - outputs/strategy_returns.csv
      - prices from data.loaders.load_prices()
    """
    os.makedirs(out_dir, exist_ok=True)

    # Strategy returns from the backtest outputs
    strat_rets = (
        pd.read_csv("outputs/strategy_returns.csv", index_col=0, parse_dates=True)
        .iloc[:, 0]
        .sort_index()
    )

    # ETF prices, then simple daily returns
    prices = loaders.load_prices().sort_index()  # columns: SHY, IEF, TLT, BIL, TBF ...
    etf_rets = prices.pct_change().fillna(0.0)

    fig_path = os.path.join(out_dir, "equity_vs_bond_etfs_2020_2025.png")

    plot_equity_vs_bonds(
        strat_rets=strat_rets,
        etf_rets=etf_rets,
        start_date="2020-01-01",
        out_path=fig_path,
    )


def plot_equity_vs_bonds(
    strat_rets: pd.Series,
    etf_rets: pd.DataFrame,
    start_date: str = "2020-01-01",
    out_path: str | None = None,
) -> None:
    """
    Plot strategy equity vs selected bond ETFs from a given start date onward.

    Shown:
      - Strategy
      - IEF, TLT, BIL

    Hidden:
      - SHY, TBF (even if present in etf_rets)
    """
    # Align on common index and cut to desired window
    all_rets = pd.concat(
        [strat_rets.rename("STRAT"), etf_rets],
        axis=1,
        join="inner",
    ).sort_index()

    all_rets = all_rets.loc[start_date:]

    strat = all_rets["STRAT"]

    # Only keep IEF, TLT, BIL if they exist
    keep_cols = [c for c in ["IEF", "TLT", "SHY"] if c in all_rets.columns]
    bonds = all_rets[keep_cols]

    # Equity curves
    eq_strat = _equity_curve(strat)
    eq_bonds = {c: _equity_curve(bonds[c]) for c in keep_cols}

    plt.figure(figsize=(14, 4))

    # Strategy in black
    plt.plot(
        eq_strat.index,
        eq_strat.values,
        color="black",
        linewidth=2.0,
        label="ML-gated rates strategy",
    )

    # ETF colors, all solid
    color_map = {
        "IEF": "tab:blue",
        "TLT": "purple",
        "BIL": "gray",
    }

    for c in keep_cols:
        plt.plot(
            eq_bonds[c].index,
            eq_bonds[c].values,
            color=color_map.get(c, "dimgray"),
            linewidth=1.5,
            label=f"Buy & hold {c}",
        )

    plt.title("Growth of $20,000 (2020-2025): Strategy vs Bond ETFs")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.legend(loc="upper left")
    plt.grid(False)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path, dpi=200)
    plt.close()
