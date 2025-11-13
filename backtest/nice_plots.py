"""
Nice presentation-ready plots.

Produces:
  reports/equity_vs_bonds_20000.png

Chart:
  - $20,000 starting capital
  - Our ML-gated strategy vs buy & hold each bond ETF
  - White background, blue lines, clear labels
"""

import os

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from backtest import backtest_core


START_CAPITAL = 20_000  # starting dollars
OUT_PATH = os.path.join("reports", f"equity_vs_bonds_{START_CAPITAL}.png")


def build_equity_curves():
    """
    Run one backtest and build dollar equity curves for:
      - strategy
      - each individual ETF (buy & hold)
    """
    res = backtest_core.run_backtest()
    prices = res["prices"]
    weights = res["weights"]

    # Daily asset returns
    rets = prices.pct_change().fillna(0)

    # Strategy portfolio returns: sum over assets of w(t-1) * r(t)
    strat_rets = (weights.shift().fillna(0) * rets).sum(axis=1)

    # Dollar equity curve for strategy
    strat_eq = START_CAPITAL * (1 + strat_rets).cumprod()

    # Dollar equity curves for buy & hold each ETF
    bench_eqs = {}
    for col in prices.columns:
        bench_eqs[col] = START_CAPITAL * (1 + rets[col]).cumprod()

    return strat_eq, bench_eqs


def plot_equity_vs_bonds():
    strat_eq, bench_eqs = build_equity_curves()

    # --- Figure + styling ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Blue palette (dark = strategy, lighter = benchmarks)
    blue_colors = [
        "#0b3c5d",  # strategy (dark navy)
        "#1f77b4",
        "#4f9dda",
        "#86c5f4",
        "#a7cbe8",
        "#c2ddf2",
    ]

    # Plot strategy first
    ax.plot(
        strat_eq.index,
        strat_eq,
        label="ML-gated rates strategy",
        color=blue_colors[0],
        linewidth=2.2,
    )

    # Plot each ETF as a lighter blue line
    for i, (ticker, eq) in enumerate(bench_eqs.items(), start=1):
        color = blue_colors[i % len(blue_colors)]  # cycle if more than colors
        ax.plot(
            eq.index,
            eq,
            label=f"Buy & hold {ticker}",
            color=color,
            linewidth=1.4,
            alpha=0.9,
        )

    # --- Labels, title, formatting ---
    ax.set_title(
        f"Growth of ${START_CAPITAL:,.0f}: ML-Gated Strategy vs Bond ETFs",
        fontsize=14,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)

    # Tight layout for PPT
    fig.tight_layout()

    # Save
    os.makedirs("reports", exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200)
    plt.close(fig)

    print("Saved chart to", OUT_PATH)


if __name__ == "__main__":
    plot_equity_vs_bonds()
