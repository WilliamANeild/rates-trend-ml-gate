# backtest/plotting.py

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("default")  # white background


# Simple blue palette
COLORS = {
    "strategy": "#0b3954",  # dark blue
    "SHY": "#1f77b4",
    "IEF": "#4f9dda",
    "TLT": "#77b7ff",
    "BIL": "#a7cfff",
    "TBF": "#c8ddff",
}


def _plot_equity_vs_benchmark(strategy_equity: pd.Series,
                              benchmark_equity: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(strategy_equity.index, strategy_equity.values,
            label="ML-gated rates strategy", linewidth=2.0,
            color=COLORS["strategy"])
    ax.plot(benchmark_equity.index, benchmark_equity.values,
            label=f"Buy & hold {benchmark_equity.name}",
            linewidth=1.5, linestyle="--", color=COLORS.get("IEF", "#1f77b4"))

    ax.set_title("Growth of $20,000: Strategy vs Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)

    # Force y-axis to stay above zero
    ymin = max(0.0, strategy_equity.min() * 0.95)
    ax.set_ylim(bottom=ymin)

    fig.tight_layout()
    fig.savefig("reports/equity_vs_benchmark.png", dpi=150)
    plt.close(fig)


def _plot_drawdown(drawdown: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(drawdown.index, drawdown.values,
            linewidth=1.5, color=COLORS["strategy"])

    ax.set_title("Strategy Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (from peak, %)")
    ax.grid(True, alpha=0.25)

    # drawdown is negative; show 0 at top
    ax.set_ylim(top=0.0)

    fig.tight_layout()
    fig.savefig("reports/drawdown.png", dpi=150)
    plt.close(fig)


def _plot_rolling_vol(rolling_vol: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(rolling_vol.index, rolling_vol.values,
            linewidth=1.5, color=COLORS["strategy"])

    ax.set_title("Rolling Volatility (63-day, annualized)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig("reports/rolling_vol.png", dpi=150)
    plt.close(fig)


def _plot_equity_vs_bonds_20000(strategy_equity: pd.Series,
                                bond_equity: pd.DataFrame) -> None:
    """
    Pretty comparison: ML strategy vs individual bond ETFs,
    all starting from $20,000.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # Strategy
    ax.plot(strategy_equity.index, strategy_equity.values,
            label="ML-gated rates strategy",
            linewidth=2.5, color=COLORS["strategy"])

    # Bonds (different blue shades)
    for col in bond_equity.columns:
        ax.plot(
            bond_equity.index,
            bond_equity[col].values,
            label=f"Buy & hold {col}",
            linewidth=1.2,
            color=COLORS.get(col, "#9ecae1"),
            alpha=0.9,
        )

    ax.set_title("Growth of $20,000: ML-Gated Strategy vs Bond ETFs")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)

    # Force everything to stay above zero on the chart
    all_vals = pd.concat([strategy_equity, bond_equity], axis=1)
    ymin = max(0.0, all_vals.min().min() * 0.95)
    ax.set_ylim(bottom=ymin)

    fig.tight_layout()
    fig.savefig("reports/equity_vs_bonds_20000.png", dpi=150)
    plt.close(fig)


def make_charts(metrics_dict: dict) -> None:
    """
    Entry point called from backtest.run

    metrics_dict must contain:
      - 'strategy_equity' (Series)
      - 'benchmark_equity' (Series)
      - 'bond_equity' (DataFrame)
      - 'drawdown' (Series)
      - 'rolling_vol' (Series)
    """
    strat_eq: pd.Series = metrics_dict["strategy_equity"]
    bench_eq: pd.Series = metrics_dict["benchmark_equity"]
    bond_eq: pd.DataFrame = metrics_dict["bond_equity"]
    drawdown: pd.Series = metrics_dict["drawdown"]
    rolling_vol: pd.Series = metrics_dict["rolling_vol"]

    _plot_equity_vs_benchmark(strat_eq, bench_eq)
    _plot_drawdown(drawdown)
    _plot_rolling_vol(rolling_vol)
    _plot_equity_vs_bonds_20000(strat_eq, bond_eq)

    print("[plotting] charts saved to reports")
