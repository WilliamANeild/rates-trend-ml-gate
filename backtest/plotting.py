"""
plotting.py — visualize and save charts.
"""
import matplotlib.pyplot as plt
import os

def make_charts(metrics_dict, outdir="reports"):
    os.makedirs(outdir, exist_ok=True)

    metrics_dict["equity"].plot(title="Equity Curve")
    plt.savefig(f"{outdir}/equity_curve.png", bbox_inches="tight")

    metrics_dict["drawdown"].plot(title="Drawdown")
    plt.savefig(f"{outdir}/drawdown.png", bbox_inches="tight")

    metrics_dict["rolling"]["rolling_vol"].plot(title="Rolling Volatility")
    plt.savefig(f"{outdir}/rolling_vol.png", bbox_inches="tight")

    print(f"[plotting] charts saved to {outdir}")
