"""
Top-level backtest runner.

Wires:
  - walk-forward engine
  - single full-sample backtest for metrics
  - plotting / ablations / CSV outputs
"""

from backtest import backtest_core, wf_runner, metrics, plotting, ablations, utils


def main():
    print("=== Pipeline start ===")

    # 1) Walk-forward run (for later ablations / diagnostics)
    wf_results = wf_runner.run_walkforward(backtest_core.run_backtest)

    # 2) Full-sample backtest for clean performance charts
    print("[metrics] computing metrics...")
    metrics_dict = metrics.compute_metrics(wf_results)

    # 3) Pretty charts (equity vs benchmark, drawdown, rolling vol)
    plotting.make_charts(metrics_dict)

    # 4) Ablations / CSV outputs (keep hooks alive)
    ablations.run_ablations(wf_results, metrics_dict)
    utils.save_outputs(wf_results, metrics_dict)

    print("=== Pipeline done ===")


if __name__ == "__main__":
    main()
