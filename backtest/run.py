"""
run.py — main backtest pipeline runner.
"""
from backtest import backtest_core, wf_runner, metrics, plotting, ablations, utils

def main():
    utils.init_dirs()
    print("=== Pipeline start ===")

    wf_results = wf_runner.run_walkforward(backtest_core.run_backtest)
    # For simplicity, just use the last segment for now
    _, _, latest = wf_results[-1]
    metrics_dict = metrics.compute_all(latest["prices"], latest["weights"])
    plotting.make_charts(metrics_dict)
    ablations.run_all(wf_results)
    utils.save_outputs(metrics_dict)

    print("=== Pipeline done ===")

if __name__ == "__main__":
    main()