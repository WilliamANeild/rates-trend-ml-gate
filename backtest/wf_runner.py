"""
wf_runner.py — run backtests across walk-forward splits.
"""
from backtest.utils import date_range_split

def run_walkforward(backtest_func, start="2010-01-01", end="2025-01-01", step_months=12):
    print("[wf_runner] Starting walk-forward run...")
    schedule = date_range_split(start, end, step_months)
    results = []
    for (train_start, train_end) in schedule:
        print(f"  → Running segment {train_start} → {train_end}")
        res = backtest_func()
        results.append((train_start, train_end, res))
    print("[wf_runner] Finished all splits")
    return results
