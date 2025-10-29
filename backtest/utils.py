"""
utils.py — small helpers: dirs, scheduling, saving.
"""
import os
import pandas as pd

def init_dirs():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

def date_range_split(start, end, step_months=6):
    from pandas.tseries.offsets import MonthEnd
    dates = pd.date_range(start, end, freq=f"{step_months}M")
    return list(zip(dates[:-1], dates[1:]))

def save_outputs(metrics_dict):
    for name, obj in metrics_dict.items():
        if hasattr(obj, "to_csv"):
            obj.to_csv(f"outputs/{name}.csv")
    print("[utils] saved CSV outputs")
