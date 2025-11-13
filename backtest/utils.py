"""
utils.py – small helpers: dirs, scheduling, saving.
"""
import os
import pandas as pd


def init_dirs():
    """Create reports/ and outputs/ directories if they do not exist."""
    os.makedirs("reports", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


def date_range_split(start, end, step_months: int = 6):
    """
    Split [start, end] into month end ranges of length `step_months`.

    Returns
    -------
    list of (start_date, end_date) tuples.
    """
    # Use MonthEnd explicitly to avoid the 'M' deprecation warning.
    from pandas.tseries.offsets import MonthEnd

    dates = pd.date_range(start, end, freq=MonthEnd(step_months))
    return list(zip(dates[:-1], dates[1:]))


def save_outputs(wf_results, metrics_dict=None):
    """
    Save key backtest outputs and metrics to CSV.

    Parameters
    ----------
    wf_results : list[dict] or dict
        Walk forward results from wf_runner.run_walkforward.
        Each dict is expected to contain DataFrames like 'prices', 'weights',
        'gated_signal', 'risk_flags', etc.
    metrics_dict : dict, optional
        Metrics returned by metrics.compute_metrics, e.g. {'equity_curve': df}.
    """
    init_dirs()

    # 1) Save metrics if provided
    if metrics_dict is not None:
        for name, obj in metrics_dict.items():
            if hasattr(obj, "to_csv"):
                out_path = os.path.join("outputs", f"{name}.csv")
                obj.to_csv(out_path)
        print("[utils] saved metrics CSVs")

    # Helper to vertically stack a given key across walk forward segments
    def _stack(key: str):
        frames = []

        if isinstance(wf_results, list):
            # List of segment dicts
            for seg in wf_results:
                if isinstance(seg, dict) and key in seg:
                    val = seg[key]
                    if isinstance(val, (pd.DataFrame, pd.Series)):
                        frames.append(val)
        elif isinstance(wf_results, dict):
            # Single dict
            val = wf_results.get(key)
            if isinstance(val, (pd.DataFrame, pd.Series)):
                frames.append(val)

        if not frames:
            return None

        combined = pd.concat(frames).sort_index()
        return combined

    # 2) Save combined time series for core objects
    for key in ["prices", "weights", "gated_signal", "risk_flags"]:
        combined = _stack(key)
        if combined is not None:
            out_path = os.path.join("outputs", f"{key}_full.csv")
            combined.to_csv(out_path)

    print("[utils] saved CSV outputs")
