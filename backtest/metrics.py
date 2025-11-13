# backtest/metrics.py

import numpy as np
import pandas as pd


def _combine_walkforward_results(wf_results):
    """
    Take the walk-forward results (list of segments) and stitch them
    into a single dict of DataFrames/Series, sorted by date index.

    Expected shapes:
      - dict with keys like "prices", "weights", ...
      - OR list of (start, end, result_dict)
      - OR list of result_dict
    """
    # If it's already a dict, assume it's the combined result
    if isinstance(wf_results, dict):
        return wf_results

    if not isinstance(wf_results, (list, tuple)) or len(wf_results) == 0:
        raise TypeError(f"wf_results has unexpected type: {type(wf_results)}")

    combined: dict[str, list[pd.Series | pd.DataFrame]] = {}

    for seg in wf_results:
        # walk-forward runner usually gives (start, end, result_dict)
        if isinstance(seg, tuple) and len(seg) == 3:
            _, _, res = seg
        elif isinstance(seg, dict):
            res = seg
        else:
            continue

        for k, v in res.items():
            if isinstance(v, (pd.Series, pd.DataFrame)):
                combined.setdefault(k, []).append(v)

    out: dict[str, pd.Series | pd.DataFrame] = {}
    for k, pieces in combined.items():
        cat = pd.concat(pieces).sort_index()
        # IMPORTANT: drop duplicate dates (keep first occurrence)
        cat = cat[~cat.index.duplicated(keep="first")]
        out[k] = cat

    return out


def _to_equity(returns: pd.Series, initial_capital: float = 20_000.0) -> pd.Series:
    """
    Turn daily percentage returns into an equity curve starting at initial_capital.
    We hard-clip at -99% so the curve never crosses zero and goes negative.
    """
    clipped = returns.clip(lower=-0.99)
    equity = (1.0 + clipped).cumprod() * initial_capital
    return equity


def compute_metrics(wf_results):
    """
    Main entry point: take walk-forward results, build daily returns,
    equity curves, drawdown, and rolling vol, and return a metrics dict.

    The returned dict is what utils.save_outputs() and plotting.py use.
    """
    print("[metrics] computing metrics...")

    combined = _combine_walkforward_results(wf_results)

    prices: pd.DataFrame = combined["prices"].sort_index()
    weights: pd.DataFrame = combined["weights"].sort_index()

    # Drop any duplicate dates again, just to be safe
    prices = prices[~prices.index.duplicated(keep="first")]
    weights = weights.reindex(prices.index)

    # Fill missing weights with zero exposure
    weights = weights.fillna(0.0)

    # --- Daily asset returns (percentage) ---
    rets = prices.pct_change().fillna(0.0)

    # --- Strategy daily return: w_{t-1} * r_t ---
    strat_rets = (weights.shift().fillna(0.0) * rets).sum(axis=1)

    # Benchmark = simple buy & hold IEF (if present), otherwise first column
    benchmark_ticker = "IEF" if "IEF" in prices.columns else prices.columns[0]
    bench_rets = rets[benchmark_ticker]

    # Align strategy and benchmark, and remove any duplicate dates
    strat_rets, bench_rets = strat_rets.align(bench_rets, join="inner")
    mask = ~strat_rets.index.duplicated(keep="first")
    strat_rets = strat_rets[mask]
    bench_rets = bench_rets.reindex(strat_rets.index)

    # Clip both so we never have a day worse than -99%
    strat_rets = strat_rets.clip(lower=-0.99)
    bench_rets = bench_rets.clip(lower=-0.99)

    # --- Equity curves from $20,000 ---
    initial_capital = 20_000.0
    strategy_equity = _to_equity(strat_rets, initial_capital)
    benchmark_equity = _to_equity(bench_rets, initial_capital)

    # --- Buy & hold bond ETFs from $20,000 ---
    bond_names = ["SHY", "IEF", "TLT", "BIL", "TBF"]
    bond_cols = [c for c in prices.columns if c in bond_names]

    bond_equity_dict: dict[str, pd.Series] = {}
    for t in bond_cols:
        r = rets[t].clip(lower=-0.99)
        # Align to strategy index so plots line up
        r = r.reindex(strat_rets.index).fillna(0.0)
        bond_equity_dict[t] = _to_equity(r, initial_capital)

    bond_equity = pd.DataFrame(bond_equity_dict)

    # --- Rolling vol of strategy (63d, annualized) ---
    rolling_vol = strat_rets.rolling(63).std() * np.sqrt(252)
    rolling = pd.DataFrame(
        {"strategy": strat_rets, "benchmark": bench_rets},
        index=strat_rets.index,
    )

    # --- Drawdown of strategy (in % from peak) ---
    equity_1 = _to_equity(strat_rets, initial_capital=1.0)
    peak = equity_1.cummax()
    drawdown = equity_1 / peak - 1.0

    metrics_dict = {
        "strategy_returns": strat_rets,
        "benchmark_returns": bench_rets,
        "strategy_equity": strategy_equity,
        "benchmark_equity": benchmark_equity,
        "bond_equity": bond_equity,
        "rolling": rolling,
        "rolling_vol": rolling_vol,
        "drawdown": drawdown,
    }

    return metrics_dict
