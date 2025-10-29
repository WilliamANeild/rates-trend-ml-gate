"""
metrics.py — compute portfolio metrics: equity, drawdown, rolling stats.
"""
import pandas as pd
import numpy as np

def equity_curve(weights_df, prices_df):
    returns = prices_df.pct_change().fillna(0)
    port_ret = (weights_df.shift() * returns).sum(axis=1)
    return (1 + port_ret).cumprod().rename("equity")

def drawdown(equity_curve):
    peak = equity_curve.cummax()
    return (equity_curve / peak - 1).rename("drawdown")

def rolling_stats(equity_curve, window=63):
    ret = equity_curve.pct_change().dropna()
    vol = ret.rolling(window).std() * np.sqrt(252)
    sharpe = ret.rolling(window).mean() / (ret.rolling(window).std() + 1e-9)
    return pd.DataFrame({"rolling_vol": vol, "rolling_sharpe": sharpe})

def compute_all(prices_df, weights_df):
    print("[metrics] computing metrics...")
    eq = equity_curve(weights_df, prices_df)
    dd = drawdown(eq)
    roll = rolling_stats(eq)
    return {"equity": eq, "drawdown": dd, "rolling": roll}
