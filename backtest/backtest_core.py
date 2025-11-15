"""
Core backtest runner.

Wires together:
  data loaders  -> signals -> ML gate -> portfolio construction -> outputs

Used by backtest/run.py via wf_runner.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import data.loaders as loaders
import features.signals as signals
import models.ml_gate as ml_gate
import portfolio.portfolio as pf


def run_backtest():
    print("[backtest] start")

    # --- Load data ---
    prices = loaders.load_prices()          # ETF prices (SHY, IEF, TLT, BIL, TBF)
    yields = loaders.load_yields()          # Treasury curve if available
    tickers = prices.columns.tolist()

    # --- Feature generation: momentum + carry + blended score ---
    momentum = signals.momentum_features(prices)
    carry = signals.carry_proxy(yields, tickers)
    score = signals.pre_gate_score(momentum, carry)

    # --- Label generation + ML gate training (currently used only for diagnostics) ---
    labels = ml_gate.make_labels(
        prices_df=prices,
        momentum_df=momentum,
        horizon_days=21,      # roughly 1 trading month
        hurdle_rule=0.01,     # ignore very weak momentum
    )

    # Cross sectional feature: just use the blended score for now
    features_df = score.stack().to_frame(name="score")

    # Align labels to features and drop missing
    labels_series = labels.stack().reindex(features_df.index).dropna()
    features_df = features_df.loc[labels_series.index]

    # Fit ML gate (probabilities are not yet used for sizing)
    model_dict = ml_gate.fit_gate(features_df, labels_series)

    # For now, gating is effectively disabled inside prob_to_multiplier,
    # but we keep the code path so it is easy to turn back on.
    prob_series = ml_gate.predict_gate(model_dict, features_df)
    multipliers = ml_gate.prob_to_multiplier(prob_series)

    # Per asset time gated score (right now equals blended score)
    gated_signal = (features_df["score"] * multipliers).unstack()

    # --- Portfolio construction ---

    # 1) Estimate vol and build risk parity base weights from the raw blended score
    vol_df = pf.estimate_vol(prices.pct_change(), window=63)
    base_weights = pf.risk_parity_scale(score, vol_df)

    # 2) Apply cross sectional gate to tilt toward stronger signals
    gated_weights = pf.apply_gate(base_weights, gated_signal)

    # 3) Regime switch: less paranoid, more return seeking
    regime_weights = gated_weights.copy()
    avg_score = score.mean(axis=1)

    # Looser threshold: only go defensive when average score is clearly bad
    risk_on_mask = avg_score > -0.10
    risk_off_mask = ~risk_on_mask

    # Identify cash column
    if "BIL" in regime_weights.columns:
        cash_col = "BIL"
    else:
        cash_col = regime_weights.columns[0]

    risk_off_idx = regime_weights.index[risk_off_mask]

    if len(risk_off_idx) > 0:
        # Defensive target:
        #   50 percent IEF (belly)
        #   30 percent SHY (front end)
        #   20 percent BIL (cash)
        regime_weights.loc[risk_off_idx] = 0.0

        if "IEF" in regime_weights.columns:
            regime_weights.loc[risk_off_idx, "IEF"] = 0.50
        elif tickers:
            regime_weights.loc[risk_off_idx, tickers[0]] = 0.50

        if "SHY" in regime_weights.columns:
            regime_weights.loc[risk_off_idx, "SHY"] = 0.30
        elif len(tickers) > 1:
            regime_weights.loc[risk_off_idx, tickers[1]] = 0.30

        regime_weights.loc[risk_off_idx, cash_col] = 0.20

    # 4) Hard constraints and cash floor
    caps = {
        "TLT": 0.50,
        "IEF": 0.50,
        "SHY": 0.40,
        "BIL": 0.50,
        "TBF": 0.15,
    }

    weights_df, risk_flags_df = pf.apply_constraints(
        regime_weights,
        caps=caps,
        cash_floor=0.01,
    )

    print("[backtest] done")

    return {
        "prices": prices,
        "weights": weights_df,
        "risk_flags": risk_flags_df,
        "gated_signal": gated_signal,
        "model": model_dict,
    }


def run_recommend():
    """
    Single period recommendation run.
    Same regime logic as run_backtest, evaluated on the latest date.
    """
    print("[recommend] run")

    # --- Load data ---
    prices = loaders.load_prices()
    yields = loaders.load_yields()
    tickers = prices.columns.tolist()

    # --- Signals ---
    momentum = signals.momentum_features(prices)
    carry = signals.carry_proxy(yields, tickers)
    score = signals.pre_gate_score(momentum, carry)

    latest_date = score.index[-1]
    latest_scores = score.loc[[latest_date]]

    # --- ML gate (same diagnostic training pattern) ---
    labels = ml_gate.make_labels(
        prices_df=prices,
        momentum_df=momentum,
        horizon_days=21,
        hurdle_rule=0.01,
    )
    features_df = latest_scores.stack().to_frame(name="score")
    labels_series = labels.stack().reindex(features_df.index).dropna()

    if labels_series.empty:
        print("[recommend] no matching labels for latest date, aborting")
        return

    features_df = features_df.loc[labels_series.index]
    model_dict = ml_gate.fit_gate(features_df, labels_series)
    prob_series = ml_gate.predict_gate(model_dict, features_df)
    multipliers = ml_gate.prob_to_multiplier(prob_series)
    gated_signal = (features_df["score"] * multipliers).unstack()

    # --- Portfolio construction as in run_backtest ---
    vol_df = pf.estimate_vol(prices.pct_change(), window=63)
    base_weights = pf.risk_parity_scale(score, vol_df)
    gated_weights = pf.apply_gate(base_weights, gated_signal)

    regime_weights = gated_weights.copy()
    avg_score = score.mean(axis=1)
    risk_on_mask = avg_score > -0.10
    risk_off_mask = ~risk_on_mask

    if "BIL" in regime_weights.columns:
        cash_col = "BIL"
    else:
        cash_col = regime_weights.columns[0]

    risk_off_idx = regime_weights.index[risk_off_mask]

    if len(risk_off_idx) > 0:
        regime_weights.loc[risk_off_idx] = 0.0

        if "IEF" in regime_weights.columns:
            regime_weights.loc[risk_off_idx, "IEF"] = 0.50
        elif tickers:
            regime_weights.loc[risk_off_idx, tickers[0]] = 0.50

        if "SHY" in regime_weights.columns:
            regime_weights.loc[risk_off_idx, "SHY"] = 0.30
        elif len(tickers) > 1:
            regime_weights.loc[risk_off_idx, tickers[1]] = 0.30

        regime_weights.loc[risk_off_idx, cash_col] = 0.20

    caps = {
        "TLT": 0.50,
        "IEF": 0.50,
        "SHY": 0.40,
        "BIL": 0.50,
        "TBF": 0.15,
    }

    weights_df, risk_flags_df = pf.apply_constraints(
        regime_weights,
        caps=caps,
        cash_floor=0.01,
    )

    latest_weights = weights_df.loc[latest_date]
    print("\n[recommend] Recommended weights on", latest_date.date(), ":")
    print(latest_weights.round(4))
    print("[recommend] end")


if __name__ == "__main__":
    run_backtest()
