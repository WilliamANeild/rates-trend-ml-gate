"""
Backtest runner (stub).
Input:
  - will call data loaders and signals later
Output:
  - prints placeholders so the pipeline shape is clear
Goal:
  Have a single entry point to wire data -> signals -> gate -> allocate -> report.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import data.loaders as loaders
import features.signals as signals
import models.ml_gate as ml_gate
import portfolio.portfolio as pf  


def run_backtest():
  print("[backtest] start")

  # --- Load data ---
  prices = loaders.load_prices()
  yields = loaders.load_yields()
  tickers = prices.columns.tolist()

  # --- Feature generation ---
  momentum = signals.momentum_features(prices)
  carry = signals.carry_proxy(yields, tickers)
  score = signals.pre_gate_score(momentum, carry)

  # --- Label generation + model training ---
  labels = ml_gate.make_labels(prices, momentum, horizon_days=21, hurdle_rule=0.01)
  features_df = score.stack().to_frame(name="score")
  labels_series = labels.stack().reindex(features_df.index).dropna()
  features_df = features_df.loc[labels_series.index]
  model_dict = ml_gate.fit_gate(features_df, labels_series)

  # --- Predictions and gating ---
  prob_series = ml_gate.predict_gate(model_dict, features_df)
  multipliers = ml_gate.prob_to_multiplier(prob_series)
  gated_signal = (features_df["score"] * multipliers).unstack()

  # --- Portfolio construction ---
  vol_df = pf.estimate_vol(prices, window=60)
  base_weights = pf.risk_parity_scale(score, vol_df)
  gated_weights = pf.apply_gate(base_weights, gated_signal)
  caps = {"TLT": 0.25, "IEF": 0.25, "SHY": 0.25, "BIL": 0.25, "TBF": 0.1}
  weights_df, risk_flags_df = pf.apply_constraints(gated_weights, caps, cash_floor=0.05)

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
  Single-period recommendation run.

  Loads latest data, computes signals, gates them, applies constraints,
  and outputs a single-row recommendation of target weights with rationale.
  """
  print("[recommend] run")

  # --- Load data ---
  etf_basket_prices = loaders.load_prices()
  yields = loaders.load_yields()

  # --- Compute latest signals ---
  momentum_df = signals.momentum_features(etf_basket_prices)
  carry_df = signals.carry_proxy(yields, etf_basket_prices.columns.tolist())
  score_df = signals.pre_gate_score(momentum_df, carry_df)

  # --- Latest date only ---
  latest_date = score_df.index[-1]
  latest_scores = score_df.loc[[latest_date]]

  # --- Run ML gate on latest features ---
  labels_df = ml_gate.make_labels(etf_basket_prices, momentum_df, horizon_days=21, hurdle_rule=0.01)
  features_df = latest_scores.stack().to_frame(name="score")
  labels_series = labels_df.stack().reindex(features_df.index).dropna()
  if labels_series.empty:
      print("[warn] no matching labels found for latest period")
      return

  features_df = features_df.loc[labels_series.index]
  model_dict = ml_gate.fit_gate(features_df, labels_series)
  prob_series = ml_gate.predict_gate(model_dict, features_df)
  multipliers = ml_gate.prob_to_multiplier(prob_series)
  gated_signal = features_df["score"] * multipliers
  gated_signal = gated_signal.unstack()

  # --- Portfolio construction ---
  import portfolio.portfolio as pf
  vol_df = pf.estimate_vol(etf_basket_prices, window=60)
  base_weights = pf.risk_parity_scale(score_df, vol_df)
  gated_weights = pf.apply_gate(base_weights, gated_signal)

  caps = {"TLT": 0.25, "IEF": 0.25, "SHY": 0.25, "BIL": 0.25, "TBF": 0.1}
  weights_df, risk_flags_df = pf.apply_constraints(gated_weights, caps, cash_floor=0.05)

  # --- Extract latest recommendations ---
  latest_weights = weights_df.loc[latest_date]
  print("\n[recommend] Recommended Weights on", latest_date.date(), ":")
  print(latest_weights.round(4))

  # --- Optional: rationale or commentary ---
  top_assets = latest_scores.T.sort_values(latest_date, ascending=False).head(3).index.tolist()
  print("\n[recommend] Top drivers:", ", ".join(top_assets))
  print("[recommend] end")

if __name__ == "__main__":
    run_backtest()