# rates-trend-ml-gate
Advisory long only Treasury ETF allocator with momentum, carry, and a logistic ML gate.

Carol. Folder: reports/
Goal: turn model results into a clear weekly note and tables.
Inputs: weights_df (indexed by date with columns for each ETF and cash), rationale_dict (per date short strings), risk_flags_df (booleans like cash_floor_hit, dd_stop_hit), charts (paths if available).
Outputs: model_weights.csv, trade_tickets.csv (from current to target weights given a prior weights vector), recommendation.md (one short paragraph plus a table), saved images if provided.
Functions to build:

build_trade_tickets(current_weights, target_weights) -> DataFrame

write_weights(weights_df, path) -> None

write_recommendation(weights_df, trade_df, rationale_dict, risk_flags_df, out_dir) -> None
Assumptions: weights sum to 1. tickers include SHY, IEF, TLT, BIL, optional TBF. prior weights for trade tickets can be yesterday or last rebalance.

Nicolas. Folder: data/
Goal: load clean daily prices for ETFs and basic Treasury yields aligned on business days.
Inputs: list of ETF tickers, list of FRED series ids for 2y, 5y, 10y, 30y, calendar of FOMC dates.
Outputs: prices_df (Adj Close level, wide format), yields_df (2y, 5y, 10y, 30y columns), calendar_flags_df (columns like is_fomc_week, is_blackout). All indexed by date with no missing days after forward fill rules.
Functions to build:

load_prices(tickers) -> DataFrame

load_yields(series_ids) -> DataFrame

build_calendar_flags(fomc_dates) -> DataFrame
Assumptions: daily frequency. forward fill small gaps. drop dates where nothing trades.

Liam. Folder: features/
Goal: create momentum features, a carry proxy per sleeve, and a pre gate score.
Inputs: prices_df from data, yields_df from data, list of tickers.
Outputs: momentum_df (multi window standardized momentum), carry_df (per ticker carry proxy), pre_gate_scores_df (per ticker blended score).
Functions to build:

build_momentum(prices_df, windows) -> DataFrame

build_carry_proxy(yields_df, tickers) -> DataFrame

blend_scores(momentum_df, carry_df, weights_tuple) -> DataFrame
Assumptions: all features are lagged one day relative to decisions. standardize features by rolling mean and std. no look ahead.

Lia and Carol. Folder: models/
Goal: train a small logistic ML gate and convert probability to a 0 to 1 exposure dial.
Inputs: feature matrix built from Liam’s outputs plus simple state features like realized volatility and breadth, labels built from future returns sign vs momentum sign, optional calendar flags.
Outputs: gate_prob_df (probability per date), gate_mult_df (multiplier per date), serialized model file if needed.
Functions to build:

make_labels(prices_df, momentum_df, horizon_days, hurdle_rule) -> Series

fit_gate(features_df, labels) -> ModelObject

predict_gate(model, features_df) -> Series

prob_to_multiplier(prob_series) -> Series
Assumptions: use logistic regression with elastic net. calibrate probabilities. clip and lag inputs. multiplier must be in [0,1].

Isaac. Folder: portfolio/
Goal: convert scores and the gate into target weights with risk parity, caps, and a cash sink.
Inputs: pre_gate_scores_df, gate_mult_df, prices_df for volatility estimates, config with caps and floors.
Outputs: weights_df (target weights per date, columns per ticker plus cash). risk_flags_df (booleans such as hit_cap, cash_floor_hit, dd_stop_hit).
Functions to build:

estimate_vol(prices_df, window) -> Series or DataFrame

risk_parity_scale(scores_df, vol_df) -> DataFrame

apply_gate(weights_df, gate_mult_df) -> DataFrame

apply_constraints(weights_df, caps_dict, cash_floor) -> (DataFrame, DataFrame_flags)
Assumptions: weights sum to 1. any leftover goes to cash. TBF has a smaller cap. turnover control will be handled here using a threshold.

Liam and Emily. Folder: backtest/
Goal: wire the pipeline end to end and produce performance artifacts and daily or weekly recommendations.
Inputs: everything upstream. a simple cost model. a schedule for walk forward splits.
Outputs: equity_curve.csv, drawdown.csv, rolling_stats.csv, ablations.csv, saved charts, plus the same outputs that Reporting uses for a current run.