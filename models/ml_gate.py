import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


def make_labels(prices_df, momentum_df, horizon_days, hurdle_rule=0):
    """
    Create binary labels by checking whether momentum direction matches
    future price direction over a given horizon.

    labels[i, t] = 1 if sign(momentum[i, t]) == sign(future_return[i, t]), else 0.
    If hurdle_rule > 0, we drop weak momentum observations instead of forcing them to 0.
    """
    # Future returns over the specified horizon
    future_returns = prices_df.pct_change(horizon_days).shift(-horizon_days)

    # Align on same dates and tickers
    future_returns, momentum_df = future_returns.align(momentum_df, join="inner")

    # Directional signs
    future_signs = np.sign(future_returns)
    momentum_signs = np.sign(momentum_df)

    # 1 if signs match, 0 otherwise
    labels = (future_signs == momentum_signs).astype(float)  # 1.0 or 0.0

    if hurdle_rule > 0:
        # Keep only strong momentum; weak signals become NaN (ignored in training)
        strong_mask = momentum_df.abs() > hurdle_rule
        labels = labels.where(strong_mask)

    # Drop any rows that are all NaN (for example at the end of series)
    labels = labels.dropna(how="all")

    return labels


def fit_gate(features_df, labels):
    """
    Train an ML gate that predicts when momentum is likely to be right.

    features_df: DataFrame of features (index = dates, columns = feature names).
    labels: DataFrame or Series from make_labels. If DataFrame (assets x time),
            we collapse cross section into a single label per date.
    """
    # If labels is a DataFrame (assets x time), flatten to a 1D Series by averaging
    if isinstance(labels, pd.DataFrame):
        # Average across assets, then treat > 0.5 as "momentum works" (1), else 0
        y_series = (labels.mean(axis=1) > 0.5).astype(int)
    else:
        y_series = labels.astype(int)

    # Align features and labels on the time index
    features_df, y_series = features_df.align(y_series, join="inner", axis=0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    # Logistic regression with elastic net, wrapped in a calibrated classifier
    base_model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        max_iter=1000,
    )
    calibrated_model = CalibratedClassifierCV(base_model, cv=5)
    calibrated_model.fit(X_scaled, y_series)

    return {"model": calibrated_model, "scaler": scaler}


def predict_gate(model_dict, features_df):
    """
    Use the trained gate to get probabilities that momentum is correct.

    Returns a Series indexed by date with values in [0, 1].
    """
    # Align features to scaler feature order if needed (assumes same columns)
    X_scaled = model_dict["scaler"].transform(features_df)
    probabilities = model_dict["model"].predict_proba(X_scaled)[:, 1]
    return pd.Series(probabilities, index=features_df.index)

def prob_to_multiplier(prob_series):
    """
    TEMP: disable ML gating.
    Always use full position (multiplier = 1.0).
    """
    return pd.Series(1.0, index=prob_series.index)
'''
def prob_to_multiplier(prob_series, floor=0.5, cap=1.5):
    """
    Map probabilities to position size multipliers around 1.0.

    p = 0.00 -> floor  (for example 0.5x)
    p = 0.50 -> 1.0x   (neutral)
    p = 1.00 -> cap    (for example 1.5x)

    This lets the gate both cut exposure when it hates momentum
    and increase exposure when it is confident.
    """
    clipped = np.clip(prob_series, 0.0, 1.0)

    # Linear mapping: 0 -> floor, 0.5 -> 1, 1 -> cap
    multipliers = floor + (cap - floor) * (clipped - 0.5) / 0.5
    multipliers = np.clip(multipliers, floor, cap)

    return pd.Series(multipliers, index=prob_series.index)
'''
# Integration Flow:
# 1. Momentum code generates signals and features.
# 2. make_labels() builds training labels off prices and momentum.
# 3. fit_gate() trains the calibrated logistic model.
# 4. predict_gate() returns probabilities that momentum is right.
# 5. prob_to_multiplier() converts those probabilities into size multipliers.
# 6. Final positions = base momentum weights * multipliers.
