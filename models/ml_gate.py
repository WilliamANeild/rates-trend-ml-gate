import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


def make_labels(prices_df, momentum_df, horizon_days, hurdle_rule=0):
    """
    Creates binary classification labels (0 or 1) by comparing future price movements
    with momentum signals.

    Inputs
    -------
    prices_df : DataFrame
        Asset prices (index = dates, columns = assets).
    momentum_df : DataFrame
        Momentum indicators from Liam's code (same structure).
    horizon_days : int
        Number of days to look ahead for returns.
    hurdle_rule : float
        Minimum absolute momentum value to consider (filters out weak signals).

    Output
    -------
    labels : DataFrame
        Binary labels where 1 indicates momentum correctly predicts future direction.
    """
    # Future returns over the specified horizon
    future_returns = prices_df.pct_change(horizon_days).shift(-horizon_days)

    # Align future_returns and momentum_df on the same index/columns
    future_returns, momentum_df = future_returns.align(momentum_df, join="inner")

    # Directional signs
    future_signs = np.sign(future_returns)
    momentum_signs = np.sign(momentum_df)

    # Label = 1 when sign matches, 0 otherwise
    labels = (future_signs == momentum_signs).astype(int)

    if hurdle_rule > 0:
        # Filter out weak signals below the hurdle threshold
        strong_mask = (momentum_df.abs() > hurdle_rule).astype(int)
        labels = labels * strong_mask

    # Drop any rows that are all NaN (e.g. at the very end of the series)
    labels = labels.dropna(how="all")

    return labels


def fit_gate(features_df, labels):
    """
    Trains a machine learning model to predict the reliability of momentum signals.

    Inputs
    -------
    features_df : DataFrame
        Features derived from Liam's momentum indicators.
    labels : DataFrame or Series
        Binary labels from make_labels() function. If DataFrame, we flatten to 1D.

    Output
    -------
    model_dict : dict
        Contains trained model and feature scaler.
    """
    # If labels is a DataFrame (assets x time), flatten to a 1D Series
    if isinstance(labels, pd.DataFrame):
        # Simple approach: average across assets and treat >0.5 as 1
        y_series = (labels.mean(axis=1) > 0.5).astype(int)
    else:
        y_series = labels.astype(int)

    # Align features and labels
    features_df, y_series = features_df.align(y_series, join="inner", axis=0)

    # Initialize logistic regression with elastic net regularization
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        max_iter=1000,
    )

    # Standardize features to zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    # Calibrate probability estimates using cross-validation
    calibrated_model = CalibratedClassifierCV(model, cv=5)
    calibrated_model.fit(X_scaled, y_series)

    return {"model": calibrated_model, "scaler": scaler}


def predict_gate(model_dict, features_df):
    """
    Generates probability predictions for new momentum signals.

    Inputs
    -------
    model_dict : dict
        Trained model dictionary from fit_gate().
    features_df : DataFrame
        New features to predict on.

    Output
    -------
    prob_series : Series
        Probabilities (0–1) indicating confidence in momentum signals.
    """
    # Use the same feature alignment as in training if needed
    X_scaled = model_dict["scaler"].transform(features_df)
    probabilities = model_dict["model"].predict_proba(X_scaled)[:, 1]
    return pd.Series(probabilities, index=features_df.index)


def prob_to_multiplier(prob_series):
    """
    Converts probability predictions into position size multipliers.

    Input
    -----
    prob_series : Series
        Probability predictions from predict_gate().

    Output
    ------
    multipliers : Series
        Multipliers (0–1) to scale position sizes.
    """
    # Ensure probabilities are in valid range
    clipped_probs = np.clip(prob_series, 0, 1)

    # Direct mapping: higher probability = larger position size
    multipliers = clipped_probs

    return pd.Series(multipliers, index=prob_series.index)

# Integration Flow:
# 1. Liam's code generates momentum signals and features.
# 2. make_labels() uses these to create training labels.
# 3. fit_gate() trains model on features and labels.
# 4. predict_gate() generates confidence scores for new signals.
# 5. prob_to_multiplier() converts confidence to position sizing.
# 6. Final positions = Liam's momentum signals * our multipliers.
