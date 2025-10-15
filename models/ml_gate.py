import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

def make_labels(prices_df, momentum_df, horizon_days, hurdle_rule=0):
    """
    Creates binary classification labels (0 or 1) by comparing future price movements with momentum signals
    
    Input:
    - prices_df: DataFrame of asset prices
    - momentum_df: DataFrame of momentum indicators from Liam's code
    - horizon_days: Number of days to look ahead for returns
    - hurdle_rule: Minimum absolute momentum value to consider (filters out weak signals)
    
    Output:
    - Binary labels where 1 indicates momentum correctly predicts future direction
    """
    # Calculate future returns over specified horizon
    future_returns = prices_df.pct_change(horizon_days).shift(-horizon_days)
    
    # Extract directional signals from momentum indicators
    momentum_signs = np.sign(momentum_df)
    
    # Generate labels: 1 if momentum sign matches future return sign
    labels = (np.sign(future_returns) == momentum_signs).astype(int)
    
    if hurdle_rule > 0:
        # Filter out weak signals below the hurdle threshold
        labels = labels * (abs(momentum_df) > hurdle_rule).astype(int)
    
    return labels

def fit_gate(features_df, labels):
    """
    Trains a machine learning model to predict the reliability of momentum signals
    
    Input:
    - features_df: DataFrame of features derived from Liam's momentum indicators
    - labels: Binary labels from make_labels() function
    
    Output:
    - Dictionary containing trained model and feature scaler
    """
    # Initialize logistic regression with elastic net regularization
    model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,  # Balance between L1 and L2 regularization
        max_iter=1000
    )
    
    # Standardize features to zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    
    # Calibrate probability estimates using cross-validation
    calibrated_model = CalibratedClassifierCV(model, cv=5)
    calibrated_model.fit(X_scaled, labels)
    
    return {'model': calibrated_model, 'scaler': scaler}

def predict_gate(model_dict, features_df):
    """
    Generates probability predictions for new momentum signals
    
    Input:
    - model_dict: Trained model dictionary from fit_gate()
    - features_df: New features to predict on
    
    Output:
    - Series of probabilities (0-1) indicating confidence in momentum signals
    """
    X_scaled = model_dict['scaler'].transform(features_df)
    probabilities = model_dict['model'].predict_proba(X_scaled)[:, 1]
    return pd.Series(probabilities, index=features_df.index)

def prob_to_multiplier(prob_series):
    """
    Converts probability predictions into position size multipliers
    
    Input:
    - prob_series: Probability predictions from predict_gate()
    
    Output:
    - Series of multipliers (0-1) to scale position sizes
    - These multipliers can be applied to Liam's raw momentum signals
    """
    # Ensure probabilities are in valid range
    clipped_probs = np.clip(prob_series, 0, 1)
    
    # Convert probabilities directly to multipliers
    # Higher probability = larger position size
    multipliers = clipped_probs
    
    return pd.Series(multipliers, index=prob_series.index)

# Integration Flow:
# 1. Liam's code generates momentum signals and features
# 2. make_labels() uses these to create training labels
# 3. fit_gate() trains model on features and labels
# 4. predict_gate() generates confidence scores for new signals
# 5. prob_to_multiplier() converts confidence to position sizing
# 6. Final positions = Liam's momentum signals * our multipliers
