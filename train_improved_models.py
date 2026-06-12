import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, log_loss
)
import xgboost as xgb

SCRIPT_DIR = Path(__file__).parent

def train_models():
    # Load data
    data_path = SCRIPT_DIR / 'data' / 'features_with_elo_v2.csv'
    df = pd.read_csv(data_path)
    print(f"Data loaded: {len(df)} matches")
    
    # All features
    features = [
        "home_elo", "away_elo", "elo_difference",
        "home_form_points_last5", "away_form_points_last5",
        "home_wins_last5", "home_draws_last5", "home_losses_last5",
        "away_wins_last5", "away_draws_last5", "away_losses_last5",
        "home_goals_scored_last5", "away_goals_scored_last5",
        "home_goals_conceded_last5", "away_goals_conceded_last5",
        "home_goal_difference_last5", "away_goal_difference_last5",
        "neutral"
    ]
    
    X = df[features]
    y = df['result'].astype(int) # Ensure integer targets for XGBoost
    
    # Time-based split
    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # ---------------------
    # 1. Logistic Regression (Baseline)
    # ---------------------
    print("\n--- Training Logistic Regression ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"Logistic Regression Log Loss: {log_loss(y_test, y_prob_lr):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_lr))
    
    # ---------------------
    # 2. XGBoost (Better model!)
    # ---------------------
    print("\n--- Training XGBoost ---")
    # Use scale_pos_weight to help with class imbalance if needed,
    # or just let XGBoost handle it with default parameters (it's good!)
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=True)
    
    y_pred_xgb = model.predict(X_test)
    y_prob_xgb = model.predict_proba(X_test)
    print(f"\nXGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
    print(f"XGBoost Log Loss: {log_loss(y_test, y_prob_xgb):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_xgb))
    
    # ---------------------
    # Save best model and scaler!
    # XGBoost will almost certainly be better, so save that!
    # ---------------------
    models_dir = SCRIPT_DIR / 'models'
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best model (XGBoost) and scaler (for later use if needed)
    with open(models_dir / 'xgboost_v1.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open(models_dir / 'features_list_v1.pkl', 'wb') as f:
        pickle.dump(features, f)
        
    # Also save the Logistic regression just in case
    with open(models_dir / 'logistic_regression_v5.pkl', 'wb') as f:
        pickle.dump(lr, f)
        
    with open(models_dir / 'scaler_v5.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":
    train_models()
