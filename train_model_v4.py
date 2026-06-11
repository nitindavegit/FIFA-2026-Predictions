import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, log_loss)
import pickle
import os

def train_v4():
    # Load data - load the cleaned one directly
    data_path = 'data/features_with_elo_cleaned.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run reprocess_data.py first.")
        return

    df = pd.read_csv(data_path)
    print(f"Training dataset size: {len(df)}")
    
    # Define features
    features = [
        "home_elo", "away_elo", "elo_difference",
        "home_form_points_last5", "away_form_points_last5",
        "home_goals_scored_last5", "away_goals_scored_last5",
        "home_goals_conceded_last5", "away_goals_conceded_last5",
        "home_goal_difference_last5", "away_goal_difference_last5",
        "neutral"
    ]

    X = df[features]
    y = df['result']
    
    # Split (80/20, time-based)
    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    # Phase 5: Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Log Loss: {log_loss(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save Model and Scaler
    if not os.path.exists('models'):
        os.makedirs('models')

    with open('models/logistic_regression_v4.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('models/scaler_v4.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("\nModel and Scaler v4 saved successfully!")

if __name__ == "__main__":
    train_v4()
