import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

SCRIPT_DIR = Path(__file__).parent

def train_shootout_model():
    print("--- Training Shootout Prediction Model ---")
    
    # 1. Load data
    data_path = SCRIPT_DIR / 'data' / 'features_shootout.csv'
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run reprocess_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Shootout data loaded: {len(df)} rows")
    
    # 2. Define features
    features = [
        "home_elo", "away_elo", "elo_difference",
        "home_shootout_rate", "away_shootout_rate",
        "home_shootout_plays", "away_shootout_plays"
    ]
    
    X = df[features]
    y = df['result'] # 1 if home won, 0 if away won
    
    # Split chronologically (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    lr = LogisticRegression(C=0.5, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    y_pred = lr.predict(X_test_scaled)
    y_prob = lr.predict_proba(X_test_scaled)
    
    print(f"Shootout Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Shootout Model Log Loss: {log_loss(y_test, y_prob):.4f}")
    
    # Save the model, scaler, and features list
    models_dir = SCRIPT_DIR / 'models'
    models_dir.mkdir(exist_ok=True, parents=True)
    
    shootout_resources = {
        'model': lr,
        'scaler': scaler,
        'features': features
    }
    
    with open(models_dir / 'shootout_model.pkl', 'wb') as f:
        pickle.dump(shootout_resources, f)
        
    print("[DONE] Shootout prediction resources saved to models/shootout_model.pkl")

if __name__ == "__main__":
    train_shootout_model()
