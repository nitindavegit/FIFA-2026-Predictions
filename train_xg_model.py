import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

SCRIPT_DIR = Path(__file__).parent

def train_xg_models():
    print("--- Training Expected Goals (xG) Models ---")
    
    # 1. Load data
    data_path = SCRIPT_DIR / 'data' / 'features_with_elo_v2.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run reprocess_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # 2. Define features
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
    
    # 3. Prepare targets
    # We need actual scores for the matches we have targets for
    # results.csv has the scores, and features_with_elo_v2.csv was merged from it.
    # We need to ensure we have home_score and away_score in our features df.
    # Let's check if they exist.
    if 'home_score' not in df.columns or 'away_score' not in df.columns:
        print("Error: home_score or away_score missing from features file.")
        return

    X = df[features]
    y_home = df['home_score']
    y_away = df['away_score']
    
    # 4. Time-based split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_home_train, y_home_test = y_home.iloc[:split_idx], y_home.iloc[split_idx:]
    y_away_train, y_away_test = y_away.iloc[:split_idx], y_away.iloc[split_idx:]
    
    # 5. Train Home xG Model
    print("Training Home xG model...")
    home_model = xgb.XGBRegressor(
        objective='count:poisson', # Poisson is best for goal counts
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    home_model.fit(X_train, y_home_train, eval_set=[(X_test, y_home_test)], verbose=False)
    
    # 6. Train Away xG Model
    print("Training Away xG model...")
    away_model = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    away_model.fit(X_train, y_away_train, eval_set=[(X_test, y_away_test)], verbose=False)
    
    # 7. Evaluate
    home_preds = home_model.predict(X_test)
    away_preds = away_model.predict(X_test)
    
    print(f"\nHome xG MAE: {mean_absolute_error(y_home_test, home_preds):.4f}")
    print(f"Away xG MAE: {mean_absolute_error(y_away_test, away_preds):.4f}")
    
    # 8. Save Models
    models_dir = SCRIPT_DIR / 'models'
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        
    with open(models_dir / 'xg_home_model.pkl', 'wb') as f:
        pickle.dump(home_model, f)
    with open(models_dir / 'xg_away_model.pkl', 'wb') as f:
        pickle.dump(away_model, f)
        
    print("\n[DONE] xG Models saved to models/ folder.")

if __name__ == "__main__":
    train_xg_models()
