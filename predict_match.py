import pandas as pd
import numpy as np
import pickle
import os

def load_resources():
    """Load model, features list, and data."""
    model_path = 'models/xgboost_v1.pkl'
    features_path = 'models/features_list_v1.pkl'
    data_path = 'data/features_with_elo_v2.csv'
    
    if not (os.path.exists(model_path) and os.path.exists(features_path)):
        raise FileNotFoundError("Model or features list missing!")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
        
    df = pd.read_csv(data_path)
    return model, features, df

def get_latest_team_features(df, team_name):
    """Get the most recent stats for the team."""
    # Standardize team name
    former_names = pd.read_csv('data/former_names.csv')
    name_map = dict(zip(former_names['former'], former_names['current']))
    std_team = name_map.get(team_name, team_name)
    
    team_data = df[(df['home_team'] == std_team) | (df['away_team'] == std_team)]
    if team_data.empty:
        return None
        
    latest_match = team_data.sort_values('date').iloc[-1]
    
    if latest_match['home_team'] == std_team:
        return {
            'elo': latest_match['home_elo'],
            'form_points': latest_match['home_form_points_last5'],
            'wins': latest_match['home_wins_last5'],
            'draws': latest_match['home_draws_last5'],
            'losses': latest_match['home_losses_last5'],
            'gf': latest_match['home_goals_scored_last5'],
            'gc': latest_match['home_goals_conceded_last5'],
            'gd': latest_match['home_goal_difference_last5']
        }
    else:
        return {
            'elo': latest_match['away_elo'],
            'form_points': latest_match['away_form_points_last5'],
            'wins': latest_match['away_wins_last5'],
            'draws': latest_match['away_draws_last5'],
            'losses': latest_match['away_losses_last5'],
            'gf': latest_match['away_goals_scored_last5'],
            'gc': latest_match['away_goals_conceded_last5'],
            'gd': latest_match['away_goal_difference_last5']
        }

def predict_match(home_team_name, away_team_name, neutral=1):
    """Predict outcome, using XGBoost model."""
    try:
        model, features, df = load_resources()
    except Exception as e:
        print(f"Error: {e}")
        return None
        
    home_feats = get_latest_team_features(df, home_team_name)
    away_feats = get_latest_team_features(df, away_team_name)
    
    if not home_feats:
        print(f"No data found for {home_team_name}")
        return None
    if not away_feats:
        print(f"No data found for {away_team_name}")
        return None
        
    # Build feature vector!
    feature_row = {
        'home_elo': home_feats['elo'],
        'away_elo': away_feats['elo'],
        'elo_difference': home_feats['elo'] - away_feats['elo'],
        'home_form_points_last5': home_feats['form_points'],
        'away_form_points_last5': away_feats['form_points'],
        'home_wins_last5': home_feats['wins'],
        'home_draws_last5': home_feats['draws'],
        'home_losses_last5': home_feats['losses'],
        'away_wins_last5': away_feats['wins'],
        'away_draws_last5': away_feats['draws'],
        'away_losses_last5': away_feats['losses'],
        'home_goals_scored_last5': home_feats['gf'],
        'away_goals_scored_last5': away_feats['gf'],
        'home_goals_conceded_last5': home_feats['gc'],
        'away_goals_conceded_last5': away_feats['gc'],
        'home_goal_difference_last5': home_feats['gd'],
        'away_goal_difference_last5': away_feats['gd'],
        'neutral': neutral
    }
    feature_df = pd.DataFrame([feature_row])
    feature_df = feature_df[features]  # Ensure correct order!
    
    probs = model.predict_proba(feature_df)[0]
    print(f"\nMatch Prediction: {home_team_name} vs {away_team_name}")
    print(f"Location: {'Neutral' if neutral else home_team_name}")
    print("-" * 45)
    print(f"{home_team_name} Win: {probs[0]*100:.1f}%")
    print(f"Draw: {probs[1]*100:.1f}%")
    print(f"{away_team_name} Win: {probs[2]*100:.1f}%")
    return probs

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        predict_match(sys.argv[1], sys.argv[2])
    else:
        predict_match("Spain", "Brazil")
