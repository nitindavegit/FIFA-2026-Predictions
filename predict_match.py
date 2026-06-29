import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from scipy.stats import poisson

SCRIPT_DIR = Path(__file__).parent

def load_resources():
    """Load all models, features list, and data."""
    resources = {
        'win_model': SCRIPT_DIR / 'models' / 'xgboost_v1.pkl',
        'xg_home': SCRIPT_DIR / 'models' / 'xg_home_model.pkl',
        'xg_away': SCRIPT_DIR / 'models' / 'xg_away_model.pkl',
        'features': SCRIPT_DIR / 'models' / 'features_list_v1.pkl',
        'data': SCRIPT_DIR / 'data' / 'features_with_elo_v2.csv'
    }
    
    loaded = {}
    for key, path in resources.items():
        if key == 'data':
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing resource: {path}")
        with open(path, 'rb') as f:
            loaded[key] = pickle.load(f)
            
    loaded['df'] = pd.read_csv(resources['data'])
    
    # Load shootout model if available
    shootout_path = SCRIPT_DIR / 'models' / 'shootout_model.pkl'
    if shootout_path.exists():
        with open(shootout_path, 'rb') as f:
            loaded['shootout_resources'] = pickle.load(f)
            
    return loaded

def get_latest_shootout_features(team_name):
    """Get cumulative shootout stats for the team up to present."""
    shootouts_path = SCRIPT_DIR / 'data' / 'shootouts.csv'
    if not shootouts_path.exists():
        return {'wins': 0, 'plays': 0, 'rate': 0.5}
        
    shootouts = pd.read_csv(shootouts_path)
    former_names = pd.read_csv(SCRIPT_DIR / 'data' / 'former_names.csv')
    name_map = dict(zip(former_names['former'], former_names['current']))
    std_team = name_map.get(team_name, team_name)
    
    # Filter for shootouts where team played
    team_shootouts = shootouts[(shootouts['home_team'] == std_team) | (shootouts['away_team'] == std_team)]
    if team_shootouts.empty:
        return {'wins': 0, 'plays': 0, 'rate': 0.5}
        
    plays = len(team_shootouts)
    wins = len(team_shootouts[team_shootouts['winner'] == std_team])
    rate = wins / plays if plays > 0 else 0.5
    
    return {'wins': wins, 'plays': plays, 'rate': rate}


def get_latest_team_features(df, team_name):
    """Get the most recent stats for the team."""
    former_names = pd.read_csv(SCRIPT_DIR / 'data' / 'former_names.csv')
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

def simulate_score(home_xg, away_xg):
    """Find the most likely score using Poisson distribution."""
    max_goals = 6
    score_matrix = np.zeros((max_goals, max_goals))
    
    for i in range(max_goals):
        for j in range(max_goals):
            # Probability of home team scoring i goals AND away team scoring j goals
            prob = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
            score_matrix[i, j] = prob
            
    # Find the indices of the maximum probability
    home_goals, away_goals = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    confidence = score_matrix[home_goals, away_goals]
    
    return home_goals, away_goals, confidence

def predict_match(home_team_name, away_team_name, neutral=1):
    """Predict outcome, xG, and final score."""
    try:
        res = load_resources()
    except Exception as e:
        print(f"Error: {e}")
        return None
        
    home_feats = get_latest_team_features(res['df'], home_team_name)
    away_feats = get_latest_team_features(res['df'], away_team_name)
    
    if not home_feats or not away_feats:
        print(f"No data found for {home_team_name} or {away_team_name}")
        return None
        
    # Build feature vector
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
    feature_df = pd.DataFrame([feature_row])[res['features']]
    
    # 1. Win/Draw/Loss Probabilities
    win_probs = res['win_model'].predict_proba(feature_df)[0]
    
    # 2. Expected Goals (xG)
    home_xg = res['xg_home'].predict(feature_df)[0]
    away_xg = res['xg_away'].predict(feature_df)[0]
    
    # 3. Most Likely Score
    h_score, a_score, score_conf = simulate_score(home_xg, away_xg)
    
    # 4. Shootout and advancement calculations if model is loaded
    has_shootout = 'shootout_resources' in res
    p_home_so, p_away_so = 0.5, 0.5
    p_home_advance, p_away_advance = win_probs[0], win_probs[2]
    
    if has_shootout:
        home_so = get_latest_shootout_features(home_team_name)
        away_so = get_latest_shootout_features(away_team_name)
        so_res = res['shootout_resources']
        so_scaler = so_res['scaler']
        so_model = so_res['model']
        so_feats_list = so_res['features']
        
        so_feature_row = {
            'home_elo': home_feats['elo'],
            'away_elo': away_feats['elo'],
            'elo_difference': home_feats['elo'] - away_feats['elo'],
            'home_shootout_rate': home_so['rate'],
            'away_shootout_rate': away_so['rate'],
            'home_shootout_plays': home_so['plays'],
            'away_shootout_plays': away_so['plays']
        }
        so_df = pd.DataFrame([so_feature_row])[so_feats_list]
        so_df_scaled = so_scaler.transform(so_df)
        probs_so = so_model.predict_proba(so_df_scaled)[0]
        
        p_home_so = probs_so[1]
        p_away_so = probs_so[0]
        
        p_home_advance = win_probs[0] + (win_probs[1] * p_home_so)
        p_away_advance = win_probs[2] + (win_probs[1] * p_away_so)
    
    print(f"\nMatch Prediction: {home_team_name} vs {away_team_name}")
    print(f"Location: {'Neutral' if neutral else home_team_name}")
    print("-" * 50)
    print(f"Probabilities: {home_team_name} {win_probs[0]*100:.1f}% | Draw {win_probs[1]*100:.1f}% | {away_team_name} {win_probs[2]*100:.1f}%")
    print(f"Expected Goals (xG): {home_team_name} {home_xg:.2f} - {away_xg:.2f} {away_team_name}")
    print(f"Most Likely Score: {h_score} - {a_score} ({score_conf*100:.1f}% confidence)")
    
    if has_shootout:
        print(f"Shootout Win Prob: {home_team_name} {p_home_so*100:.1f}% | {p_away_so*100:.1f}% {away_team_name}")
        print(f"Probability to Advance: {home_team_name} {p_home_advance*100:.1f}% | {p_away_advance*100:.1f}% {away_team_name}")
        
    print("-" * 50)
    
    out_dict = {
        'probs': win_probs,
        'xg': (home_xg, away_xg),
        'score': (h_score, a_score)
    }
    if has_shootout:
        out_dict['shootout_probs'] = (p_home_so, p_away_so)
        out_dict['advance_probs'] = (p_home_advance, p_away_advance)
        
    return out_dict

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        predict_match(sys.argv[1], sys.argv[2])
    else:
        # Test case for example 
        predict_match("Germany", "France")
