import pandas as pd
import numpy as np
import os
from pathlib import Path

# Get script directory as base for all paths
SCRIPT_DIR = Path(__file__).parent

def reprocess():
    # Load data
    results = pd.read_csv(SCRIPT_DIR / 'data' / 'results.csv')
    elo = pd.read_csv(SCRIPT_DIR / 'data' / 'elo_ratings.csv')
    former_names = pd.read_csv(SCRIPT_DIR / 'data' / 'former_names.csv')
    
    # 1. Standardize Names
    # Create mapping: former -> current
    name_mapping = dict(zip(former_names['former'], former_names['current']))
    
    def standardize(team_name):
        return name_mapping.get(team_name, team_name)
    
    results['home_team'] = results['home_team'].apply(standardize)
    results['away_team'] = results['away_team'].apply(standardize)
    elo['team'] = elo['team'].apply(standardize)
    
    
    # 2. Rolling Features (Expanded!)
    results['date'] = pd.to_datetime(results['date'])
    results = results.sort_values('date').reset_index(drop=True)
    
    team_history = {}
    
    home_form = []
    away_form = []
    
    home_gf = []
    away_gf = []
    
    home_gc = []
    away_gc = []
    
    home_gd = []
    away_gd = []
    
    home_wins = []
    home_draws = []
    home_losses = []
    
    away_wins = []
    away_draws = []
    away_losses = []

    for idx, row in results.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        if home not in team_history:
            team_history[home] = []
        if away not in team_history:
            team_history[away] = []
        
        # Get last 5 matches
        home_last5 = team_history[home][-5:]
        away_last5 = team_history[away][-5:]
        
        # Calculate features for home team
        hw = sum(1 for m in home_last5 if m['res'] == 'W')
        hd = sum(1 for m in home_last5 if m['res'] == 'D')
        hl = sum(1 for m in home_last5 if m['res'] == 'L')
        
        hgf = sum(m['gf'] for m in home_last5)
        hgc = sum(m['gc'] for m in home_last5)
        hpts = sum(m['p'] for m in home_last5)
        
        # Calculate features for away team
        aw = sum(1 for m in away_last5 if m['res'] == 'W')
        ad = sum(1 for m in away_last5 if m['res'] == 'D')
        al = sum(1 for m in away_last5 if m['res'] == 'L')
        
        agf = sum(m['gf'] for m in away_last5)
        agc = sum(m['gc'] for m in away_last5)
        apts = sum(m['p'] for m in away_last5)
        
        # Append all features
        home_wins.append(hw)
        home_draws.append(hd)
        home_losses.append(hl)
        
        away_wins.append(aw)
        away_draws.append(ad)
        away_losses.append(al)
        
        home_gf.append(hgf)
        away_gf.append(agf)
        
        home_gc.append(hgc)
        away_gc.append(agc)
        
        home_gd.append(hgf - hgc)
        away_gd.append(agf - agc)
        
        home_form.append(hpts)
        away_form.append(apts)
        
        # Update history after prediction
        if pd.isna(row['home_score']) or pd.isna(row['away_score']):
            continue  # Skip future matches
        
        home_score = row['home_score']
        away_score = row['away_score']
        
        if home_score > away_score:
            h_res, a_res = 'W', 'L'
            h_pts, a_pts = 3, 0
        elif home_score < away_score:
            h_res, a_res = 'L', 'W'
            h_pts, a_pts = 0, 3
        else:
            h_res, a_res = 'D', 'D'
            h_pts, a_pts = 1, 1
            
        team_history[home].append({
            'res': h_res, 'p': h_pts,
            'gf': home_score, 'gc': away_score
        })
        
        team_history[away].append({
            'res': a_res, 'p': a_pts,
            'gf': away_score, 'gc': home_score
        })
    
    # Assign all new features
    results['home_form_points_last5'] = home_form
    results['away_form_points_last5'] = away_form
    
    results['home_goals_scored_last5'] = home_gf
    results['away_goals_scored_last5'] = away_gf
    
    results['home_goals_conceded_last5'] = home_gc
    results['away_goals_conceded_last5'] = away_gc
    
    results['home_goal_difference_last5'] = home_gd
    results['away_goal_difference_last5'] = away_gd
    
    results['home_wins_last5'] = home_wins
    results['home_draws_last5'] = home_draws
    results['home_losses_last5'] = home_losses
    
    results['away_wins_last5'] = away_wins
    results['away_draws_last5'] = away_draws
    results['away_losses_last5'] = away_losses

    # 3. Elo Merging
    results['year'] = results['date'].dt.year
    
    # Merge home Elo
    results = results.merge(
        elo[['year', 'team', 'rating']],
        left_on=['year', 'home_team'],
        right_on=['year', 'team'],
        how='left'
    )
    results.rename(columns={'rating': 'home_elo'}, inplace=True)
    results.drop('team', axis=1, inplace=True)
    
    # Merge away Elo
    results = results.merge(
        elo[['year', 'team', 'rating']],
        left_on=['year', 'away_team'],
        right_on=['year', 'team'],
        how='left'
    )
    results.rename(columns={'rating': 'away_elo'}, inplace=True)
    results.drop('team', axis=1, inplace=True)
    
    # Elo difference
    results['elo_difference'] = results['home_elo'] - results['away_elo']
    
    # Result calculation
    def get_result(row):
        if pd.isna(row['home_score']) or pd.isna(row['away_score']):
            return np.nan
        if row['home_score'] > row['away_score']:
            return 0
        if row['home_score'] == row['away_score']:
            return 1
        return 2
    results['result'] = results.apply(get_result, axis=1)
    
    # 4. Final Cleaning
    results = results[results['year'] >= 1901]
    results.dropna(subset=['home_elo', 'away_elo', 'result'], inplace=True)
    
    results.to_csv(SCRIPT_DIR / 'data' / 'features_with_elo_v2.csv', index=False)
    print(f"Reprocessing complete. Saved {len(results)} rows to data/features_with_elo_v2.csv!")

if __name__ == "__main__":
    reprocess()