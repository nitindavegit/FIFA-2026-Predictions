import pandas as pd
import datetime
import os
from pathlib import Path
from predict_match import predict_match

# Get script directory as base
SCRIPT_DIR = Path(__file__).parent

def load_results():
    results_path = SCRIPT_DIR / 'data' / 'results.csv'
    return pd.read_csv(results_path)

def generate_daily_post(date_str=None):
    if date_str is None:
        # In our scenario today is June 11th 2026 (based on system date)
        # We want to predict for the same day or next day matches
        target_date = datetime.date(2026, 6, 17)
        date_str = target_date.strftime('%Y-%m-%d')
        
    results_df = load_results()
    day_matches = results_df[results_df['date'] == date_str]
    
    if len(day_matches) == 0:
        print(f"No matches found for date {date_str}!")
        return
           
    post_content = []
    post_content.append(f"FIFA WORLD CUP 2026 PREDICTIONS - {date_str}")
    post_content.append(f"AI-Powered Score Predictions (xG & Poisson Simulation)")
    post_content.append("="*60)
    post_content.append("")
    
    for _, match in day_matches.iterrows():
        home = match['home_team']
        away = match['away_team']
        tournament = match['tournament']
        neutral = 1 if match.get('neutral') else 0
        
        # Use our advanced prediction engine
        prediction = predict_match(home, away, neutral=neutral)
        
        if not prediction:
            post_content.append(f"SKIPPED: {home} vs {away} (insufficient data)")
            post_content.append("")
            continue
            
        probs = prediction['probs']
        home_xg, away_xg = prediction['xg']
        h_score, a_score = prediction['score']
        
        post_content.append(f"--- {home} vs {away}")
        post_content.append(f"Tournament: {tournament}")
        post_content.append(f"Win Probabilities: {home} {probs[0]*100:.1f}% | Draw {probs[1]*100:.1f}% | {away} {probs[2]*100:.1f}%")
        post_content.append(f"Expected Goals (xG): {home} {home_xg:.2f} - {away_xg:.2f} {away}")
        post_content.append(f"Most Likely Score: {h_score} - {a_score}")
        
        # Add a verdict
        if h_score > a_score:
            verdict = f"{home} is the favorite to win."
        elif a_score > h_score:
            verdict = f"{away} is the favorite to win."
        else:
            verdict = "This looks like a very close draw."
            
        post_content.append(f"AI Verdict: {verdict}")
        post_content.append("")
        
    final_post = "\n".join(post_content)
        
    # Save it
    out_dir = SCRIPT_DIR / "content_posts"
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        
    fn = out_dir / f"post_{date_str}.txt"
    with open(fn, 'w', encoding='utf8') as f:
        f.write(final_post)
        
    print("Post generated successfully!")
    
if __name__ == "__main__":
    import sys
    d = None
    if len(sys.argv) > 1:
        d = sys.argv[1]
    generate_daily_post(d)
