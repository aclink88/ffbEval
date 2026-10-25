"""
Empirical Prior Optimization Engine for Fantasy Football
Calculates:
1. Within-player variance (sigma_within^2) and Between-player variance (sigma_between^2)
   by position group across 2023-2024 historical data.
2. Optimal Empirical Bayes Regularization Constant m_pos = (sigma_within^2 / sigma_between^2).
3. Grid Search Validation minimizing Out-of-Time MAE/RMSE.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_data() -> pd.DataFrame:
    raw_path = os.path.join(DATA_DIR, "weekly_featured_data.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File not found: {raw_path}. Run 02_feature_engineering.py first.")
    return pd.read_csv(raw_path)

def calculate_empirical_bayes_priors(df: pd.DataFrame):
    """
    Computes ANOVA variance components (sigma_within^2 and sigma_between^2)
    for each position group to derive theoretical m_pos = sigma_within^2 / sigma_between^2.
    """
    positions = ['QB', 'RB', 'WR', 'TE']
    results = []
    
    for pos in positions:
        pos_df = df[df['position'] == pos].copy()
        
        # Determine primary opportunity metric per position
        if pos == 'QB':
            opp_col = 'passing_yards'
            rate_col = 'fantasy_points_ppr'
        elif pos == 'RB':
            opp_col = 'carries'
            rate_col = 'fantasy_points_ppr'
        else: # WR / TE
            opp_col = 'targets'
            rate_col = 'fantasy_points_ppr'
            
        # Group by player to calculate variance components
        player_stats = pos_df.groupby('player_id')[rate_col].agg(['count', 'mean', 'var']).dropna()
        player_stats = player_stats[player_stats['count'] >= 4] # Require at least 4 active games
        
        if len(player_stats) < 5:
            continue
            
        # 1. Within-player variance (average of individual player variances)
        sigma_within_sq = player_stats['var'].mean()
        
        # 2. Total variance between player means
        total_between_var = player_stats['mean'].var()
        
        # 3. Average sample size per player (n_bar)
        n_bar = player_stats['count'].mean()
        
        # 4. Net between-player variance (removing sample noise)
        sigma_between_sq = max(0.01, total_between_var - (sigma_within_sq / n_bar))
        
        # Theoretical optimal regularization constant m = sigma_within_sq / sigma_between_sq
        m_theoretical = sigma_within_sq / sigma_between_sq
        
        results.append({
            "Position": pos,
            "Metric": rate_col,
            "Sigma_Within_Sq": round(sigma_within_sq, 2),
            "Sigma_Between_Sq": round(sigma_between_sq, 2),
            "Optimal_M_Constant": round(m_theoretical, 1)
        })
        
    return pd.DataFrame(results)

def grid_search_m_validation(df: pd.DataFrame):
    """
    Validates candidate m values via grid search on 2024 test data
    to minimize out-of-time prediction error.
    """
    positions = ['QB', 'RB', 'WR', 'TE']
    grid_results = []
    
    # Candidate m constants to test (opportunity units: dropbacks/touches/targets)
    candidate_m_values = [5, 10, 20, 35, 50, 75, 100, 150]
    
    train_df = df[df['season'] == 2023].copy()
    test_df = df[df['season'] == 2024].copy()
    
    for pos in positions:
        pos_train = train_df[train_df['position'] == pos]
        pos_test = test_df[test_df['position'] == pos]
        
        pos_prior_mean = pos_train['fantasy_points_ppr'].mean()
        
        best_m = None
        best_mae = float('inf')
        
        for m in candidate_m_values:
            # Apply shrinkage to test predictions
            # w = n / (n + m)
            weights = pos_test['week'] / (pos_test['week'] + m) # Using week as sample proxy
            preds = (weights * pos_test['roll3_actual_ppr']) + ((1 - weights) * pos_prior_mean)
            
            mae = np.mean(np.abs(pos_test['fantasy_points_ppr'] - preds))
            
            if mae < best_mae:
                best_mae = mae
                best_m = m
                
        grid_results.append({
            "Position": pos,
            "Prior_Mean_PPR": round(pos_prior_mean, 2),
            "Empirical_Grid_M": best_m,
            "Min_Test_MAE": round(best_mae, 3)
        })
        
    return pd.DataFrame(grid_results)

def main():
    print("Loading historical data for Empirical Bayes Prior Optimization...")
    df = load_data()
    
    print("\n========================================================")
    print("   1. EMPIRICAL BAYES VARIANCE DECOMPOSITION (ANOVA)    ")
    print("========================================================")
    eb_df = calculate_empirical_bayes_priors(df)
    print(eb_df.to_string(index=False))
    
    print("\n========================================================")
    print("   2. OUT-OF-TIME GRID SEARCH VALIDATION (TEST 2024)    ")
    print("========================================================")
    grid_df = grid_search_m_validation(df)
    print(grid_df.to_string(index=False))
    print("========================================================")

if __name__ == "__main__":
    main()
