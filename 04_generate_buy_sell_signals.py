"""
Buy/Sell Signal & Valuation Classification Engine (Position-Decoupled DvP & Pure Opportunity Models)
Trains 4 specialized position models (QB, RB, WR, TE) using our validated Pure Opportunity & DvP feature sets,
computes individual player efficiency baselines, and generates position-percentile Buy/Sell signals.
"""

import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Lasso

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

# Champion Position-Specific Feature Subsets (Validated via Walk-Forward AIC/BIC & DvP Benchmarks)
POSITION_CHAMPION_FEATURES = {
    'QB': [
        'roll3_sched_adj_xfp', 'efficiency_delta_roll3', 'next_opp_dvp_xfp_adj',
        'attempts', 'roll3_carries', 'passing_air_yards'
    ],
    'RB': [
        'roll3_expected_ppr', 'efficiency_delta_roll3', 'rushing_yards',
        'next_opp_dvp_xfp_adj', 'receptions', 'receiving_air_yards', 'roll3_targets'
    ],
    'WR': [
        'roll3_targets', 'efficiency_delta_roll3', 'receiving_yards',
        'roll3_sched_adj_xfp', 'rushing_yards', 'next_opp_dvp_xfp_adj'
    ],
    'TE': [
        'roll3_targets', 'receiving_yards', 'roll3_expected_ppr',
        'efficiency_delta_roll3', 'next_opp_dvp_xfp_adj'
    ]
}

def load_featured_data() -> pd.DataFrame:
    featured_path = os.path.join(DATA_DIR, "weekly_featured_data.csv")
    if not os.path.exists(featured_path):
        raise FileNotFoundError(f"File not found: {featured_path}. Run 02_feature_engineering.py first.")
    return pd.read_csv(featured_path, low_memory=False)

def train_and_predict_position_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits dataset by position (QB, RB, WR, TE), trains each position's decoupled
    champion model on historical seasons (2023-2024), and predicts next-week PPR points.
    """
    df = df.sort_values(['player_id', 'season', 'week']).copy()
    df['target_next_week_ppr'] = df.groupby(['player_id', 'season'])['fantasy_points_ppr'].shift(-1)
    df['predicted_next_week_ppr'] = 0.0
    
    for pos, feats in POSITION_CHAMPION_FEATURES.items():
        pos_mask = df['position'] == pos
        pos_df = df[pos_mask].copy()
        if len(pos_df) == 0:
            continue
            
        valid_feats = [f for f in feats if f in df.columns]
        train_sub = pos_df[pos_df['season'].isin([2023, 2024])].dropna(subset=['target_next_week_ppr'])
        
        X_train = sm.add_constant(train_sub[valid_feats].fillna(0), has_constant='add')
        y_train = train_sub['target_next_week_ppr']
        
        ols_model = sm.OLS(y_train, X_train).fit()
        
        X_all = sm.add_constant(pos_df[valid_feats].fillna(0), has_constant='add')
        preds = np.clip(ols_model.predict(X_all), 0.0, None)
        
        # Keep predicted_next_week_ppr as the Per-Game Talent Projection (neutral 0.0 DvP on BYE weeks)
        df.loc[pos_mask, 'predicted_next_week_ppr'] = preds
        
        # Add separate next_week_lineup_projection that zeroes out on BYE weeks for immediate start/sit context
        if 'effective_next_opponent' in pos_df.columns:
            lineup_preds = np.where(pos_df['effective_next_opponent'] == 'BYE', 0.0, preds)
        else:
            lineup_preds = preds
        df.loc[pos_mask, 'next_week_lineup_projection'] = lineup_preds
        
        print(f"Trained & applied decoupled {pos} model ({len(valid_feats)} features: {valid_feats})")
        
    return df

def calculate_player_baselines(df: pd.DataFrame) -> pd.DataFrame:
    player_stats = df.groupby('player_id').agg(
        total_actual_ppr=('fantasy_points_ppr', 'sum'),
        total_xfp=('expected_fantasy_points_ppr', 'sum')
    ).reset_index()
    
    player_stats['personal_efficiency_baseline'] = np.where(
        player_stats['total_xfp'] > 5.0,
        player_stats['total_actual_ppr'] / player_stats['total_xfp'],
        1.0
    )
    
    df = df.merge(player_stats[['player_id', 'personal_efficiency_baseline']], on='player_id', how='left')
    df['personal_efficiency_baseline'] = df['personal_efficiency_baseline'].fillna(1.0)
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df['recent_actual_3wk'] = df['roll3_actual_ppr']
    # Use schedule-adjusted 3-week expected points when available for fairer valuation comparison
    if 'roll3_sched_adj_xfp' in df.columns:
        df['recent_expected_3wk'] = df['roll3_sched_adj_xfp']
    else:
        df['recent_expected_3wk'] = df['roll3_expected_ppr']
        
    df['efficiency_gap'] = df['recent_expected_3wk'] - df['recent_actual_3wk']
    
    df['actual_position_percentile'] = df.groupby('position')['recent_actual_3wk'].rank(pct=True)
    df['expected_position_percentile'] = df.groupby('position')['recent_expected_3wk'].rank(pct=True)
    
    def classify_player_percentile(row):
        actual_pct = row['actual_position_percentile']
        expected_pct = row['expected_position_percentile']
        gap = row['efficiency_gap']
        baseline = row['personal_efficiency_baseline']
        
        # 1. GOD-TIER FANTASY ALPHA (Top 10% Actual + Top 10% Expected + NOT heavily underperforming xFP)
        if actual_pct >= 0.90 and expected_pct >= 0.90 and gap <= 3.0:
            return "FANTASY ALPHA"
            
        # 2. ELITE BUY TARGET (High producers with INSANE volume upside)
        elif expected_pct >= 0.85 and gap > 3.0:
            return "ELITE BUY TARGET"
            
        # 3. STAR PERFORMERS (Top 25% of Position: P75+)
        elif actual_pct >= 0.75:
            if gap < -4.0:
                return "HIGH-VALUE PRODUCER"
            else:
                return "SOLID STARTER"
                
        # 4. MID-TIER PERFORMERS (P50 - P74)
        elif actual_pct >= 0.50:
            if expected_pct >= 0.70 and gap > 2.5:
                return "TRUE BUY LOW"
            elif expected_pct < 0.40 and gap < -3.0:
                return "TRUE SELL HIGH"
            else:
                return "SOLID STARTER"
                
        # 5. LOW/BELOW-MEDIAN PERFORMERS (< P50 of Position)
        else:
            if expected_pct >= 0.50 and gap > 2.5:
                if baseline < 0.85:
                    return "PERPETUAL TRAP"
                else:
                    return "TRUE BUY LOW"
            elif actual_pct >= 0.35 and expected_pct < 0.25 and gap < -2.5:
                return "TRUE SELL HIGH"
            else:
                return "BENCH / FRINGE"
                
    df['valuation_signal'] = df.apply(classify_player_percentile, axis=1)
    
    next_dvp_bonus = df['next_opp_dvp_total_adj'] if 'next_opp_dvp_total_adj' in df.columns else 0.0
    
    df['undervalued_index'] = (
        (df['predicted_next_week_ppr'] * 3.0) +
        (df['efficiency_gap'] * 2.0) +
        (next_dvp_bonus * 1.5) +
        ((df['personal_efficiency_baseline'] - 1.0) * 10.0)
    )
    
    return df

def main():
    print("Loading featured dataset...")
    df = load_featured_data()
    
    print("Training & applying 4 decoupled position-specific DvP models...")
    df = train_and_predict_position_models(df)
    
    print("Calculating individual player efficiency baselines...")
    df = calculate_player_baselines(df)
    
    print("Generating Position-Percentile & Schedule-Adjusted Valuation Signals...")
    df_signals = generate_signals(df)
    
    output_path = os.path.join(DATA_DIR, "player_valuation_signals.csv")
    df_signals.to_csv(output_path, index=False)
    print(f"Refined signals successfully saved to {output_path} ({len(df_signals)} rows)")

if __name__ == "__main__":
    main()
