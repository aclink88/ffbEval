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

def compute_dynamic_offseason_and_rookie_priors(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Dynamically computes:
    1) Position-specific Offseason Retention Weights w_offseason = Cov(xFP_{S, wk1}, roll3_xFP_{S-1, end}) / Var(roll3_xFP_{S-1, end})
    2) Position-specific Rookie Log-Draft-Pick Curves: mu_xFP(Pick) = alpha_pos - beta_pos * ln(Pick)
    directly from the historical seasons present in `df`.
    """
    eos = df.sort_values(['player_id', 'season', 'week']).groupby(['player_id', 'season']).last().reset_index()
    eos['next_season'] = eos['season'] + 1
    wk1 = df[df['week'] == 1].copy()
    
    merged = pd.merge(
        wk1[['player_id', 'season', 'position', 'expected_fantasy_points_ppr']],
        eos[['player_id', 'next_season', 'roll3_expected_ppr']],
        left_on=['player_id', 'season'],
        right_on=['player_id', 'next_season']
    )
    
    w_offseason_map = {}
    for pos in ['QB', 'RB', 'WR', 'TE']:
        sub = merged[merged['position'] == pos].dropna(subset=['expected_fantasy_points_ppr', 'roll3_expected_ppr'])
        if len(sub) >= 10:
            cov = np.cov(sub['expected_fantasy_points_ppr'], sub['roll3_expected_ppr'])[0, 1]
            var = np.var(sub['roll3_expected_ppr'], ddof=1)
            w_val = float(np.clip(cov / var, 0.20, 0.90)) if var > 0 else 0.60
        else:
            w_val = 0.60
        w_offseason_map[pos] = w_val
        
    rookie_curve_map = {}
    has_rookie_cols = ('rookie_season' in wk1.columns) or ('draft_year' in wk1.columns)
    if has_rookie_cols and 'draft_pick' in wk1.columns:
        is_rookie = pd.Series(False, index=wk1.index)
        if 'rookie_season' in wk1.columns:
            is_rookie = is_rookie | (wk1['season'] == wk1['rookie_season'])
        if 'draft_year' in wk1.columns:
            is_rookie = is_rookie | (wk1['season'] == wk1['draft_year'])
            
        rookies_wk1 = wk1[is_rookie].copy()
        rookies_wk1['effective_pick'] = pd.to_numeric(rookies_wk1['draft_pick'], errors='coerce').fillna(300.0).clip(lower=1.0, upper=300.0)
        rookies_wk1['ln_pick'] = np.log(rookies_wk1['effective_pick'])
        
        for pos in ['QB', 'RB', 'WR', 'TE']:
            r_sub = rookies_wk1[rookies_wk1['position'] == pos].dropna(subset=['ln_pick', 'expected_fantasy_points_ppr'])
            if len(r_sub) >= 5 and r_sub['ln_pick'].nunique() >= 2:
                slope, intercept = np.polyfit(r_sub['ln_pick'], r_sub['expected_fantasy_points_ppr'], 1)
                beta = max(0.5, float(-slope))
                alpha = max(5.0, float(intercept))
                rookie_curve_map[pos] = (alpha, beta)
            else:
                pos_mean = float(df[df['position'] == pos]['expected_fantasy_points_ppr'].mean())
                rookie_curve_map[pos] = (pos_mean * 1.6, pos_mean * 0.22)
    else:
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_mean = float(df[df['position'] == pos]['expected_fantasy_points_ppr'].mean())
            rookie_curve_map[pos] = (pos_mean * 1.6, pos_mean * 0.22)
            
    return w_offseason_map, rookie_curve_map

def train_and_predict_position_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits dataset by position (QB, RB, WR, TE), dynamically fits offseason retention weights
    and rookie log-draft-pick curves, trains each position's decoupled champion model,
    and predicts both next-week PPR points and Season S Week 1 Pre-Game Projections.
    """
    df = df.sort_values(['player_id', 'season', 'week']).copy()
    df['target_next_week_ppr'] = df.groupby(['player_id', 'season'])['fantasy_points_ppr'].shift(-1)
    df['predicted_next_week_ppr'] = 0.0
    df['pre_week1_proj_xfp'] = np.nan
    df['pre_week1_pred_ppr'] = np.nan
    
    w_offseason_map, rookie_curve_map = compute_dynamic_offseason_and_rookie_priors(df)
    
    for pos, feats in POSITION_CHAMPION_FEATURES.items():
        pos_mask = df['position'] == pos
        pos_df = df[pos_mask].copy()
        if len(pos_df) == 0:
            continue
            
        valid_feats = [f for f in feats if f in df.columns]
        # Dynamically train on all historical completed seasons (< max_season), or all available target rows if single season
        max_season = int(pos_df['season'].max())
        train_sub = pos_df[pos_df['season'] < max_season].dropna(subset=['target_next_week_ppr'])
        if len(train_sub) < 50:
            train_sub = pos_df.dropna(subset=['target_next_week_ppr'])
        
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
        
        # --- Compute Dynamic Season S First-Game (Week 1) Pre-Game Projections ---
        w_off = w_offseason_map[pos]
        alpha_r, beta_r = rookie_curve_map[pos]
        pos_mean_xfp = float(train_sub['expected_fantasy_points_ppr'].mean())
        pos_feat_means = train_sub[valid_feats].fillna(0).mean()
        
        eos_df = pos_df.groupby(['player_id', 'season']).last().reset_index()
        eos_lookup = {
            (row['player_id'], int(row['season']) + 1): row
            for _, row in eos_df.iterrows()
        }
        
        first_game_indices = pos_df.groupby(['player_id', 'season']).head(1).index
        pre_X_rows = []
        pre_xfp_vals = []
        
        for idx in first_game_indices:
            row = pos_df.loc[idx]
            p_id = row['player_id']
            seas = int(row['season'])
            prev_row = eos_lookup.get((p_id, seas), None)
            
            if prev_row is not None:
                # Returning Veteran: Shrink Season S-1 end-of-season rolling baseline by dynamic w_off
                prev_xfp = float(prev_row.get('roll3_sched_adj_xfp', prev_row.get('roll3_expected_ppr', pos_mean_xfp)))
                shrunk_xfp = (w_off * prev_xfp) + ((1.0 - w_off) * pos_mean_xfp)
                
                feat_dict = {}
                for f in valid_feats:
                    if f == 'next_opp_dvp_xfp_adj':
                        feat_dict[f] = 0.0
                    elif f == 'efficiency_delta_roll3':
                        feat_dict[f] = 0.25 * float(prev_row.get(f, 0.0))
                    else:
                        feat_dict[f] = (w_off * float(prev_row.get(f, 0.0))) + ((1.0 - w_off) * float(pos_feat_means[f]))
            else:
                # Incoming Rookie / New Player: Dynamic Log-Draft-Pick Conditional Prior
                raw_pick = row.get('draft_pick', np.nan)
                eff_pick = float(raw_pick) if pd.notna(raw_pick) and float(raw_pick) >= 1.0 else 300.0
                eff_pick = np.clip(eff_pick, 1.0, 300.0)
                shrunk_xfp = max(1.5, alpha_r - beta_r * np.log(eff_pick))
                
                scale_ratio = shrunk_xfp / max(1.0, pos_mean_xfp)
                feat_dict = {}
                for f in valid_feats:
                    if f in ('next_opp_dvp_xfp_adj', 'efficiency_delta_roll3'):
                        feat_dict[f] = 0.0
                    else:
                        feat_dict[f] = float(pos_feat_means[f]) * scale_ratio
                        
            pre_xfp_vals.append(max(0.0, shrunk_xfp))
            pre_X_rows.append(feat_dict)
            
        if len(first_game_indices) > 0:
            pre_X_df = pd.DataFrame(pre_X_rows, index=first_game_indices)[valid_feats]
            pre_X_const = sm.add_constant(pre_X_df.fillna(0), has_constant='add')
            pre_preds = np.clip(ols_model.predict(pre_X_const), 0.0, None)
            
            df.loc[first_game_indices, 'pre_week1_proj_xfp'] = pre_xfp_vals
            df.loc[first_game_indices, 'pre_week1_pred_ppr'] = pre_preds
        
        print(
            f"Trained & applied decoupled {pos} model | Dynamic w_offseason={w_off:.3f} | "
            f"Rookie Curve: {alpha_r:.2f} - {beta_r:.2f}*ln(Pick)"
        )
        
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
        
    # Compute Pre-Game Projected xFP for Next Week (t+1): Schedule-Adjusted 3Wk xFP + Next Opponent DvP xFP Boost
    next_xfp_boost = df['next_opp_dvp_xfp_adj'] if 'next_opp_dvp_xfp_adj' in df.columns else 0.0
    df['projected_next_week_xfp'] = np.clip(df['recent_expected_3wk'] + next_xfp_boost, 0.0, None)
        
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
