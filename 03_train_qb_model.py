"""
Quarterback (QB) Position Model Pipeline
Evaluates QB-specific regression models:
- QB Naive Baseline
- QB Ridge Regression (L2)
- Bilateral AIC OLS Model
- QB Lasso Selection (L1)
- QB Support Vector Regression (SVR - RBF Kernel with StandardScaler)
- QB Random Forest
- QB Gradient Boosting
Evaluates out-of-time walk-forward predictions on 2025/2026 QB data.
"""

import sys
import os
import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_qb_data() -> pd.DataFrame:
    raw_path = os.path.join(DATA_DIR, "weekly_featured_data.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File not found: {raw_path}. Run 02_feature_engineering.py first.")
    
    pl_df = pl.read_csv(raw_path, ignore_errors=True)
    qb_pl = pl_df.filter(pl.col('position') == 'QB').sort(['player_id', 'season', 'week'])
    
    qb_pl = qb_pl.with_columns(
        pl.col('fantasy_points_ppr').shift(-1).over(['player_id', 'season']).alias('target_next_week_ppr')
    )
    
    qb_clean = qb_pl.filter(pl.col('target_next_week_ppr').is_not_null())
    return qb_clean.to_pandas()

def stepwise_aic_selection(X: pd.DataFrame, y: pd.Series, direction: str = "both"):
    features = list(X.columns)
    
    if direction == "backward":
        selected = list(features)
        best_model = sm.OLS(y, sm.add_constant(X[selected])).fit()
        best_aic = best_model.aic
        
        while len(selected) > 1:
            best_candidate = None
            for feat in selected:
                candidate_feats = [f for f in selected if f != feat]
                model = sm.OLS(y, sm.add_constant(X[candidate_feats])).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_candidate = feat
            if best_candidate:
                selected.remove(best_candidate)
            else:
                break
        return selected

    elif direction == "forward":
        selected = []
        best_aic = float('inf')
        
        while len(selected) < len(features):
            best_candidate = None
            for feat in features:
                if feat not in selected:
                    candidate_feats = selected + [feat]
                    model = sm.OLS(y, sm.add_constant(X[candidate_feats])).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_candidate = feat
            if best_candidate:
                selected.append(best_candidate)
            else:
                break
        return selected if selected else features

    else: # "both" (Bilateral Forward-Backward)
        selected = []
        best_aic = float('inf')
        
        for _ in range(len(features)):
            changed = False
            best_add = None
            for feat in features:
                if feat not in selected:
                    trial_feats = selected + [feat]
                    model = sm.OLS(y, sm.add_constant(X[trial_feats])).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_add = feat
            if best_add:
                selected.append(best_add)
                changed = True
                
            best_remove = None
            if len(selected) > 1:
                for feat in selected:
                    trial_feats = [f for f in selected if f != feat]
                    model = sm.OLS(y, sm.add_constant(X[trial_feats])).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_remove = feat
            if best_remove:
                selected.remove(best_remove)
                changed = True
                
            if not changed:
                break
                
        return selected if selected else features

def main():
    print("Loading QB-specific dataset...")
    df = load_qb_data()
    
    qb_base_features = [
        'attempts', 'completions', 'passing_yards', 'passing_tds', 'passing_air_yards',
        'carries', 'rushing_yards', 'rushing_tds',
        'fantasy_points_ppr', 'expected_fantasy_points_ppr', 'actual_minus_expected',
        'roll3_actual_ppr', 'roll3_expected_ppr', 'roll3_carries',
        'bayesian_shrunk_ppr', 'efficiency_delta_roll3'
    ]
    
    qb_dvp_features = qb_base_features + [
        'sched_adj_actual_ppr', 'sched_adj_xfp',
        'roll3_sched_adj_actual_ppr', 'roll3_sched_adj_xfp',
        'next_opp_dvp_xfp_adj', 'next_opp_dvp_eff_adj', 'next_opp_dvp_total_adj'
    ]
    
    # Pure Opportunity + DvP Pool (excluding raw backward-looking fantasy points)
    qb_pure_opp_dvp_features = [
        'attempts', 'passing_air_yards', 'carries', 'rushing_yards',
        'expected_fantasy_points_ppr', 'roll3_expected_ppr', 'roll3_carries',
        'sched_adj_xfp', 'roll3_sched_adj_xfp', 'efficiency_delta_roll3',
        'next_opp_dvp_xfp_adj', 'next_opp_dvp_eff_adj', 'next_opp_dvp_total_adj'
    ]
    
    train_df = df[df['season'].isin([2023, 2024])].copy()
    test_df = df[df['season'].isin([2025, 2026])].copy()
    
    print(f"QB Train Set (2023-2024): {len(train_df)} rows | QB Test Set (2025-2026): {len(test_df)} rows")
    
    X_train = train_df[qb_dvp_features].fillna(0)
    y_train = train_df['target_next_week_ppr']
    X_test = test_df[qb_dvp_features].fillna(0)
    y_test = test_df['target_next_week_ppr']
    
    print("\n========================================================")
    print(" 1. STEPWISE AIC SELECTION COMPARISON (QB TRAIN SET)    ")
    print("========================================================")
    base_aic_feats = stepwise_aic_selection(X_train[qb_base_features], y_train, direction="both")
    dvp_aic_feats = stepwise_aic_selection(X_train[qb_dvp_features], y_train, direction="both")
    pure_opp_aic_feats = stepwise_aic_selection(X_train[qb_pure_opp_dvp_features], y_train, direction="both")
    print(f"Baseline AIC Features ({len(base_aic_feats)}): {base_aic_feats}")
    print(f"DvP-Enhanced AIC Features ({len(dvp_aic_feats)}): {dvp_aic_feats}")
    print(f"Pure Opp + DvP AIC Features ({len(pure_opp_aic_feats)}): {pure_opp_aic_feats}")
    
    print("\n========================================================")
    print(" 2. QB MODEL WALK-FORWARD EVALUATION (TEST ON 2025-2026)")
    print("========================================================")
    
    models = {
        "QB Naive Baseline (Roll3 Avg)": (None, ['roll3_actual_ppr']),
        "QB Baseline AIC OLS (No DvP)": (sm.OLS, base_aic_feats),
        "QB DvP-Enhanced AIC OLS": (sm.OLS, dvp_aic_feats),
        "QB Pure Opp + DvP AIC OLS": (sm.OLS, pure_opp_aic_feats),
        "QB Ridge Regression (With DvP)": (Ridge(alpha=10.0), qb_dvp_features),
        "QB Lasso Selection (With DvP)": (Lasso(alpha=0.1), qb_dvp_features),
        "QB Support Vector Regression (With DvP)": (make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10.0, epsilon=0.1)), qb_dvp_features),
        "QB Random Forest (No DvP)": (RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), qb_base_features),
        "QB Random Forest (With DvP)": (RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), qb_dvp_features),
        "QB Random Forest (DvP AIC Subset)": (RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), dvp_aic_feats),
        "QB Gradient Boosting (With DvP)": (GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42), qb_dvp_features)
    }
    
    results = []
    for name, (model, feats) in models.items():
        if name == "QB Naive Baseline (Roll3 Avg)":
            preds = test_df['roll3_actual_ppr']
        elif "OLS" in name:
            fit_ols = sm.OLS(y_train, sm.add_constant(X_train[feats])).fit()
            preds = fit_ols.predict(sm.add_constant(X_test[feats]))
        else:
            model.fit(X_train[feats], y_train)
            preds = model.predict(X_test[feats])
            
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        spearman_corr, _ = spearmanr(y_test, preds)
        
        results.append({
            "QB Model Variant": name,
            "Num Features": len(feats),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "Spearman Rank Corr": round(spearman_corr, 4)
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("========================================================")

if __name__ == "__main__":
    main()
