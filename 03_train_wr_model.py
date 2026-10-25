"""
Wide Receiver (WR) Position Model Pipeline
Performs VIF Multicollinearity Audit, Stepwise Bilateral AIC Feature Selection,
and Walk-Forward Machine Learning Model Benchmarking for Wide Receivers.
Evaluates predictions on 2025/2026 WR test set.
"""

import sys
import os
import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_wr_data() -> pd.DataFrame:
    raw_path = os.path.join(DATA_DIR, "weekly_featured_data.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File not found: {raw_path}. Run 02_feature_engineering.py first.")
    
    pl_df = pl.read_csv(raw_path, ignore_errors=True)
    wr_pl = pl_df.filter(pl.col('position') == 'WR').sort(['player_id', 'season', 'week'])
    
    wr_pl = wr_pl.with_columns(
        pl.col('fantasy_points_ppr').shift(-1).over(['player_id', 'season']).alias('target_next_week_ppr')
    )
    
    wr_clean = wr_pl.filter(pl.col('target_next_week_ppr').is_not_null())
    return wr_clean.to_pandas()

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    X_const = sm.add_constant(X.fillna(0))
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    return vif_data[vif_data["Feature"] != "const"].sort_values(by="VIF", ascending=False)

def stepwise_aic_selection(X: pd.DataFrame, y: pd.Series, direction: str = "both"):
    features = list(X.columns)
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
    print("Loading WR-specific dataset...")
    df = load_wr_data()
    
    wr_base_features = [
        'targets', 'receptions', 'receiving_yards', 'receiving_tds', 'receiving_air_yards',
        'fantasy_points_ppr', 'expected_fantasy_points_ppr', 'actual_minus_expected',
        'roll3_actual_ppr', 'roll3_expected_ppr', 'roll3_targets', 'roll3_receiving_air_yards',
        'bayesian_shrunk_ppr', 'efficiency_delta_roll3', 'carries', 'rushing_yards'
    ]
    wr_base_features = [f for f in wr_base_features if f in df.columns]
    
    wr_dvp_features = wr_base_features + [
        'sched_adj_actual_ppr', 'sched_adj_xfp',
        'roll3_sched_adj_actual_ppr', 'roll3_sched_adj_xfp',
        'next_opp_dvp_xfp_adj', 'next_opp_dvp_eff_adj', 'next_opp_dvp_total_adj'
    ]
    wr_dvp_features = [f for f in wr_dvp_features if f in df.columns]
    
    wr_pure_opp_dvp_features = [
        'targets', 'receptions', 'receiving_yards', 'receiving_air_yards',
        'expected_fantasy_points_ppr', 'roll3_expected_ppr', 'roll3_targets',
        'sched_adj_xfp', 'roll3_sched_adj_xfp', 'efficiency_delta_roll3',
        'carries', 'rushing_yards',
        'next_opp_dvp_xfp_adj', 'next_opp_dvp_eff_adj', 'next_opp_dvp_total_adj'
    ]
    wr_pure_opp_dvp_features = [f for f in wr_pure_opp_dvp_features if f in df.columns]
    
    train_df = df[df['season'].isin([2023, 2024])].copy()
    test_df = df[df['season'].isin([2025, 2026])].copy()
    
    print(f"WR Train Set (2023-2024): {len(train_df)} rows | WR Test Set (2025-2026): {len(test_df)} rows")
    
    X_train = train_df[wr_dvp_features].fillna(0)
    y_train = train_df['target_next_week_ppr']
    X_test = test_df[wr_dvp_features].fillna(0)
    y_test = test_df['target_next_week_ppr']
    
    print("\n========================================================")
    print(" 1. STEPWISE BILATERAL AIC SELECTION (WR TRAIN SET)     ")
    print("========================================================")
    base_feats = stepwise_aic_selection(X_train[wr_base_features], y_train, direction="both")
    dvp_feats = stepwise_aic_selection(X_train[wr_dvp_features], y_train, direction="both")
    pure_opp_feats = stepwise_aic_selection(X_train[wr_pure_opp_dvp_features], y_train, direction="both")
    print(f"Baseline AIC Features ({len(base_feats)}): {base_feats}")
    print(f"DvP-Enhanced AIC Features ({len(dvp_feats)}): {dvp_feats}")
    print(f"Pure Opp + DvP AIC Features ({len(pure_opp_feats)}): {pure_opp_feats}")
    
    print("\n========================================================")
    print(" 2. WR MODEL WALK-FORWARD EVALUATION (TEST ON 2025-2026)")
    print("========================================================")
    
    models = {
        "WR Naive Baseline (Roll3 Avg)": (None, ['roll3_actual_ppr']),
        "WR Baseline AIC OLS (No DvP)": (sm.OLS, base_feats),
        "WR DvP-Enhanced AIC OLS": (sm.OLS, dvp_feats),
        "WR Pure Opp + DvP AIC OLS": (sm.OLS, pure_opp_feats),
        "WR Ridge Regression (No DvP)": (Ridge(alpha=10.0), wr_base_features),
        "WR Ridge Regression (With DvP)": (Ridge(alpha=10.0), wr_dvp_features),
        "WR Lasso Selection (With DvP)": (Lasso(alpha=0.1), wr_dvp_features),
        "WR Random Forest (With DvP)": (RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), wr_dvp_features),
        "WR Random Forest (DvP AIC Subset)": (RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), dvp_feats),
        "WR Gradient Boosting (With DvP)": (GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42), wr_dvp_features),
        "WR Support Vector Regressor (With DvP)": (SVR(C=1.0, epsilon=0.1), wr_dvp_features)
    }
    
    results = []
    for name, (model, feats) in models.items():
        if name == "WR Naive Baseline (Roll3 Avg)":
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
        pearson_corr, _ = pearsonr(y_test, preds)
        
        results.append({
            "WR Model Variant": name,
            "Num Features": len(feats),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "Spearman Rank Corr": round(spearman_corr, 4),
            "Pearson Corr": round(pearson_corr, 4)
        })
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("========================================================")

if __name__ == "__main__":
    main()
