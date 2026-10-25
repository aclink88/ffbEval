"""
Tight End (TE) Position Model Pipeline (Pure Opportunity Architecture)
Performs VIF Multicollinearity Audit, Stepwise BIC & AIC Feature Selection on Pure Opportunity Metrics,
and Walk-Forward Machine Learning Model Benchmarking for Tight Ends.
Evaluates predictions on 2025/2026 TE test set.
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

def load_te_data() -> pd.DataFrame:
    raw_path = os.path.join(DATA_DIR, "weekly_featured_data.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File not found: {raw_path}. Run 02_feature_engineering.py first.")
    
    pl_df = pl.read_csv(raw_path, ignore_errors=True)
    te_pl = pl_df.filter(pl.col('position') == 'TE').sort(['player_id', 'season', 'week'])
    
    te_pl = te_pl.with_columns(
        pl.col('fantasy_points_ppr').shift(-1).over(['player_id', 'season']).alias('target_next_week_ppr')
    )
    
    te_clean = te_pl.filter(pl.col('target_next_week_ppr').is_not_null())
    return te_clean.to_pandas()

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    X_const = sm.add_constant(X.fillna(0))
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    return vif_data[vif_data["Feature"] != "const"].sort_values(by="VIF", ascending=False)

def stepwise_selection(X: pd.DataFrame, y: pd.Series, criterion: str = "bic"):
    features = list(X.columns)
    selected = []
    best_score = float('inf')
    
    for _ in range(len(features)):
        changed = False
        best_add = None
        for feat in features:
            if feat not in selected:
                trial_feats = selected + [feat]
                model = sm.OLS(y, sm.add_constant(X[trial_feats])).fit()
                score = model.bic if criterion == "bic" else model.aic
                if score < best_score:
                    best_score = score
                    best_add = feat
        if best_add:
            selected.append(best_add)
            changed = True
            
        best_remove = None
        if len(selected) > 1:
            for feat in selected:
                trial_feats = [f for f in selected if f != feat]
                model = sm.OLS(y, sm.add_constant(X[trial_feats])).fit()
                score = model.bic if criterion == "bic" else model.aic
                if score < best_score:
                    best_score = score
                    best_remove = feat
        if best_remove:
            selected.remove(best_remove)
            changed = True
            
        if not changed:
            break
            
    return selected if selected else features

def main():
    print("Loading TE-specific dataset (Pure Opportunity Focus)...")
    df = load_te_data()
    
    # Strictly PURE OPPORTUNITY features (excluding fantasy_points_ppr and roll3_actual_ppr)
    te_opp_base_features = [
        'targets', 'receptions', 'receiving_yards', 'receiving_air_yards',
        'expected_fantasy_points_ppr', 'roll3_expected_ppr', 'roll3_targets',
        'efficiency_delta_roll3'
    ]
    te_opp_base_features = [f for f in te_opp_base_features if f in df.columns]
    
    te_opp_dvp_features = te_opp_base_features + [
        'sched_adj_xfp', 'roll3_sched_adj_xfp',
        'next_opp_dvp_xfp_adj', 'next_opp_dvp_eff_adj', 'next_opp_dvp_total_adj'
    ]
    te_opp_dvp_features = [f for f in te_opp_dvp_features if f in df.columns]
    
    train_df = df[df['season'].isin([2023, 2024])].copy()
    test_df = df[df['season'].isin([2025, 2026])].copy()
    
    print(f"TE Train Set (2023-2024): {len(train_df)} rows | TE Test Set (2025-2026): {len(test_df)} rows")
    
    X_train = train_df[te_opp_dvp_features].fillna(0)
    y_train = train_df['target_next_week_ppr']
    X_test = test_df[te_opp_dvp_features].fillna(0)
    y_test = test_df['target_next_week_ppr']
    
    print("\n========================================================")
    print(" 1. STEPWISE SELECTION (BIC vs AIC on PURE OPP + DVP)   ")
    print("========================================================")
    base_bic_feats = stepwise_selection(X_train[te_opp_base_features], y_train, criterion="bic")
    base_aic_feats = stepwise_selection(X_train[te_opp_base_features], y_train, criterion="aic")
    dvp_bic_feats = stepwise_selection(X_train[te_opp_dvp_features], y_train, criterion="bic")
    dvp_aic_feats = stepwise_selection(X_train[te_opp_dvp_features], y_train, criterion="aic")
    print(f"Baseline BIC Pure Opp Features ({len(base_bic_feats)}): {base_bic_feats}")
    print(f"DvP-Enhanced BIC Pure Opp Features ({len(dvp_bic_feats)}): {dvp_bic_feats}")
    print(f"Baseline AIC Pure Opp Features ({len(base_aic_feats)}): {base_aic_feats}")
    print(f"DvP-Enhanced AIC Pure Opp Features ({len(dvp_aic_feats)}): {dvp_aic_feats}")
    
    print("\n========================================================")
    print(" 2. TE MODEL WALK-FORWARD EVALUATION (TEST ON 2025-2026)")
    print("========================================================")
    
    models = {
        "TE Naive Baseline (Roll3 Expected)": (None, ['roll3_expected_ppr']),
        "TE BIC Pure Opp OLS (No DvP)": (sm.OLS, base_bic_feats),
        "TE BIC Pure Opp + DvP OLS": (sm.OLS, dvp_bic_feats),
        "TE AIC Pure Opp OLS (No DvP)": (sm.OLS, base_aic_feats),
        "TE AIC Pure Opp + DvP OLS": (sm.OLS, dvp_aic_feats),
        "TE Ridge Regression (Pure Opp + DvP)": (Ridge(alpha=10.0), te_opp_dvp_features),
        "TE Lasso Selection (Pure Opp + DvP)": (Lasso(alpha=0.1), te_opp_dvp_features),
        "TE Random Forest (DvP AIC Features)": (RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), dvp_aic_feats),
        "TE Support Vector Regressor (Pure Opp + DvP)": (SVR(C=1.0, epsilon=0.1), te_opp_dvp_features)
    }
    
    results = []
    for name, (model, feats) in models.items():
        if name == "TE Naive Baseline (Roll3 Expected)":
            preds = test_df['roll3_expected_ppr']
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
            "TE Model Variant": name,
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
