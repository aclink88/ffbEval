"""
Model Training & Walk-Forward Evaluation Pipeline (Polars & Scikit-Learn Engine)
Trains multiple regression models using out-of-time walk-forward validation:
- Train: 2023 & 2024 seasons
- Test: 2025 season
"""

import sys
import os
import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_featured_data() -> pl.DataFrame:
    featured_path = os.path.join(DATA_DIR, "weekly_featured_data.csv")
    if not os.path.exists(featured_path):
        raise FileNotFoundError(f"File not found: {featured_path}. Run 02_feature_engineering.py first.")
    return pl.read_csv(featured_path, ignore_errors=True)

def prepare_dataset(df: pl.DataFrame) -> pl.DataFrame:
    """Create target variable by shifting fantasy points within player & season using Polars."""
    df = df.sort(['player_id', 'season', 'week'])
    
    df = df.with_columns(
        pl.col('fantasy_points_ppr').shift(-1).over(['player_id', 'season']).alias('target_next_week_ppr')
    )
    
    return df.filter(pl.col('target_next_week_ppr').is_not_null())

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    X_const = sm.add_constant(X.fillna(0))
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    return vif_data[vif_data["Feature"] != "const"].sort_values(by="VIF", ascending=False)

def main():
    print("Loading featured dataset via Polars...")
    pl_df = load_featured_data()
    
    print("Preparing target variable (next week fantasy points)...")
    dataset_pl = prepare_dataset(pl_df)
    
    # Convert clean modeling subset to Pandas for Scikit-Learn
    dataset = dataset_pl.to_pandas()
    
    all_features = [
        'carries', 'rushing_yards', 'rushing_tds',
        'targets', 'receptions', 'receiving_yards', 'receiving_tds',
        'receiving_air_yards', 'receiving_yards_after_catch',
        'fantasy_points_ppr', 'expected_fantasy_points_ppr', 'actual_minus_expected',
        'roll3_actual_ppr', 'roll3_expected_ppr', 'roll3_targets', 'roll3_carries',
        'efficiency_delta_roll3'
    ]
    
    # Out-of-Time Walk-Forward Split: Train 2023-2024 -> Test 2025
    train_df = dataset[dataset['season'].isin([2023, 2024])].copy()
    test_df = dataset[dataset['season'] == 2025].copy()
    
    print(f"Train Set (2023-2024): {len(train_df)} rows | Test Set (2025): {len(test_df)} rows")
    
    X_train = train_df[all_features].fillna(0)
    y_train = train_df['target_next_week_ppr']
    X_test = test_df[all_features].fillna(0)
    y_test = test_df['target_next_week_ppr']
    
    print("\n========================================================")
    print("    1. MULTICOLLINEARITY AUDIT (TRAIN 2023-2024)        ")
    print("========================================================")
    vif_df = calculate_vif(X_train)
    print(vif_df.head(10).to_string(index=False))
    
    low_vif_features = vif_df[vif_df['VIF'] <= 10.0]['Feature'].tolist()
    
    print("\n========================================================")
    print("    2. WALK-FORWARD MODEL EVALUATION (TEST ON 2025)     ")
    print("========================================================")
    
    models = {
        "Naive Baseline (Roll3 Avg)": (None, ['roll3_actual_ppr']),
        "Full Feature Ridge (L2)": (Ridge(alpha=10.0), all_features),
        "Low-VIF Ridge (VIF <= 10)": (Ridge(alpha=10.0), low_vif_features),
        "Lasso Selection (L1)": (Lasso(alpha=0.1), all_features),
        "Random Forest (Full)": (RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42), all_features)
    }
    
    results = []
    for name, (model, feats) in models.items():
        if name == "Naive Baseline (Roll3 Avg)":
            preds = test_df['roll3_actual_ppr']
        else:
            model.fit(X_train[feats], y_train)
            preds = model.predict(X_test[feats])
            
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        spearman_corr, _ = spearmanr(y_test, preds)
        
        results.append({
            "Model Variant": name,
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
