import os
import sys
from pathlib import Path

# --- THE BACKEND BYPASS ---
# Force PyTensor to use NUMBA instead of C++. 
# This compiles the math directly in memory, bypassing the Windows DLL firewall entirely.
os.environ["PYTENSOR_FLAGS"] = "mode=NUMBA"
import pytensor

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds


def validate_grid_coordinates(df_base):
    x_min, x_max = df_base['centroid_x'].min(), df_base['centroid_x'].max()
    y_min, y_max = df_base['centroid_y'].min(), df_base['centroid_y'].max()
    if -180 <= x_min <= 180 and -90 <= y_min <= 90:
        print("  WARNING: base grid coordinates look like geographic degrees.")
    elif x_min > 10000 and y_min > 10000:
        print("  Base grid appears to use projected coordinates in meters.")


def run_spatial_cv(df=None, spatial_blocks=None, n_folds=4, output_prefix='', plot_folds=False):
    print("--- 1. Loading Datasets ---")
    if df is None:
        data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
        df = pd.read_csv(data_path)

    validate_grid_coordinates(df)

    lithology_col = 'litho_contact_litho_class'
    dist_to_lith_col = 'distance_to_lithology_contact'
    dist_to_fault_col = 'distance_to_fault'
    gravity_col = 'bouguer'  
    continuous_features = [dist_to_lith_col, dist_to_fault_col, gravity_col]

    required_columns = ['id', 'deposit_present', 'centroid_x', 'centroid_y', lithology_col, 'domain'] + continuous_features
    df = df.dropna(subset=['centroid_x', 'centroid_y', lithology_col, 'domain'] + continuous_features).copy()
    
    print("--- 2. Engineering Daly's Tectonic Domains ---")
    df['domain'] = df['domain'].fillna('Unknown')
    
    def map_daly_domain(x):
        x_str = str(x).lower()
        if '3a' in x_str: return 'NRB_3a'
        elif '3b' in x_str: return 'NRB_3b'
        elif 'crz' in x_str: return 'CRZ'
        elif 'srb' in x_str: return 'SRB'
        elif 'nkb' in x_str: return 'NKB'
        elif 'mmsb' in x_str: return 'MMSB'
        else: return 'Unknown' 
        
    df['daly_domain'] = df['domain'].apply(map_daly_domain)
    df = df[df['daly_domain'] != 'Unknown'].copy()
    
    print(f"  Cells: {len(df)}  |  Deposits: {int(df['deposit_present'].sum())}")
    
    df = pd.get_dummies(df, columns=['daly_domain'], drop_first=False, dtype=float)
    domain_cols = [c for c in df.columns if c.startswith('daly_domain_')]
    
    interaction_cols = [f'fault_x_{d}' for d in domain_cols]
    for int_col in interaction_cols:
        df[int_col] = 0.0

    print("--- 3. Engineering Lithological Features ---")
    if spatial_blocks is None:
        spatial_blocks = get_along_belt_folds(df, n_folds=n_folds)

    df['spatial_block'] = spatial_blocks

    df = pd.get_dummies(df, columns=[lithology_col], drop_first=True, dtype=float)
    rock_features = [col for col in df.columns if col.startswith(f'{lithology_col}_')]

    zero_deposit_dummies = [c for c in rock_features if df.loc[df[c] == 1, 'deposit_present'].sum() == 0]
    if zero_deposit_dummies:
        df = df.drop(columns=zero_deposit_dummies)
        rock_features = [c for c in rock_features if c not in zero_deposit_dummies]

    fold_metrics = []
    all_y_true, all_y_pred_mean, all_y_pred_std = [], [], []
    all_ids, all_centroid_x, all_centroid_y = [], [], []

    print("\n--- 4. Executing V6 Hierarchical Spatial CV (Numba Backend) ---")
    for fold in range(n_folds):
        print(f"\nEvaluating Spatial Block {fold + 1} of {n_folds}...")
        
        train_df = df[df['spatial_block'] != fold].copy()
        test_df = df[df['spatial_block'] == fold].copy()
        y_train_temp = train_df['deposit_present'].values

        scaler = StandardScaler()
        train_df[continuous_features] = scaler.fit_transform(train_df[continuous_features])
        test_df[continuous_features] = scaler.transform(test_df[continuous_features])
        
        for d_col, int_col in zip(domain_cols, interaction_cols):
            train_df[int_col] = train_df[dist_to_fault_col] * train_df[d_col]
            test_df[int_col] = test_df[dist_to_fault_col] * test_df[d_col]

        model_continuous = [c for c in continuous_features if c != dist_to_fault_col]
        valid_features = model_continuous + domain_cols + interaction_cols
        
        for col in rock_features:
            positive_cases = (train_df[col] == 1) & (y_train_temp == 1)
            negative_cases = (train_df[col] == 1) & (y_train_temp == 0)
            if positive_cases.sum() > 0 and negative_cases.sum() > 0:
                valid_features.append(col)

        X_train_scaled = train_df[valid_features].values.astype(float)
        X_test_scaled = test_df[valid_features].values.astype(float)

        y_train = train_df['deposit_present'].values.astype(np.int32)
        y_test = test_df['deposit_present'].values.astype(np.int32)

        if sum(y_test) == 0 or sum(y_train) == 0:
            print(f"Skipping Block {fold+1}: Class imbalance anomaly.")
            continue

        n_pos = y_train.sum()
        n_total = len(y_train)
        logit_base_rate = float(np.log(n_pos / (n_total - n_pos)))

        trace = None
        for attempt, seed in enumerate([42, 7, 0]):
            try:
                with pm.Model() as prospectivity_model:
                    alpha = pm.Normal('alpha', mu=logit_base_rate, sigma=1, initval=logit_base_rate)
                    
                    mu_dom = pm.Normal('mu_dom', mu=0.0, sigma=1.0)
                    sigma_dom = pm.HalfNormal('sigma_dom', sigma=1.0)
                    
                    mu_fault = pm.Normal('mu_fault', mu=0.0, sigma=1.0)
                    sigma_fault = pm.HalfNormal('sigma_fault', sigma=1.0)
                    
                    beta_coefficients = []
                    for idx, feat_name in enumerate(valid_features):
                        if feat_name in domain_cols:
                            offset = pm.Normal(f'offset_{feat_name}', mu=0.0, sigma=1.0)
                            b = pm.Deterministic(f'beta_{feat_name}', mu_dom + offset * sigma_dom)
                            
                        elif feat_name in interaction_cols:
                            offset = pm.Normal(f'offset_{feat_name}', mu=0.0, sigma=1.0)
                            b = pm.Deterministic(f'beta_{feat_name}', mu_fault + offset * sigma_fault)
                            
                        else:
                            b = pm.Normal(f'beta_{feat_name}', mu=0.0, sigma=1.0, initval=0.0)
                            
                        beta_coefficients.append(b)
                    
                    beta_vector = pm.math.stack(beta_coefficients)
                    mu = alpha + pm.math.dot(X_train_scaled, beta_vector)
                    y_obs = pm.Bernoulli('y_obs', logit_p=mu, observed=y_train)
                    
                    print(f"  -> Attempt {attempt + 1} (seed={seed}): Starting Numba JIT sampling...")
                    
                    trace = pm.sample(draws=1000, tune=2000, chains=1, cores=1,
                                      target_accept=0.95, init="adapt_diag",
                                      progressbar=True, random_seed=seed)
                    
                    print(f"  -> Attempt {attempt + 1}: Sampling completed successfully.")
                break
            except Exception as e:
                print(f"  -> Attempt {attempt + 1} failed (seed={seed}). Error: {str(e).splitlines()[0]}")

        if trace is None:
            continue

        alpha_samples = trace.posterior['alpha'].values.flatten()
        beta_sample_list = [trace.posterior[f'beta_{f}'].values.flatten() for f in valid_features]
        beta_samples = np.column_stack(beta_sample_list)

        mean_probs, std_probs = [], []
        for i in range(len(X_test_scaled)):
            cell_features = X_test_scaled[i]
            cell_distribution = 1 / (1 + np.exp(-(alpha_samples + np.dot(beta_samples, cell_features))))
            mean_probs.append(np.mean(cell_distribution))
            std_probs.append(np.std(cell_distribution))

        fold_auc = roc_auc_score(y_test, mean_probs)
        fold_metrics.append(fold_auc)
        print(f"Block {fold + 1} Spatial AUC: {fold_auc:.3f}")

        all_y_true.extend(y_test)
        all_y_pred_mean.extend(mean_probs)
        all_y_pred_std.extend(std_probs)
        all_ids.extend(test_df['id'].values)
        all_centroid_x.extend(test_df['centroid_x'].values)
        all_centroid_y.extend(test_df['centroid_y'].values)

    if fold_metrics:
        print("\n=========================================")
        print(f"AGGREGATE SPATIAL AUC (Mean): {np.mean(fold_metrics):.3f}")
        print("=========================================")

if __name__ == "__main__":
    run_spatial_cv(plot_folds=True)