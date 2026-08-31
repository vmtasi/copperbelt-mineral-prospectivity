import os
import sys
from pathlib import Path

# --- THE BACKEND BYPASS ---
os.environ["PYTENSOR_FLAGS"] = "mode=NUMBA"
import pytensor

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import pymc as pm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds

def run_ablation_suite(n_folds=4):
    print("--- 1. Loading Datasets ---")
    data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
    df = pd.read_csv(data_path)

    lithology_col = 'litho_contact_litho_class'
    dist_to_lith_col = 'distance_to_lithology_contact'
    dist_to_fault_col = 'distance_to_fault'
    gravity_col = 'bouguer'  
    continuous_features = [dist_to_lith_col, dist_to_fault_col, gravity_col]

    # V9: Create raw squared features BEFORE scaling
    dist_to_lith_sq_col = 'distance_to_lithology_contact_sq'
    dist_to_fault_sq_col = 'distance_to_fault_sq'
    df[dist_to_lith_sq_col] = df[dist_to_lith_col] ** 2
    df[dist_to_fault_sq_col] = df[dist_to_fault_col] ** 2
    continuous_sq_features = [dist_to_lith_sq_col, dist_to_fault_sq_col]

    all_continuous = continuous_features + continuous_sq_features
    df = df.dropna(subset=['centroid_x', 'centroid_y', lithology_col, 'domain'] + all_continuous).copy()
    
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
    
    df = pd.get_dummies(df, columns=['daly_domain'], drop_first=False, dtype=float)
    domain_cols = [c for c in df.columns if c.startswith('daly_domain_')]
    
    interaction_cols = [f'fault_x_{d}' for d in domain_cols]
    for int_col in interaction_cols:
        df[int_col] = 0.0

    print("--- 3. Engineering Lithological Features ---")
    df['spatial_block'] = get_along_belt_folds(df, n_folds=n_folds)

    df = pd.get_dummies(df, columns=[lithology_col], drop_first=True, dtype=float)
    rock_features = [col for col in df.columns if col.startswith(f'{lithology_col}_')]

    zero_deposit_dummies = [c for c in rock_features if df.loc[df[c] == 1, 'deposit_present'].sum() == 0]
    if zero_deposit_dummies:
        df = df.drop(columns=zero_deposit_dummies)
        rock_features = [c for c in rock_features if c not in zero_deposit_dummies]

    models_to_run = ['V6_Full', 'V7_NoDomains', 'V8_DomainsOnly', 'V9_NonLinear']
    results_summary = {}

    print(f"\n--- 4. Executing Scientific Ablation Suite ({len(models_to_run)} Models) ---")
    
    for model_name in models_to_run:
        print(f"\n{'='*50}\nSTARTING MODEL: {model_name}\n{'='*50}")
        fold_metrics = []
        
        for fold in range(n_folds):
            print(f"\n  Evaluating {model_name} | Spatial Block {fold + 1} of {n_folds}...")
            
            train_df = df[df['spatial_block'] != fold].copy()
            test_df = df[df['spatial_block'] == fold].copy()
            y_train_temp = train_df['deposit_present'].values

            scaler = StandardScaler()
            train_df[all_continuous] = scaler.fit_transform(train_df[all_continuous])
            test_df[all_continuous] = scaler.transform(test_df[all_continuous])
            
            for d_col, int_col in zip(domain_cols, interaction_cols):
                train_df[int_col] = train_df[dist_to_fault_col] * train_df[d_col]
                test_df[int_col] = test_df[dist_to_fault_col] * test_df[d_col]

            # Determine feature set based on ablation model
            if model_name == 'V6_Full':
                model_continuous_features = [c for c in continuous_features if c != dist_to_fault_col]
                base_features = model_continuous_features + domain_cols + interaction_cols
            elif model_name == 'V7_NoDomains':
                base_features = continuous_features.copy()
            elif model_name == 'V8_DomainsOnly':
                base_features = domain_cols.copy()
            elif model_name == 'V9_NonLinear':
                model_continuous_features = [c for c in continuous_features if c != dist_to_fault_col]
                base_features = model_continuous_features + continuous_sq_features + domain_cols + interaction_cols

            valid_features = base_features.copy()
            
            # Add valid rocks unless we are testing Domains Only
            if model_name != 'V8_DomainsOnly':
                for col in rock_features:
                    if ((train_df[col] == 1) & (y_train_temp == 1)).sum() > 0 and \
                       ((train_df[col] == 1) & (y_train_temp == 0)).sum() > 0:
                        valid_features.append(col)

            X_train_scaled = train_df[valid_features].values.astype(float)
            X_test_scaled = test_df[valid_features].values.astype(float)

            y_train = train_df['deposit_present'].values.astype(np.int32)
            y_test = test_df['deposit_present'].values.astype(np.int32)

            if sum(y_test) == 0 or sum(y_train) == 0:
                print(f"    Skipping Block {fold+1}: Class imbalance.")
                continue

            logit_base_rate = float(np.log(y_train.sum() / (len(y_train) - y_train.sum())))

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
                        
                        # Increased target_accept to 0.99 and chains to 2 for diagnostic health
                        trace = pm.sample(draws=1000, tune=1500, chains=2, cores=1,
                                          target_accept=0.99, init="adapt_diag",
                                          progressbar=True, random_seed=seed)
                    break
                except Exception as e:
                    print(f"    -> Attempt {attempt + 1} failed. Error: {str(e).splitlines()[0]}")

            if trace is None:
                continue

            # Pool both chains for prediction
            alpha_samples = trace.posterior['alpha'].values.flatten()
            beta_sample_list = [trace.posterior[f'beta_{f}'].values.flatten() for f in valid_features]
            beta_samples = np.column_stack(beta_sample_list)

            mean_probs = []
            for i in range(len(X_test_scaled)):
                cell_distribution = 1 / (1 + np.exp(-(alpha_samples + np.dot(beta_samples, X_test_scaled[i]))))
                mean_probs.append(np.mean(cell_distribution))

            fold_auc = roc_auc_score(y_test, mean_probs)
            fold_metrics.append(fold_auc)
            print(f"    -> {model_name} Block {fold + 1} Spatial AUC: {fold_auc:.3f}")

        mean_auc = np.mean(fold_metrics) if fold_metrics else 0.0
        results_summary[model_name] = mean_auc
        print(f"\n>>> {model_name} Aggregate AUC: {mean_auc:.3f} <<<")

    print("\n" + "="*50)
    print("FINAL ABLATION SUITE RESULTS")
    print("="*50)
    for model, auc in results_summary.items():
        print(f"{model.ljust(20)}: {auc:.3f}")
    print("="*50)

if __name__ == "__main__":
    run_ablation_suite()