import os
import sys
import pandas as pd
import numpy as np
import arviz as az
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds

def extract_v11_probabilities():
    print("--- 1. Loading Base Dataset & Folds ---")
    data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
    df = pd.read_csv(data_path)

    lithology_col = 'litho_contact_litho_class'
    features = ['distance_to_fault', 'distance_to_lithology_contact', 'bouguer']
    df = df.dropna(subset=['centroid_x', 'centroid_y', 'domain', lithology_col] + features).copy()
    
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
    
    unique_domains = sorted(df['daly_domain'].unique())
    domain_to_idx = {dom: i for i, dom in enumerate(unique_domains)}
    df['domain_idx'] = df['daly_domain'].map(domain_to_idx)
    n_domains = len(unique_domains)
    
    df['spatial_block'] = get_along_belt_folds(df, n_folds=4)
    df = pd.get_dummies(df, columns=[lithology_col], drop_first=True, dtype=float)
    all_rock_features = [col for col in df.columns if col.startswith(f'{lithology_col}_')]
    
    df['prob_v11'] = np.nan
    output_dir = ROOT / 'figures'

    print("--- 2. Reconstructing Out-Of-Fold Probabilities ---")
    for fold in range(4):
        trace_path = output_dir / f'v11_fold_{fold+1}_trace.nc'
        if not trace_path.exists():
            print(f"[!] Missing trace for fold {fold+1} at {trace_path}.")
            return

        print(f"Loading {trace_path.name}...")
        trace = az.from_netcdf(trace_path)
        
        train_df = df[df['spatial_block'] != fold].copy()
        test_df = df[df['spatial_block'] == fold].copy()
        y_train = train_df['deposit_present'].values
        
        # Exact scaling logic from V11
        scaler_fault = StandardScaler().fit(train_df[['distance_to_fault']])
        scaler_lith = StandardScaler().fit(train_df[['distance_to_lithology_contact']])
        scaler_grav = StandardScaler().fit(train_df[['bouguer']])
        
        for d in [train_df, test_df]:
            d['fault_z'] = scaler_fault.transform(d[['distance_to_fault']])
            d['lith_z'] = scaler_lith.transform(d[['distance_to_lithology_contact']])
            d['grav_z'] = scaler_grav.transform(d[['bouguer']])
            d['fault_z_sq'] = d['fault_z'] ** 2
            d['lith_z_sq'] = d['lith_z'] ** 2
            
        valid_rocks = [c for c in all_rock_features if ((train_df[c] == 1) & (y_train == 1)).sum() > 0 and ((train_df[c] == 1) & (y_train == 0)).sum() > 0]
        
        X_test_grav = test_df['grav_z'].values
        X_test_f_lin = test_df['fault_z'].values
        X_test_f_sq = test_df['fault_z_sq'].values
        X_test_l_lin = test_df['lith_z'].values
        X_test_l_sq = test_df['lith_z_sq'].values
        X_test_rocks = test_df[valid_rocks].values
        test_domain_idx = test_df['domain_idx'].values

        # Extract posterior samples
        alpha_samples = trace.posterior['alpha_dom'].values.reshape(-1, n_domains)
        b_f_lin_samples = trace.posterior['beta_f_lin'].values.reshape(-1, n_domains)
        b_f_sq_samples = trace.posterior['beta_f_sq'].values.reshape(-1, n_domains)
        b_l_lin_samples = trace.posterior['beta_l_lin'].values.reshape(-1, n_domains)
        b_l_sq_samples = trace.posterior['beta_l_sq'].values.reshape(-1, n_domains)
        b_grav_samples = trace.posterior['beta_grav'].values.flatten()
        b_rocks_samples = trace.posterior['beta_rocks'].values.reshape(-1, len(valid_rocks))
        
        # Reconstruct Logit
        logit_test = (
            alpha_samples[:, test_domain_idx] +
            b_f_lin_samples[:, test_domain_idx] * X_test_f_lin +
            b_f_sq_samples[:, test_domain_idx] * X_test_f_sq +
            b_l_lin_samples[:, test_domain_idx] * X_test_l_lin +
            b_l_sq_samples[:, test_domain_idx] * X_test_l_sq +
            b_grav_samples[:, None] * X_test_grav +
            np.dot(b_rocks_samples, X_test_rocks.T)
        )
        
        # Calculate mean expected probability
        prob_test = 1 / (1 + np.exp(-logit_test))
        mean_probs = np.mean(prob_test, axis=0)
        
        df.loc[df['spatial_block'] == fold, 'prob_v11'] = mean_probs

    print("--- 3. Saving Extracted V11 Predictions ---")
    out_path = output_dir / 'v11_oof_predictions.csv'
    df[['centroid_x', 'centroid_y', 'prob_v11']].to_csv(out_path, index=False)
    
    print(f"[+] Extraction complete. Data saved to: {out_path}")
    print("[+] Proceed to run `phase7_spatial_validation.py`.")

if __name__ == "__main__":
    extract_v11_probabilities()