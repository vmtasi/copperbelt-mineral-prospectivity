import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds

# ============================================================
# SPATIAL BOOTSTRAP FUNCTION
# ============================================================
def spatial_block_bootstrap(df_target, df_full, n_bins, n_bootstraps=1000, random_state=42):
    np.random.seed(random_state)
    
    # Grid creation strictly on the full dataset's bounding box
    x_bins = np.linspace(df_full['centroid_x'].min(), df_full['centroid_x'].max(), n_bins + 1)
    y_bins = np.linspace(df_full['centroid_y'].min(), df_full['centroid_y'].max(), n_bins + 1)
    
    df_target = df_target.copy()
    df_target['grid_x'] = np.digitize(df_target['centroid_x'], x_bins)
    df_target['grid_y'] = np.digitize(df_target['centroid_y'], y_bins)
    df_target['spatial_cluster'] = df_target['grid_x'].astype(str) + "_" + df_target['grid_y'].astype(str)
    
    blocks = df_target['spatial_cluster'].unique()
    n_blocks = len(blocks)
    deposit_blocks = df_target.loc[df_target['deposit_present'] == 1, 'spatial_cluster'].nunique()
    
    delta_aucs = []
    valid_draws = 0
    attempts = 0
    max_attempts = n_bootstraps * 20
    
    while valid_draws < n_bootstraps and attempts < max_attempts:
        attempts += 1
        boot_blocks = np.random.choice(blocks, size=len(blocks), replace=True)
        
        # Reconstruct the dataset from the drawn blocks
        boot_sample = pd.concat([df_target[df_target['spatial_cluster'] == b] for b in boot_blocks])
        y_b = boot_sample['deposit_present'].values
        
        # Skip if bootstrap sample lacks either positive or negative class
        if sum(y_b) == 0 or sum(y_b) == len(y_b):
            continue
            
        m5_auc = roc_auc_score(y_b, boot_sample['prob_m5'].values)
        v11_auc = roc_auc_score(y_b, boot_sample['prob_v11'].values)
        
        delta_aucs.append(v11_auc - m5_auc)
        valid_draws += 1
        
    if valid_draws < n_bootstraps:
        print(f"[!] Warning: Only {valid_draws}/{n_bootstraps} valid draws found after {max_attempts} attempts.")
        
    if not delta_aucs:
        return np.nan, np.nan, np.nan, np.nan, n_blocks, deposit_blocks
        
    delta_aucs = np.array(delta_aucs)
    med = np.median(delta_aucs)
    ci_lower = np.percentile(delta_aucs, 2.5)
    ci_upper = np.percentile(delta_aucs, 97.5)
    p_gt_0 = np.mean(delta_aucs > 0)
    
    return med, ci_lower, ci_upper, p_gt_0, n_blocks, deposit_blocks

# ============================================================
# MAIN EXECUTION
# ============================================================
def run_multiscale_robustness():
    print("--- 1. Loading Data & Fitting Baseline M5 ---")
    data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
    df = pd.read_csv(data_path)

    features = ['distance_to_fault', 'distance_to_lithology_contact', 'bouguer']
    df = df.dropna(subset=['centroid_x', 'centroid_y', 'domain', 'litho_contact_litho_class'] + features).copy()
    df['spatial_block'] = get_along_belt_folds(df, n_folds=4)
    df['prob_m5'] = np.nan

    for fold in range(4):
        train_idx = df['spatial_block'] != fold
        test_idx = df['spatial_block'] == fold
        train = df[train_idx].copy()
        test = df[test_idx].copy()
        
        for feat in features:
            scaler = StandardScaler().fit(train[[feat]])
            train[f'{feat}_z'] = scaler.transform(train[[feat]])
            test[f'{feat}_z'] = scaler.transform(test[[feat]])
        for feat in ['distance_to_lithology_contact']:
            train[f'{feat}_z_sq'] = train[f'{feat}_z'] ** 2
            test[f'{feat}_z_sq'] = test[f'{feat}_z'] ** 2

        feats_m5 = ['bouguer_z', 'distance_to_lithology_contact_z', 'distance_to_lithology_contact_z_sq']
        clf_m5 = LogisticRegression(max_iter=1000, random_state=42).fit(train[feats_m5].values, train['deposit_present'].values)
        df.loc[test_idx, 'prob_m5'] = clf_m5.predict_proba(test[feats_m5].values)[:, 1]

    print("--- 2. Integrating V11 Predictions & Validating Integrity ---")
    v11_path = ROOT / 'figures' / 'v11_oof_predictions.csv'
    v11_df = pd.read_csv(v11_path)
    
    # Strict pandas merge integrity checks
    assert not df.duplicated(['centroid_x', 'centroid_y']).any(), "CRITICAL: Duplicates found in base dataframe."
    assert not v11_df.duplicated(['centroid_x', 'centroid_y']).any(), "CRITICAL: Duplicates found in V11 dataframe."
    
    n_before = len(df)
    df = df.merge(v11_df[['centroid_x', 'centroid_y', 'prob_v11']], 
                  on=['centroid_x', 'centroid_y'], 
                  how='left', 
                  validate='one_to_one')
    
    assert len(df) == n_before, "CRITICAL: Merge altered the row count."
    
    valid_df = df.dropna(subset=['deposit_present', 'prob_m5', 'prob_v11']).copy()

    print("--- 3. Executing Multi-Scale Spatial Bootstrapping ---")
    scales = [10, 15, 20, 25]
    records = []
    
    for scale in scales:
        print(f" -> Processing {scale}x{scale} spatial grid...")
        
        # Calculate physical dimensions (in km)
        x_block_km = (valid_df['centroid_x'].max() - valid_df['centroid_x'].min()) / scale / 1000
        y_block_km = (valid_df['centroid_y'].max() - valid_df['centroid_y'].min()) / scale / 1000
        
        # Individual Folds
        for fold in range(4):
            fold_df = valid_df[valid_df['spatial_block'] == fold]
            if sum(fold_df['deposit_present']) == 0: continue
            
            obs_delta = roc_auc_score(fold_df['deposit_present'], fold_df['prob_v11']) - \
                        roc_auc_score(fold_df['deposit_present'], fold_df['prob_m5'])
            
            med, low, high, p_gt, nb, db = spatial_block_bootstrap(fold_df, valid_df, scale)
            
            records.append({
                'Block_Scale': f"{scale}x{scale}",
                'Approx_Width_km': round(x_block_km, 2),
                'Approx_Height_km': round(y_block_km, 2),
                'Scope': f"Fold_{fold+1}",
                'Occupied_Blocks': nb,
                'Deposit_Blocks': db,
                'Observed_Delta_AUC': round(obs_delta, 3),
                'Bootstrap_Median_Delta_AUC': round(med, 3),
                'Delta_95CI_Low': round(low, 3),
                'Delta_95CI_High': round(high, 3),
                'Bootstrap_Prop_Delta_gt_0': round(p_gt, 3)
            })
            
        # Pooled Global
        obs_delta_G = roc_auc_score(valid_df['deposit_present'], valid_df['prob_v11']) - \
                      roc_auc_score(valid_df['deposit_present'], valid_df['prob_m5'])
        
        medG, lowG, highG, p_gtG, nbG, dbG = spatial_block_bootstrap(valid_df, valid_df, scale)
        
        records.append({
            'Block_Scale': f"{scale}x{scale}",
            'Approx_Width_km': round(x_block_km, 2),
            'Approx_Height_km': round(y_block_km, 2),
            'Scope': "Pooled_Global",
            'Occupied_Blocks': nbG,
            'Deposit_Blocks': dbG,
            'Observed_Delta_AUC': round(obs_delta_G, 3),
            'Bootstrap_Median_Delta_AUC': round(medG, 3),
            'Delta_95CI_Low': round(lowG, 3),
            'Delta_95CI_High': round(highG, 3),
            'Bootstrap_Prop_Delta_gt_0': round(p_gtG, 3)
        })

    results_df = pd.DataFrame(records)
    output_dir = ROOT / 'figures' / 'audit'
    results_df.to_csv(output_dir / 'phase7_multiscale_robustness.csv', index=False)
    
    print("\n" + "="*140)
    print("PHASE 7.5: MULTI-SCALE SPATIAL ROBUSTNESS CHECK (Block Bootstrap)")
    print("="*140)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1500)
    print(results_df.to_string(index=False))
    print("="*140)
    print(f"\n[+] Final robustness matrix frozen and saved to {output_dir / 'phase7_multiscale_robustness.csv'}")
    print("[+] Phase 7 is closed. Model execution for Paper 1 is complete.")

if __name__ == "__main__":
    run_multiscale_robustness()