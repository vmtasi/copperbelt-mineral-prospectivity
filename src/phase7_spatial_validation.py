import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.runtime_shim  # noqa: F401,E402  (must precede pandas)

import warnings

import numpy as np
import pandas as pd

from src.numerics import (
    LogisticRegression,
    StandardScaler,
    average_precision_score,
    brier_score_loss,
    resample,
    roc_auc_score,
)
from src.oof_provenance import load_v14_oof, merge_v14_oof
from src.validation_strategies import get_along_belt_folds

warnings.filterwarnings('ignore')

# ============================================================
# METRIC FUNCTIONS
# ============================================================
def calc_pr_auc(y_true, y_prob):
    if sum(y_true) == 0 or sum(y_true) == len(y_true): return np.nan
    return average_precision_score(y_true, y_prob)

def get_observed_metrics(y_true, y_prob):
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return np.nan, np.nan, np.nan
    return (
        roc_auc_score(y_true, y_prob),
        calc_pr_auc(y_true, y_prob),
        brier_score_loss(y_true, y_prob)
    )

def bootstrap_fold_metrics(y_true, p_m5, p_m7, p_v14, n_bootstraps=1000, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(y_true))
    
    boot_res = {
        'M5_AUC': [], 'M5_Brier': [], 'M5_PRAUC': [],
        'M7_AUC': [], 'M7_Brier': [], 'M7_PRAUC': [],
        'V14_AUC': [], 'V14_Brier': [], 'V14_PRAUC': [],
        'Delta_AUC_V14_M5': [], 'Delta_AUC_V14_M7': []
    }
    
    for _ in range(n_bootstraps):
        idx = resample(indices)[0]
        y_b = y_true[idx]
        
        if sum(y_b) == 0 or sum(y_b) == len(y_b): continue
            
        m5_auc, m5_pr, m5_br = get_observed_metrics(y_b, p_m5[idx])
        m7_auc, m7_pr, m7_br = get_observed_metrics(y_b, p_m7[idx])
        v14_auc, v14_pr, v14_br = get_observed_metrics(y_b, p_v14[idx])
        
        boot_res['M5_AUC'].append(m5_auc)
        boot_res['M5_Brier'].append(m5_br)
        boot_res['M5_PRAUC'].append(m5_pr)
        boot_res['M7_AUC'].append(m7_auc)
        boot_res['M7_Brier'].append(m7_br)
        boot_res['M7_PRAUC'].append(m7_pr)
        boot_res['V14_AUC'].append(v14_auc)
        boot_res['V14_Brier'].append(v14_br)
        boot_res['V14_PRAUC'].append(v14_pr)
        
        boot_res['Delta_AUC_V14_M5'].append(v14_auc - m5_auc)
        boot_res['Delta_AUC_V14_M7'].append(v14_auc - m7_auc)
        
    ci_dict = {}
    for key, values in boot_res.items():
        if len(values) > 0:
            ci_dict[key] = f"[{np.percentile(values, 2.5):.3f}, {np.percentile(values, 97.5):.3f}]"
        else:
            ci_dict[key] = "[NaN, NaN]"
    return ci_dict

# ============================================================
# MAIN EXECUTION
# ============================================================
def run_phase7():
    print("--- 1. Loading Data & Folds ---")
    data_path = ROOT / 'data' / 'copperbelt_training_v14_centroid_contact.csv'
    df = pd.read_csv(data_path)

    lithology_col = 'litho_contact_litho_class'
    features = ['distance_to_fault', 'distance_to_lithology_contact', 'bouguer']
    df = df.dropna(subset=['centroid_x', 'centroid_y', 'domain', lithology_col] + features).copy()
    
    df['spatial_block'] = get_along_belt_folds(df, n_folds=4)
    df = pd.get_dummies(df, columns=[lithology_col], drop_first=True, dtype=float)
    all_rock_features = [col for col in df.columns if col.startswith(f'{lithology_col}_')]

    # Arrays to store Out-Of-Fold predictions
    df['prob_m5'] = np.nan
    df['prob_m7'] = np.nan

    print("--- 2. Computing M5 and M7 Out-Of-Fold Predictions ---")
    for fold in range(4):
        train_idx = df['spatial_block'] != fold
        test_idx = df['spatial_block'] == fold
        
        train = df[train_idx].copy()
        test = df[test_idx].copy()
        y_train = train['deposit_present'].values
        
        # Scale & Create Quadratics strictly on train set
        for feat in features:
            scaler = StandardScaler().fit(train[[feat]])
            train[f'{feat}_z'] = scaler.transform(train[[feat]])
            test[f'{feat}_z'] = scaler.transform(test[[feat]])
            
        for feat in ['distance_to_fault', 'distance_to_lithology_contact']:
            train[f'{feat}_z_sq'] = train[f'{feat}_z'] ** 2
            test[f'{feat}_z_sq'] = test[f'{feat}_z'] ** 2

        valid_rocks = [c for c in all_rock_features if ((train[c] == 1) & (y_train == 1)).sum() > 0 and ((train[c] == 1) & (y_train == 0)).sum() > 0]
        
        feats_m5 = ['bouguer_z', 'distance_to_lithology_contact_z', 'distance_to_lithology_contact_z_sq']
        feats_m7 = ['bouguer_z', 'distance_to_fault_z', 'distance_to_fault_z_sq', 'distance_to_lithology_contact_z', 'distance_to_lithology_contact_z_sq'] + valid_rocks

        # Fit M5
        clf_m5 = LogisticRegression(max_iter=1000, random_state=42).fit(train[feats_m5].values, y_train)
        df.loc[test_idx, 'prob_m5'] = clf_m5.predict_proba(test[feats_m5].values)[:, 1]
        
        # Fit M7
        clf_m7 = LogisticRegression(max_iter=1000, random_state=42).fit(train[feats_m7].values, y_train)
        df.loc[test_idx, 'prob_m7'] = clf_m7.predict_proba(test[feats_m7].values)[:, 1]

    print("--- 3. Loading ID-Bearing V14 Predictions ---")
    v14_df = load_v14_oof()
    df, oof_report = merge_v14_oof(df, v14_df)
    print(
        f"    linkage={oof_report['linkage']}; rows={oof_report['rows']}; "
        f"fold_counts={oof_report['fold_counts']}; "
        f"label_counts={oof_report['label_counts']}"
    )

    # Drop any remaining NaNs to ensure strict pairing
    valid_df = df.dropna(subset=['deposit_present', 'spatial_block', 'prob_m5', 'prob_m7', 'prob_v14'])
    
    y_true = valid_df['deposit_present'].values
    fold_ids = valid_df['spatial_block'].values
    p_m5 = valid_df['prob_m5'].values
    p_m7 = valid_df['prob_m7'].values
    p_v14 = valid_df['prob_v14'].values

    print("--- 4. Executing Spatial Bootstrap Validation ---")
    records = []
    
    for fold in np.unique(fold_ids):
        mask = (fold_ids == fold)
        y_t = y_true[mask]
        if sum(y_t) == 0: continue
            
        m5_auc, m5_pr, m5_br = get_observed_metrics(y_t, p_m5[mask])
        m7_auc, m7_pr, m7_br = get_observed_metrics(y_t, p_m7[mask])
        v14_auc, v14_pr, v14_br = get_observed_metrics(y_t, p_v14[mask])
        ci = bootstrap_fold_metrics(y_t, p_m5[mask], p_m7[mask], p_v14[mask])
        
        records.append({
            'Fold': f"{int(fold)+1}",
            'M5_AUC_Obs': f"{m5_auc:.3f} {ci['M5_AUC']}",
            'M7_AUC_Obs': f"{m7_auc:.3f} {ci['M7_AUC']}",
            'V14_AUC_Obs': f"{v14_auc:.3f} {ci['V14_AUC']}",
            'V14-M5_Delta_AUC': f"{(v14_auc - m5_auc):.3f} {ci['Delta_AUC_V14_M5']}",
            'V14-M7_Delta_AUC': f"{(v14_auc - m7_auc):.3f} {ci['Delta_AUC_V14_M7']}",
            'M5_Brier_Obs': f"{m5_br:.3f} {ci['M5_Brier']}",
            'V14_Brier_Obs': f"{v14_br:.3f} {ci['V14_Brier']}",
            'M5_PRAUC': f"{m5_pr:.3f}",
            'V14_PRAUC': f"{v14_pr:.3f}"
        })

    results_df = pd.DataFrame(records)
    
    output_dir = ROOT / 'figures' / 'audit'
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_dir / 'phase7_spatial_validation_v14_centroid.csv', index=False)
    
    print("\n--- PHASE 7: Spatial Inference Validation ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_df.to_string(index=False))
    print(f"\n[+] Validation matrix saved to {output_dir / 'phase7_spatial_validation_v14_centroid.csv'}")

if __name__ == "__main__":
    run_phase7()
