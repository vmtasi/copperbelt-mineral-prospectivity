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
# METRIC & BOOTSTRAP FUNCTIONS
# ============================================================
def calc_pr_auc(y_true, y_prob):
    if sum(y_true) == 0 or sum(y_true) == len(y_true): return np.nan
    # PR-AUC is non-interpolated average precision. Trapezoidal integration
    # of sklearn's threshold curve is dominated by the low-recall tail and is
    # not the standard rare-positive summary used in the manuscript.
    return average_precision_score(y_true, y_prob)

def get_observed_metrics(y_true, y_prob):
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return np.nan, np.nan, np.nan
    return (
        roc_auc_score(y_true, y_prob),
        calc_pr_auc(y_true, y_prob),
        brier_score_loss(y_true, y_prob)
    )

def bootstrap_metrics(y_true, p_m5, p_v14, n_bootstraps=1000, random_state=42):
    """Calculate CIs and Bootstrap_Prob_Delta_gt_0 for M5 vs V14."""
    np.random.seed(random_state)
    indices = np.arange(len(y_true))
    
    boot_res = {
        'M5_AUC': [], 'V14_AUC': [], 'Delta_AUC': [],
        'M5_PRAUC': [], 'V14_PRAUC': [],
        'M5_Brier': [], 'V14_Brier': []
    }
    
    for _ in range(n_bootstraps):
        idx = resample(indices)[0]
        y_b = y_true[idx]
        
        if sum(y_b) == 0 or sum(y_b) == len(y_b): continue
            
        m5_auc, m5_pr, m5_br = get_observed_metrics(y_b, p_m5[idx])
        v14_auc, v14_pr, v14_br = get_observed_metrics(y_b, p_v14[idx])
        
        boot_res['M5_AUC'].append(m5_auc)
        boot_res['V14_AUC'].append(v14_auc)
        boot_res['Delta_AUC'].append(v14_auc - m5_auc)
        boot_res['M5_PRAUC'].append(m5_pr)
        boot_res['V14_PRAUC'].append(v14_pr)
        boot_res['M5_Brier'].append(m5_br)
        boot_res['V14_Brier'].append(v14_br)
        
    results = {}
    for key, values in boot_res.items():
        if len(values) > 0:
            results[f'{key}_95CI'] = f"[{np.percentile(values, 2.5):.3f}, {np.percentile(values, 97.5):.3f}]"
        else:
            results[f'{key}_95CI'] = "[NaN, NaN]"
            
    if len(boot_res['Delta_AUC']) > 0:
        results['Bootstrap_Prob_Delta_gt_0'] = f"{np.mean(np.array(boot_res['Delta_AUC']) > 0):.3f}"
    else:
        results['Bootstrap_Prob_Delta_gt_0'] = "NaN"
        
    return results

# ============================================================
# MAIN EXECUTION
# ============================================================
def run_phase7_final():
    print("--- 1. Loading Data & Folds ---")
    data_path = ROOT / 'data' / 'copperbelt_training_v14_centroid_contact.csv'
    df = pd.read_csv(data_path)

    lithology_col = 'litho_contact_litho_class'
    features = ['distance_to_fault', 'distance_to_lithology_contact', 'bouguer']
    df = df.dropna(subset=['centroid_x', 'centroid_y', 'domain', lithology_col] + features).copy()
    
    df['spatial_block'] = get_along_belt_folds(df, n_folds=4)

    print("--- 2. Computing M5 Out-Of-Fold Predictions ---")
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

    print("--- 3. Loading ID-Bearing V14 OOF Predictions ---")
    v14_df = load_v14_oof()
    df, oof_report = merge_v14_oof(df, v14_df)
    print(
        f"    linkage={oof_report['linkage']}; rows={oof_report['rows']}; "
        f"fold_counts={oof_report['fold_counts']}; "
        f"label_counts={oof_report['label_counts']}"
    )

    valid_df = df.dropna(subset=['deposit_present', 'spatial_block', 'prob_m5', 'prob_v14'])
    y_true = valid_df['deposit_present'].values
    fold_ids = valid_df['spatial_block'].values
    p_m5 = valid_df['prob_m5'].values
    p_v14 = valid_df['prob_v14'].values

    print("--- 4. Executing Final Bootstrap Validation ---")
    records = []
    
    # 4A. Fold-Level Metrics
    for fold in np.unique(fold_ids):
        mask = (fold_ids == fold)
        y_t = y_true[mask]
        if sum(y_t) == 0: continue
            
        m5_auc, m5_pr, m5_br = get_observed_metrics(y_t, p_m5[mask])
        v14_auc, v14_pr, v14_br = get_observed_metrics(y_t, p_v14[mask])
        ci = bootstrap_metrics(y_t, p_m5[mask], p_v14[mask])
        
        records.append({
            'Scope': f"Fold_{int(fold)+1}",
            'Deposits': sum(y_t),
            'M5_AUC': f"{m5_auc:.3f} {ci['M5_AUC_95CI']}",
            'V14_AUC': f"{v14_auc:.3f} {ci['V14_AUC_95CI']}",
            'Delta_AUC(V14-M5)': f"{(v14_auc - m5_auc):.3f} {ci['Delta_AUC_95CI']}",
            'Bootstrap_Prob_Delta_gt_0': ci['Bootstrap_Prob_Delta_gt_0'],
            'M5_PRAUC': f"{m5_pr:.3f}",
            'V14_PRAUC': f"{v14_pr:.3f}",
            'M5_Brier': f"{m5_br:.3f}",
            'V14_Brier': f"{v14_br:.3f}"
        })

    # 4B. Pooled OOF (Global) Metrics
    print("--- 5. Computing Pooled Global OOF Metrics ---")
    m5_auc_glob, m5_pr_glob, m5_br_glob = get_observed_metrics(y_true, p_m5)
    v14_auc_glob, v14_pr_glob, v14_br_glob = get_observed_metrics(y_true, p_v14)
    ci_glob = bootstrap_metrics(y_true, p_m5, p_v14)
    
    records.append({
        'Scope': "Pooled_Global_OOF",
        'Deposits': sum(y_true),
        'M5_AUC': f"{m5_auc_glob:.3f} {ci_glob['M5_AUC_95CI']}",
        'V14_AUC': f"{v14_auc_glob:.3f} {ci_glob['V14_AUC_95CI']}",
        'Delta_AUC(V14-M5)': f"{(v14_auc_glob - m5_auc_glob):.3f} {ci_glob['Delta_AUC_95CI']}",
        'Bootstrap_Prob_Delta_gt_0': ci_glob['Bootstrap_Prob_Delta_gt_0'],
        'M5_PRAUC': f"{m5_pr_glob:.3f}",
        'V14_PRAUC': f"{v14_pr_glob:.3f}",
        'M5_Brier': f"{m5_br_glob:.3f}",
        'V14_Brier': f"{v14_br_glob:.3f}"
    })

    results_df = pd.DataFrame(records)
    
    output_dir = ROOT / 'figures' / 'audit'
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_dir / 'phase7_final_validation_v14_centroid.csv', index=False)
    
    print("\n" + "="*105)
    print("PHASE 7: FINAL SPATIAL VALIDATION MATRIX")
    print("="*105)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_df.to_string(index=False))
    print("="*105)
    print(f"\n[+] Final matrix saved to {output_dir / 'phase7_final_validation_v14_centroid.csv'}")

if __name__ == "__main__":
    run_phase7_final()
