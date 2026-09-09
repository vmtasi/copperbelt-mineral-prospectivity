import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds

# ============================================================
# METRIC & BOOTSTRAP FUNCTIONS
# ============================================================
def calc_pr_auc(y_true, y_prob):
    if sum(y_true) == 0 or sum(y_true) == len(y_true): return np.nan
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)

def get_observed_metrics(y_true, y_prob):
    if sum(y_true) == 0 or sum(y_true) == len(y_true):
        return np.nan, np.nan, np.nan
    return (
        roc_auc_score(y_true, y_prob),
        calc_pr_auc(y_true, y_prob),
        brier_score_loss(y_true, y_prob)
    )

def bootstrap_metrics(y_true, p_m5, p_v11, n_bootstraps=1000, random_state=42):
    """Calculates CIs and Bootstrap_Prob_Delta_gt_0 for M5 vs V11"""
    np.random.seed(random_state)
    indices = np.arange(len(y_true))
    
    boot_res = {
        'M5_AUC': [], 'V11_AUC': [], 'Delta_AUC': [],
        'M5_PRAUC': [], 'V11_PRAUC': [],
        'M5_Brier': [], 'V11_Brier': []
    }
    
    for _ in range(n_bootstraps):
        idx = resample(indices)
        y_b = y_true[idx]
        
        if sum(y_b) == 0 or sum(y_b) == len(y_b): continue
            
        m5_auc, m5_pr, m5_br = get_observed_metrics(y_b, p_m5[idx])
        v11_auc, v11_pr, v11_br = get_observed_metrics(y_b, p_v11[idx])
        
        boot_res['M5_AUC'].append(m5_auc)
        boot_res['V11_AUC'].append(v11_auc)
        boot_res['Delta_AUC'].append(v11_auc - m5_auc)
        boot_res['M5_PRAUC'].append(m5_pr)
        boot_res['V11_PRAUC'].append(v11_pr)
        boot_res['M5_Brier'].append(m5_br)
        boot_res['V11_Brier'].append(v11_br)
        
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
    data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
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

    print("--- 3. Loading V11 OOF Predictions ---")
    v11_path = ROOT / 'figures' / 'v11_oof_predictions.csv'
    if not v11_path.exists():
        print(f"[!] Error: {v11_path} not found. Run extract_v11_oof.py first.")
        return
        
    v11_df = pd.read_csv(v11_path)
    df = df.merge(v11_df[['centroid_x', 'centroid_y', 'prob_v11']], on=['centroid_x', 'centroid_y'], how='left')

    valid_df = df.dropna(subset=['deposit_present', 'spatial_block', 'prob_m5', 'prob_v11'])
    y_true = valid_df['deposit_present'].values
    fold_ids = valid_df['spatial_block'].values
    p_m5 = valid_df['prob_m5'].values
    p_v11 = valid_df['prob_v11'].values

    print("--- 4. Executing Final Bootstrap Validation ---")
    records = []
    
    # 4A. Fold-Level Metrics
    for fold in np.unique(fold_ids):
        mask = (fold_ids == fold)
        y_t = y_true[mask]
        if sum(y_t) == 0: continue
            
        m5_auc, m5_pr, m5_br = get_observed_metrics(y_t, p_m5[mask])
        v11_auc, v11_pr, v11_br = get_observed_metrics(y_t, p_v11[mask])
        ci = bootstrap_metrics(y_t, p_m5[mask], p_v11[mask])
        
        records.append({
            'Scope': f"Fold_{int(fold)+1}",
            'Deposits': sum(y_t),
            'M5_AUC': f"{m5_auc:.3f} {ci['M5_AUC_95CI']}",
            'V11_AUC': f"{v11_auc:.3f} {ci['V11_AUC_95CI']}",
            'Delta_AUC(V11-M5)': f"{(v11_auc - m5_auc):.3f} {ci['Delta_AUC_95CI']}",
            'Bootstrap_Prob_Delta_gt_0': ci['Bootstrap_Prob_Delta_gt_0'],
            'M5_PRAUC': f"{m5_pr:.3f}",
            'V11_PRAUC': f"{v11_pr:.3f}",
            'M5_Brier': f"{m5_br:.3f}",
            'V11_Brier': f"{v11_br:.3f}"
        })

    # 4B. Pooled OOF (Global) Metrics
    print("--- 5. Computing Pooled Global OOF Metrics ---")
    m5_auc_glob, m5_pr_glob, m5_br_glob = get_observed_metrics(y_true, p_m5)
    v11_auc_glob, v11_pr_glob, v11_br_glob = get_observed_metrics(y_true, p_v11)
    ci_glob = bootstrap_metrics(y_true, p_m5, p_v11)
    
    records.append({
        'Scope': "Pooled_Global_OOF",
        'Deposits': sum(y_true),
        'M5_AUC': f"{m5_auc_glob:.3f} {ci_glob['M5_AUC_95CI']}",
        'V11_AUC': f"{v11_auc_glob:.3f} {ci_glob['V11_AUC_95CI']}",
        'Delta_AUC(V11-M5)': f"{(v11_auc_glob - m5_auc_glob):.3f} {ci_glob['Delta_AUC_95CI']}",
        'Bootstrap_Prob_Delta_gt_0': ci_glob['Bootstrap_Prob_Delta_gt_0'],
        'M5_PRAUC': f"{m5_pr_glob:.3f}",
        'V11_PRAUC': f"{v11_pr_glob:.3f}",
        'M5_Brier': f"{m5_br_glob:.3f}",
        'V11_Brier': f"{v11_br_glob:.3f}"
    })

    results_df = pd.DataFrame(records)
    
    output_dir = ROOT / 'figures' / 'audit'
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_dir / 'phase7_final_validation.csv', index=False)
    
    print("\n" + "="*105)
    print("PHASE 7: FINAL SPATIAL VALIDATION MATRIX")
    print("="*105)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_df.to_string(index=False))
    print("="*105)
    print(f"\n[+] Final matrix frozen and saved to {output_dir / 'phase7_final_validation.csv'}")

if __name__ == "__main__":
    run_phase7_final()