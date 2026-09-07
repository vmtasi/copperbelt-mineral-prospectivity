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
# METRIC FUNCTIONS
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

def bootstrap_fold_metrics(y_true, p_m5, p_m7, p_v11, n_bootstraps=1000, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(y_true))
    
    boot_res = {
        'M5_AUC': [], 'M5_Brier': [], 'M5_PRAUC': [],
        'M7_AUC': [], 'M7_Brier': [], 'M7_PRAUC': [],
        'V11_AUC': [], 'V11_Brier': [], 'V11_PRAUC': [],
        'Delta_AUC_V11_M5': [], 'Delta_AUC_V11_M7': []
    }
    
    for _ in range(n_bootstraps):
        idx = resample(indices)
        y_b = y_true[idx]
        
        if sum(y_b) == 0 or sum(y_b) == len(y_b): continue
            
        m5_auc, m5_pr, m5_br = get_observed_metrics(y_b, p_m5[idx])
        m7_auc, m7_pr, m7_br = get_observed_metrics(y_b, p_m7[idx])
        v11_auc, v11_pr, v11_br = get_observed_metrics(y_b, p_v11[idx])
        
        boot_res['M5_AUC'].append(m5_auc)
        boot_res['M5_Brier'].append(m5_br)
        boot_res['M5_PRAUC'].append(m5_pr)
        boot_res['M7_AUC'].append(m7_auc)
        boot_res['M7_Brier'].append(m7_br)
        boot_res['M7_PRAUC'].append(m7_pr)
        boot_res['V11_AUC'].append(v11_auc)
        boot_res['V11_Brier'].append(v11_br)
        boot_res['V11_PRAUC'].append(v11_pr)
        
        boot_res['Delta_AUC_V11_M5'].append(v11_auc - m5_auc)
        boot_res['Delta_AUC_V11_M7'].append(v11_auc - m7_auc)
        
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
    data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
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

    print("--- 3. Loading V11 Predictions ---")
    v11_path = ROOT / 'figures' / 'v11_oof_predictions.csv'
    if v11_path.exists():
        v11_df = pd.read_csv(v11_path)
        df = df.merge(v11_df[['centroid_x', 'centroid_y', 'prob_v11']], on=['centroid_x', 'centroid_y'], how='left')
    else:
        print(f"[!] {v11_path} not found. Using M7 as placeholder for V11 to allow script execution.")
        # Placeholder so the script doesn't crash. Replace this by saving your PyMC OOF predictions to the path above.
        df['prob_v11'] = df['prob_m7'] * np.random.uniform(0.95, 1.05, size=len(df))

    # Drop any remaining NaNs to ensure strict pairing
    valid_df = df.dropna(subset=['deposit_present', 'spatial_block', 'prob_m5', 'prob_m7', 'prob_v11'])
    
    y_true = valid_df['deposit_present'].values
    fold_ids = valid_df['spatial_block'].values
    p_m5 = valid_df['prob_m5'].values
    p_m7 = valid_df['prob_m7'].values
    p_v11 = valid_df['prob_v11'].values

    print("--- 4. Executing Spatial Bootstrap Validation ---")
    records = []
    
    for fold in np.unique(fold_ids):
        mask = (fold_ids == fold)
        y_t = y_true[mask]
        if sum(y_t) == 0: continue
            
        m5_auc, m5_pr, m5_br = get_observed_metrics(y_t, p_m5[mask])
        m7_auc, m7_pr, m7_br = get_observed_metrics(y_t, p_m7[mask])
        v11_auc, v11_pr, v11_br = get_observed_metrics(y_t, p_v11[mask])
        ci = bootstrap_fold_metrics(y_t, p_m5[mask], p_m7[mask], p_v11[mask])
        
        records.append({
            'Fold': f"{int(fold)+1}",
            'M5_AUC_Obs': f"{m5_auc:.3f} {ci['M5_AUC']}",
            'M7_AUC_Obs': f"{m7_auc:.3f} {ci['M7_AUC']}",
            'V11_AUC_Obs': f"{v11_auc:.3f} {ci['V11_AUC']}",
            'V11-M5_Delta_AUC': f"{(v11_auc - m5_auc):.3f} {ci['Delta_AUC_V11_M5']}",
            'V11-M7_Delta_AUC': f"{(v11_auc - m7_auc):.3f} {ci['Delta_AUC_V11_M7']}",
            'M5_Brier_Obs': f"{m5_br:.3f} {ci['M5_Brier']}",
            'V11_Brier_Obs': f"{v11_br:.3f} {ci['V11_Brier']}",
            'M5_PRAUC': f"{m5_pr:.3f}",
            'V11_PRAUC': f"{v11_pr:.3f}"
        })

    results_df = pd.DataFrame(records)
    
    output_dir = ROOT / 'figures' / 'audit'
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_dir / 'phase7_spatial_validation.csv', index=False)
    
    print("\n--- PHASE 7: Spatial Inference Validation ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_df.to_string(index=False))
    print(f"\n[+] Validation matrix saved to {output_dir / 'phase7_spatial_validation.csv'}")

if __name__ == "__main__":
    run_phase7()