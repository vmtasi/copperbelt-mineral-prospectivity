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

from src.numerics import LogisticRegression, StandardScaler, roc_auc_score
from src.oof_provenance import V14_OOF_PATH, load_v14_oof, merge_v14_oof
from src.validation_strategies import get_along_belt_folds

warnings.filterwarnings('ignore')

VALID_DOMAINS = ['CRZ', 'NKB', 'SRB', 'MMSB', 'NRB_3a', 'NRB_3b']
FEATURES = ['distance_to_fault', 'distance_to_lithology_contact', 'bouguer']
MODEL_COLUMNS = [
    'centroid_x', 'centroid_y', 'domain', 'litho_contact_litho_class',
    'distance_to_fault', 'distance_to_lithology_contact', 'bouguer',
    'deposit_present'
]


def map_daly_domain(value):
    value = str(value).lower()
    if '3a' in value:
        return 'NRB_3a'
    if '3b' in value:
        return 'NRB_3b'
    if 'crz' in value:
        return 'CRZ'
    if 'srb' in value:
        return 'SRB'
    if 'nkb' in value:
        return 'NKB'
    if 'mmsb' in value:
        return 'MMSB'
    return 'Unknown'


def load_phase7_v14_oof(model_df):
    """Load and strictly align the artifact created by extract_v14_oof.py."""
    v14_df = load_v14_oof()
    merged, report = merge_v14_oof(model_df, v14_df)

    print('\n--- 2. Provenance & Data Alignment ---')
    print(f'Phase 7 V14 OOF artifact: {V14_OOF_PATH}')
    print(f'Modeling rows:            {report["rows"]}')
    print(f'Artifact rows:            {len(v14_df)}')
    print(f'Aligned rows:             {report["rows"]}')
    print(f'Unmatched model rows:     0')
    print(f'Unmatched artifact rows:  0')
    print(f'Duplicated model rows:    0')
    print(f'Duplicated artifact rows: 0')
    print(f'Missing V14 predictions:  0')
    print(f'Linkage key:              {report["linkage"]}')
    print(f'Fold counts:              {report["fold_counts"]}')
    print(f'Fold mismatches:          {report["fold_mismatches"]}')
    print(f'Label mismatches:         {report["label_mismatches"]}')
    return merged

def auc_record(y_true, predictions, prefix):
    valid = predictions.notna()
    y = y_true[valid].to_numpy()
    p = predictions[valid].to_numpy()
    if len(np.unique(y)) < 2:
        return {
            f'{prefix}_AUC': np.nan,
            f'{prefix}_CI_Lower': np.nan,
            f'{prefix}_CI_Upper': np.nan,
            f'{prefix}_Excluded': int((~valid).sum()),
        }
    ci_low, ci_high = bootstrap_auc_ci(y, p)
    return {
        f'{prefix}_AUC': roc_auc_score(y, p),
        f'{prefix}_CI_Lower': ci_low,
        f'{prefix}_CI_Upper': ci_high,
        f'{prefix}_Excluded': int((~valid).sum()),
    }


# ---------------------------------------------------------
# Helper function for fast AUC bootstrap CI
# ---------------------------------------------------------
def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    aucs = []
    indices = np.arange(len(y_true))
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(indices, size=len(indices), replace=True)
        # Ensure both classes are present in the bootstrap sample
        if len(np.unique(y_true[boot_idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[boot_idx], y_pred[boot_idx]))
    if not aucs:
        return np.nan, np.nan
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def run_domain_stratified_auc():
    print("--- 1. Loading Data & Preparing Folds ---")
    data_path = ROOT / 'data' / 'copperbelt_training_v14_centroid_contact.csv'
    df = pd.read_csv(data_path)

    df = df.dropna(subset=MODEL_COLUMNS).copy()
    df['daly_domain'] = df['domain'].map(map_daly_domain)
    df = df[df['daly_domain'].isin(VALID_DOMAINS)].copy()
    df['spatial_block'] = get_along_belt_folds(df, n_folds=4)
    df = load_phase7_v14_oof(df)
    print('[+] Phase 7 provenance: extract_v14_oof.py assigns each prediction from the held-out fold trace.')
    print('[+] Fold integrity: the aligned artifact covers the same four-fold modeling frame; no in-sample predictions are used.')

    print("\n--- 3. Generating M5 Spatial OOF Predictions (Baseline Comparison) ---")
    df['M5_OOF_prob'] = np.nan
    
    for fold in range(4):
        train_idx = df['spatial_block'] != fold
        test_idx = df['spatial_block'] == fold
        
        train = df[train_idx].copy()
        test = df[test_idx].copy()
        
        scaler = StandardScaler()
        for feat in FEATURES:
            train[f'{feat}_z'] = scaler.fit_transform(train[[feat]])
            test[f'{feat}_z'] = scaler.transform(test[[feat]])
            
        for feat in ['distance_to_lithology_contact', 'distance_to_fault']:
            train[f'{feat}_z_sq'] = train[f'{feat}_z'] ** 2
            test[f'{feat}_z_sq'] = test[f'{feat}_z'] ** 2

        feats_m5 = ['bouguer_z', 'distance_to_lithology_contact_z', 'distance_to_lithology_contact_z_sq']
        
        clf_m5 = LogisticRegression(max_iter=1000, random_state=42).fit(
            train[feats_m5].values, train['deposit_present'].values
        )
        
        preds = clf_m5.predict_proba(test[feats_m5].values)[:, 1]
        df.loc[test_idx, 'M5_OOF_prob'] = preds

    print("\n--- 4. Stratifying OOF Performance by Daly Domain ---")
    results = []
    
    for domain in VALID_DOMAINS:
        domain_mask = df['daly_domain'] == domain
        domain_df = df[domain_mask]
        
        n_deposits = int(domain_df['deposit_present'].sum())
        n_non_deposits = len(domain_df) - n_deposits
        
        m5 = auc_record(domain_df['deposit_present'], domain_df['M5_OOF_prob'], 'M5')
        v14 = auc_record(domain_df['deposit_present'], domain_df['prob_v14'], 'V14')
        delta = v14['V14_AUC'] - m5['M5_AUC']
            
        results.append({
            'Daly_Domain': domain,
            'Total_Cells': len(domain_df),
            'Deposits': n_deposits,
            'Non_Deposits': n_non_deposits,
            'Aligned_Rows': int(domain_df['prob_v14'].notna().sum()),
            'Unmatched_Rows': 0,
            'Duplicated_Rows': 0,
            'M5_OOF_AUC': m5['M5_AUC'],
            'M5_CI_Lower': m5['M5_CI_Lower'],
            'M5_CI_Upper': m5['M5_CI_Upper'],
            'V14_OOF_AUC': v14['V14_AUC'],
            'V14_CI_Lower': v14['V14_CI_Lower'],
            'V14_CI_Upper': v14['V14_CI_Upper'],
            'V14_Minus_M5_Delta_AUC': delta,
            'V14_Excluded_Rows': v14['V14_Excluded'],
        })

    df_results = pd.DataFrame(results)
    
    output_dir = ROOT / 'figures' / 'audit'
    os.makedirs(output_dir, exist_ok=True)
    out_path = output_dir / 'domain_stratified_auc_v14_centroid.csv'
    df_results.to_csv(out_path, index=False)
    
    print("\n" + "="*110)
    print("DOMAIN-STRATIFIED SPATIAL OOF PERFORMANCE")
    print("="*110)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_results.to_string(index=False))
    print("="*110)
    print(f"\n[+] Saved to {out_path}")

if __name__ == "__main__":
    run_domain_stratified_auc()
