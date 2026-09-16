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


def load_phase7_v11_oof(model_df):
    """Load and strictly align the artifact created by extract_v11_oof.py."""
    v11_oof_path = ROOT / 'figures' / 'v11_oof_predictions.csv'
    if not v11_oof_path.exists():
        raise FileNotFoundError(
            'The frozen Phase 7 V11 OOF artifact was not found: '
            f'{v11_oof_path}. Run extract_v11_oof.py before this analysis.'
        )

    v11_df = pd.read_csv(v11_oof_path)
    required_columns = {'centroid_x', 'centroid_y', 'prob_v11'}
    missing_columns = required_columns.difference(v11_df.columns)
    if missing_columns:
        raise ValueError(
            f'{v11_oof_path} is missing required columns: {sorted(missing_columns)}'
        )

    key = ['centroid_x', 'centroid_y']
    duplicate_model = model_df.duplicated(key, keep=False)
    duplicate_oof = v11_df.duplicated(key, keep=False)
    duplicate_model_count = int(duplicate_model.sum())
    duplicate_oof_count = int(duplicate_oof.sum())
    if duplicate_model_count or duplicate_oof_count:
        raise ValueError(
            'Coordinate alignment is unsafe because the observation key is not '
            f'unique (model duplicate rows={duplicate_model_count}, '
            f'OOF duplicate rows={duplicate_oof_count}).'
        )

    model_keys = set(map(tuple, model_df[key].to_numpy()))
    oof_keys = set(map(tuple, v11_df[key].to_numpy()))
    unmatched_model = len(model_keys - oof_keys)
    unmatched_oof = len(oof_keys - model_keys)
    merged = model_df.merge(
        v11_df[key + ['prob_v11']],
        on=key,
        how='left',
        validate='one_to_one',
        indicator=True,
    )
    aligned = int((merged['_merge'] == 'both').sum())
    missing_predictions = int(merged['prob_v11'].isna().sum())
    merged = merged.drop(columns='_merge')

    print('\n--- 2. Provenance & Data Alignment ---')
    print(f'Phase 7 V11 OOF artifact: {v11_oof_path}')
    print(f'Modeling rows:            {len(model_df)}')
    print(f'Artifact rows:            {len(v11_df)}')
    print(f'Aligned rows:             {aligned}')
    print(f'Unmatched model rows:     {unmatched_model}')
    print(f'Unmatched artifact rows:  {unmatched_oof}')
    print(f'Duplicated model rows:    {duplicate_model_count}')
    print(f'Duplicated artifact rows: {duplicate_oof_count}')
    print(f'Missing V11 predictions:  {missing_predictions}')

    if unmatched_model or unmatched_oof:
        raise ValueError('Observation alignment is incomplete; no AUCs were calculated.')

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
    data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
    df = pd.read_csv(data_path)

    df = df.dropna(subset=MODEL_COLUMNS).copy()
    df['daly_domain'] = df['domain'].map(map_daly_domain)
    df = df[df['daly_domain'].isin(VALID_DOMAINS)].copy()
    df['spatial_block'] = get_along_belt_folds(df, n_folds=4)
    df = load_phase7_v11_oof(df)
    print('[+] Phase 7 provenance: extract_v11_oof.py assigns each prediction from the held-out fold trace.')
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
        v11 = auc_record(domain_df['deposit_present'], domain_df['prob_v11'], 'V11')
        delta = v11['V11_AUC'] - m5['M5_AUC']
            
        results.append({
            'Daly_Domain': domain,
            'Total_Cells': len(domain_df),
            'Deposits': n_deposits,
            'Non_Deposits': n_non_deposits,
            'Aligned_Rows': int(domain_df['prob_v11'].notna().sum()),
            'Unmatched_Rows': 0,
            'Duplicated_Rows': 0,
            'M5_OOF_AUC': m5['M5_AUC'],
            'M5_CI_Lower': m5['M5_CI_Lower'],
            'M5_CI_Upper': m5['M5_CI_Upper'],
            'V11_OOF_AUC': v11['V11_AUC'],
            'V11_CI_Lower': v11['V11_CI_Lower'],
            'V11_CI_Upper': v11['V11_CI_Upper'],
            'V11_Minus_M5_Delta_AUC': delta,
            'V11_Excluded_Rows': v11['V11_Excluded'],
        })

    df_results = pd.DataFrame(results)
    
    output_dir = ROOT / 'figures' / 'audit'
    os.makedirs(output_dir, exist_ok=True)
    out_path = output_dir / 'domain_stratified_auc.csv'
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