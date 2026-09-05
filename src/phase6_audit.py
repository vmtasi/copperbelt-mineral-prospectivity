import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = ROOT / "data" / "copperbelt_training_v5_with_tectonic_domain.csv"
OUTPUT_DIR = ROOT / "figures" / "audit"
N_FOLDS = 4
FEATURES = ["distance_to_fault", "distance_to_lithology_contact", "bouguer"]

def map_daly_domain(value):
    x = str(value).lower()
    if "3a" in x: return "NRB_3a"
    elif "3b" in x: return "NRB_3b"
    elif "crz" in x: return "CRZ"
    elif "srb" in x: return "SRB"
    elif "nkb" in x: return "NKB"
    elif "mmsb" in x: return "MMSB"
    return "Unknown"

# ============================================================
# MAIN
# ============================================================
def run_phase6_audit():
    print("--- PHASE 6: Observation & Missingness Audit ---")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    raw_df = pd.read_csv(DATA_PATH)
    raw_df["daly_domain"] = raw_df["domain"].apply(map_daly_domain)
    df = raw_df[raw_df["daly_domain"] != "Unknown"].copy()
    
    # Recreate EXACT spatial folds
    df["spatial_block"] = get_along_belt_folds(df, n_folds=N_FOLDS)
    print(f"Rows in raw audit dataset: {len(df)}\n")

    # 1. DOMAIN-LEVEL MISSINGNESS
    domain_records = []
    for domain, group in df.groupby("daly_domain"):
        total = len(group)
        record = {
            "Domain": domain, "Total_Cells": total,
            "Deposits_Known": int(group["deposit_present"].sum()),
            "Non_Deposits": int((group["deposit_present"] == 0).sum())
        }
        
        for feat in FEATURES:
            missing = group[feat].isna().sum()
            record[f"{feat}_Missing_N"] = int(missing)
            record[f"{feat}_Missing_%"] = round(100 * missing / total, 2)
            
        complete_cases = group[FEATURES].notna().all(axis=1).sum()
        record["Complete_Cases_N"] = int(complete_cases)
        record["Complete_Cases_%"] = round(100 * complete_cases / total, 2)
        domain_records.append(record)
        
    domain_df = pd.DataFrame(domain_records)

    # 2. FOLD-LEVEL MISSINGNESS
    fold_records = []
    for fold, group in df.groupby("spatial_block"):
        total = len(group)
        record = {
            "Fold": fold + 1, "Total_Cells": total,
            "Deposits_Known": int(group["deposit_present"].sum()),
            "Non_Deposits": int((group["deposit_present"] == 0).sum())
        }
        
        for feat in FEATURES:
            missing = group[feat].isna().sum()
            record[f"{feat}_Missing_N"] = int(missing)
            record[f"{feat}_Missing_%"] = round(100 * missing / total, 2)
            
        complete_cases = group[FEATURES].notna().all(axis=1).sum()
        record["Complete_Cases_N"] = int(complete_cases)
        record["Complete_Cases_%"] = round(100 * complete_cases / total, 2)
        fold_records.append(record)
        
    fold_df = pd.DataFrame(fold_records)

    # 3. MISSINGNESS BY TARGET
    target_records = []
    for domain, group in df.groupby("daly_domain"):
        for target_value, target_name in [(1, "Deposit"), (0, "Non_Deposit")]:
            subset = group[group["deposit_present"] == target_value]
            total = len(subset)
            record = {"Domain": domain, "Target": target_name, "Cells": total}
            
            if total == 0:
                for feat in FEATURES: record[f"{feat}_Missing_%"] = np.nan
                record["Complete_Cases_%"] = np.nan
            else:
                for feat in FEATURES:
                    missing = subset[feat].isna().sum()
                    record[f"{feat}_Missing_%"] = round(100 * missing / total, 2)
                complete_cases = subset[FEATURES].notna().all(axis=1).sum()
                record["Complete_Cases_%"] = round(100 * complete_cases / total, 2)
                
            target_records.append(record)
            
    target_df = pd.DataFrame(target_records)

    # 4. FOLD × DOMAIN MISSINGNESS
    fold_domain_records = []
    for (fold, domain), group in df.groupby(["spatial_block", "daly_domain"]):
        total = len(group)
        record = {
            "Fold": fold + 1, "Domain": domain, "Cells": total, 
            "Deposits": int(group["deposit_present"].sum())
        }
        
        for feat in FEATURES:
            missing = group[feat].isna().sum()
            record[f"{feat}_Missing_%"] = round(100 * missing / total, 2) if total > 0 else np.nan
            
        complete_cases = group[FEATURES].notna().all(axis=1).sum()
        record["Complete_Cases_%"] = round(100 * complete_cases / total, 2) if total > 0 else np.nan
        fold_domain_records.append(record)
        
    fold_domain_df = pd.DataFrame(fold_domain_records)

    # 5. FEATURE-WISE OBSERVATION SUPPORT
    feature_records = []
    for feat in FEATURES:
        observed = df[feat].notna().sum()
        missing = df[feat].isna().sum()
        feature_records.append({
            "Feature": feat,
            "Observed_N": int(observed),
            "Missing_N": int(missing),
            "Observed_%": round(100 * observed / len(df), 2),
            "Missing_%": round(100 * missing / len(df), 2),
        })
        
    feature_df = pd.DataFrame(feature_records)

    # ========================================================
    # SAVE & DISPLAY
    # ========================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    domain_df.to_csv(OUTPUT_DIR / "phase6_missingness_by_domain.csv", index=False)
    fold_df.to_csv(OUTPUT_DIR / "phase6_missingness_by_fold.csv", index=False)
    target_df.to_csv(OUTPUT_DIR / "phase6_missingness_by_target.csv", index=False)
    fold_domain_df.to_csv(OUTPUT_DIR / "phase6_missingness_fold_domain.csv", index=False)
    feature_df.to_csv(OUTPUT_DIR / "phase6_feature_observation_support.csv", index=False)

    print("--- DOMAIN SUMMARY ---")
    print(domain_df.to_string(index=False))
    
    print("\n--- FOLD SUMMARY ---")
    print(fold_df.to_string(index=False))
    
    print("\n--- MISSINGNESS BY TARGET ---")
    print(target_df.to_string(index=False))
    
    print(f"\n[+] Phase 6 complete. Saved 5 outputs to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_phase6_audit()