import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_strategies import get_along_belt_folds


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = ROOT / "data" / "copperbelt_training_v5_with_tectonic_domain.csv"
OUTPUT_DIR = ROOT / "figures" / "audit"

N_FOLDS = 4

# V11 results already obtained from Bayesian hierarchical model
V11_RESULTS = {
    0: 0.817,
    1: 0.688,
    2: 0.676,
    3: 0.561,
}


# ============================================================
# HELPERS
# ============================================================

def safe_auc(y_true, predictions):
    """Return ROC-AUC or NaN when both classes are not present."""
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, predictions)


def safe_pr_auc(y_true, predictions):
    """Return PR-AUC or NaN when both classes are not present."""
    if len(np.unique(y_true)) < 2:
        return np.nan

    return average_precision_score(y_true, predictions)


def prepare_fold_features(train_df, test_df, features):
    """
    Fit all scalers on training data only, then transform train/test.
    Construct polynomial terms after scaling.

    Returns:
        train_df, test_df
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    for feat in features:
        scaler = StandardScaler()

        scaler.fit(train_df[[feat]])

        train_df[f"{feat}_z"] = scaler.transform(train_df[[feat]])
        test_df[f"{feat}_z"] = scaler.transform(test_df[[feat]])

    # Polynomial terms constructed in z-space
    for feat in [
        "distance_to_fault",
        "distance_to_lithology_contact",
    ]:
        train_df[f"{feat}_z_sq"] = train_df[f"{feat}_z"] ** 2
        test_df[f"{feat}_z_sq"] = test_df[f"{feat}_z"] ** 2

    return train_df, test_df


def get_valid_rocks(train_df, y_train, all_rock_features):
    """
    Target-aware rock filtering performed ONLY on the training fold.
    """

    valid_rocks = []

    for col in all_rock_features:

        positive_present = (
            (train_df[col] == 1) &
            (y_train == 1)
        ).sum()

        negative_present = (
            (train_df[col] == 1) &
            (y_train == 0)
        ).sum()

        if positive_present > 0 and negative_present > 0:
            valid_rocks.append(col)

    return valid_rocks


# ============================================================
# MAIN
# ============================================================

def run_phase3_ablation(n_folds=N_FOLDS):

    print("--- 1. Loading Dataset ---")

    df = pd.read_csv(DATA_PATH)

    lithology_col = "litho_contact_litho_class"

    continuous_features = [
        "distance_to_fault",
        "distance_to_lithology_contact",
        "bouguer",
    ]

    required_columns = [
        "centroid_x",
        "centroid_y",
        "deposit_present",
        lithology_col,
    ] + continuous_features

    df = df.dropna(subset=required_columns).copy()

    # --------------------------------------------------------
    # Spatial folds
    # --------------------------------------------------------

    df["spatial_block"] = get_along_belt_folds(
        df,
        n_folds=n_folds
    )

    # --------------------------------------------------------
    # Lithology dummies
    #
    # Do NOT filter target-aware here.
    # Filtering happens separately inside each training fold.
    # --------------------------------------------------------

    df = pd.get_dummies(
        df,
        columns=[lithology_col],
        drop_first=True,
        dtype=float,
    )

    all_rock_features = [
        col
        for col in df.columns
        if col.startswith(f"{lithology_col}_")
    ]

    print(f"Rows: {len(df)}")
    print(f"Deposits: {int(df['deposit_present'].sum())}")
    print(f"Spatial folds: {n_folds}")

    # ========================================================
    # MODEL DEFINITIONS
    # ========================================================

    model_features = {

        # Reference
        "M0_Reference": [],

        # Single-feature models
        "M1_Gravity": [
            "bouguer_z"
        ],

        "M2_Fault": [
            "distance_to_fault_z",
            "distance_to_fault_z_sq",
        ],

        "M3_Lithology": [
            "distance_to_lithology_contact_z",
            "distance_to_lithology_contact_z_sq",
        ],

        # Two-feature models
        "M4_Grav_Fault": [
            "bouguer_z",
            "distance_to_fault_z",
            "distance_to_fault_z_sq",
        ],

        "M5_Grav_Lith": [
            "bouguer_z",
            "distance_to_lithology_contact_z",
            "distance_to_lithology_contact_z_sq",
        ],

        "M6_Fault_Lith": [
            "distance_to_fault_z",
            "distance_to_fault_z_sq",
            "distance_to_lithology_contact_z",
            "distance_to_lithology_contact_z_sq",
        ],
    }

    fold_records = []

    # ========================================================
    # LOOP OVER SPATIAL FOLDS
    # ========================================================

    print("\n--- 2. Executing M0-M7 Spatial Ablation ---")

    for fold in range(n_folds):

        print("\n" + "=" * 60)
        print(f"SPATIAL FOLD {fold + 1} OF {n_folds}")
        print("=" * 60)

        train_df = df[df["spatial_block"] != fold].copy()
        test_df = df[df["spatial_block"] == fold].copy()

        y_train = train_df["deposit_present"].astype(int).values
        y_test = test_df["deposit_present"].astype(int).values

        n_train_deposits = int(y_train.sum())
        n_test_deposits = int(y_test.sum())

        print(
            f"Train: {len(train_df)} cells | "
            f"{n_train_deposits} deposits"
        )

        print(
            f"Test:  {len(test_df)} cells | "
            f"{n_test_deposits} deposits"
        )

        if len(np.unique(y_train)) < 2:
            print("Skipping fold: training set has only one class.")
            continue

        if len(np.unique(y_test)) < 2:
            print("Skipping fold: test set has only one class.")
            continue

        # ----------------------------------------------------
        # Training-only scaling
        # ----------------------------------------------------

        train_df, test_df = prepare_fold_features(
            train_df,
            test_df,
            continuous_features
        )

        # ----------------------------------------------------
        # Training-only rock feature selection
        # ----------------------------------------------------

        valid_rocks = get_valid_rocks(
            train_df,
            y_train,
            all_rock_features
        )

        print(
            f"Valid rock features in training fold: "
            f"{len(valid_rocks)}"
        )

        # ----------------------------------------------------
        # M7 = global nonlinear model + valid rocks
        # ----------------------------------------------------

        model_features["M7_All_Global"] = [
            "bouguer_z",
            "distance_to_fault_z",
            "distance_to_fault_z_sq",
            "distance_to_lithology_contact_z",
            "distance_to_lithology_contact_z_sq",
        ] + valid_rocks

        # ----------------------------------------------------
        # Fold result record
        # ----------------------------------------------------

        fold_record = {
            "Fold": fold + 1,
            "Train_Cells": len(train_df),
            "Test_Cells": len(test_df),
            "Train_Deposits": n_train_deposits,
            "Test_Deposits": n_test_deposits,
            "Valid_Rock_Features": len(valid_rocks),
        }

        # ====================================================
        # M0: NO-INFORMATION REFERENCE
        # ====================================================

        # Every prediction is tied.
        # ROC-AUC is therefore represented by the conventional
        # no-discrimination value of 0.5.

        fold_record["M0_AUC"] = 0.500
        fold_record["M0_PR_AUC"] = (
            n_test_deposits / len(y_test)
        )

        # ====================================================
        # M1-M7
        # ====================================================

        predictions_by_model = {}

        for model_name, features in model_features.items():

            if model_name == "M0_Reference":
                continue

            print(f"  -> Fitting {model_name}")

            X_train = train_df[features].astype(float).values
            X_test = test_df[features].astype(float).values

            # Standard L2-regularized logistic regression.
            #
            # Used here as a fast predictive benchmark for
            # the non-hierarchical models.
            clf = LogisticRegression(
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )

            clf.fit(X_train, y_train)

            predictions = clf.predict_proba(X_test)[:, 1]

            predictions_by_model[model_name] = predictions

            auc = safe_auc(y_test, predictions)
            pr_auc = safe_pr_auc(y_test, predictions)

            fold_record[f"{model_name}_AUC"] = auc
            fold_record[f"{model_name}_PR_AUC"] = pr_auc

            print(
                f"     AUC={auc:.3f} | "
                f"PR-AUC={pr_auc:.3f}"
            )

        # ====================================================
        # M8: PREVIOUSLY COMPLETED V11
        # ====================================================

        m8_auc = V11_RESULTS[fold]

        fold_record["M8_V11_Hierarchical_AUC"] = m8_auc

        fold_records.append(fold_record)

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(fold_records)

    # --------------------------------------------------------
    # Add ΔAUC relative to M0
    # --------------------------------------------------------

    auc_columns = [
        col
        for col in results_df.columns
        if col.endswith("_AUC")
    ]

    for col in auc_columns:
        model_name = col.replace("_AUC", "")
        results_df[f"{model_name}_DeltaAUC"] = (
            results_df[col] - results_df["M0_AUC"]
        )

    # --------------------------------------------------------
    # Add key comparison:
    # M8 V11 vs M7 global
    # --------------------------------------------------------

    results_df["V11_minus_M7_AUC"] = (
        results_df["M8_V11_Hierarchical_AUC"]
        - results_df["M7_All_Global_AUC"]
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_records = []

    model_auc_columns = [
        "M0_AUC",
        "M1_Gravity_AUC",
        "M2_Fault_AUC",
        "M3_Lithology_AUC",
        "M4_Grav_Fault_AUC",
        "M5_Grav_Lith_AUC",
        "M6_Fault_Lith_AUC",
        "M7_All_Global_AUC",
        "M8_V11_Hierarchical_AUC",
    ]

    for col in model_auc_columns:

        model = col.replace("_AUC", "")

        summary_records.append({
            "Model": model,
            "Mean_AUC": results_df[col].mean(),
            "SD_AUC": results_df[col].std(ddof=1),
            "Min_AUC": results_df[col].min(),
            "Max_AUC": results_df[col].max(),
            "Mean_DeltaAUC": (
                results_df[col] -
                results_df["M0_AUC"]
            ).mean()
        })

    summary_df = pd.DataFrame(summary_records)

    # ========================================================
    # EXPORT
    # ========================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_path = OUTPUT_DIR / "phase3_ablation_fold_results.csv"
    summary_path = OUTPUT_DIR / "phase3_ablation_summary.csv"

    results_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print("\n")
    print("=" * 80)
    print("PHASE 3 — FEATURE ABLATION RESULTS")
    print("=" * 80)

    print("\nAUC BY SPATIAL FOLD")
    print(
        results_df[
            [
                "Fold",
                "M0_AUC",
                "M1_Gravity_AUC",
                "M2_Fault_AUC",
                "M3_Lithology_AUC",
                "M4_Grav_Fault_AUC",
                "M5_Grav_Lith_AUC",
                "M6_Fault_Lith_AUC",
                "M7_All_Global_AUC",
                "M8_V11_Hierarchical_AUC",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nMEAN AUC")
    print(
        summary_df[
            [
                "Model",
                "Mean_AUC",
                "SD_AUC",
                "Mean_DeltaAUC",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nV11 vs M7")
    print(
        results_df[
            [
                "Fold",
                "M7_All_Global_AUC",
                "M8_V11_Hierarchical_AUC",
                "V11_minus_M7_AUC",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\n" + "=" * 80)
    print(f"Saved: {fold_path}")
    print(f"Saved: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_phase3_ablation()