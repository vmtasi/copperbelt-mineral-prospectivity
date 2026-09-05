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

DATA_PATH = (
    ROOT
    / "data"
    / "copperbelt_training_v5_with_tectonic_domain.csv"
)

OUTPUT_DIR = ROOT / "figures" / "audit"

N_FOLDS = 4


# ============================================================
# HELPERS
# ============================================================

def safe_auc(y_true, y_pred):
    """
    ROC-AUC is undefined if only one class is present.
    Return NaN rather than forcing a value.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, y_pred)


def safe_pr_auc(y_true, y_pred):
    """
    PR-AUC is undefined if only one class is present.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    return average_precision_score(y_true, y_pred)


def map_daly_domain(value):
    """
    Map raw domain labels to the six Daly modelling domains.
    """

    x = str(value).lower()

    if "3a" in x:
        return "NRB_3a"
    elif "3b" in x:
        return "NRB_3b"
    elif "crz" in x:
        return "CRZ"
    elif "srb" in x:
        return "SRB"
    elif "nkb" in x:
        return "NKB"
    elif "mmsb" in x:
        return "MMSB"
    else:
        return "Unknown"


def prepare_fold_data(train_df, test_df, continuous_features):
    """
    Fit scalers on training data only and apply them to train/test.
    Create quadratic terms after scaling.
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    for feat in continuous_features:

        scaler = StandardScaler()

        scaler.fit(train_df[[feat]])

        train_df[f"{feat}_z"] = scaler.transform(
            train_df[[feat]]
        )

        test_df[f"{feat}_z"] = scaler.transform(
            test_df[[feat]]
        )

    # Quadratic terms in standardized space
    for feat in [
        "distance_to_fault",
        "distance_to_lithology_contact",
    ]:

        train_df[f"{feat}_z_sq"] = (
            train_df[f"{feat}_z"] ** 2
        )

        test_df[f"{feat}_z_sq"] = (
            test_df[f"{feat}_z"] ** 2
        )

    return train_df, test_df


def get_valid_rocks(train_df, y_train, all_rock_features):
    """
    Target-aware rock selection using TRAINING data only.
    """

    valid_rocks = []

    for col in all_rock_features:

        positive_count = (
            (train_df[col] == 1) &
            (y_train == 1)
        ).sum()

        negative_count = (
            (train_df[col] == 1) &
            (y_train == 0)
        ).sum()

        if positive_count > 0 and negative_count > 0:
            valid_rocks.append(col)

    return valid_rocks


# ============================================================
# MAIN
# ============================================================

def run_phase4_domain_analysis(n_folds=N_FOLDS):

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
        "domain",
        lithology_col,
    ] + continuous_features

    df = df.dropna(
        subset=required_columns
    ).copy()

    # --------------------------------------------------------
    # Daly domain mapping
    # --------------------------------------------------------

    df["daly_domain"] = df["domain"].apply(
        map_daly_domain
    )

    df = df[
        df["daly_domain"] != "Unknown"
    ].copy()

    # --------------------------------------------------------
    # Spatial folds
    # --------------------------------------------------------

    df["spatial_block"] = get_along_belt_folds(
        df,
        n_folds=n_folds
    )

    # --------------------------------------------------------
    # Lithology dummies
    # --------------------------------------------------------

    df = pd.get_dummies(
        df,
        columns=[lithology_col],
        drop_first=True,
        dtype=float,
    )

    all_rock_features = [
        c
        for c in df.columns
        if c.startswith(
            f"{lithology_col}_"
        )
    ]

    unique_domains = sorted(
        df["daly_domain"].unique()
    )

    print(f"Rows: {len(df)}")
    print(
        f"Deposits: "
        f"{int(df['deposit_present'].sum())}"
    )
    print(
        f"Daly domains: "
        f"{', '.join(unique_domains)}"
    )

    # ========================================================
    # MODEL DEFINITIONS
    # ========================================================

    model_features = {

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
        "M7_All_Global": []
    }

    # This dictionary will hold genuinely out-of-fold
    # predictions for every observation.
    oof_predictions = {
        model_name: np.full(
            len(df),
            np.nan
        )
        for model_name in model_features
    }

    # Preserve original row indices so we can put the
    # predictions back into the master dataframe.
    df = df.reset_index(drop=True)

    # ========================================================
    # SPATIAL CROSS-VALIDATION
    # ========================================================

    print(
        "\n--- 2. Generating Out-of-Fold Predictions ---"
    )

    for fold in range(n_folds):

        print("\n" + "=" * 60)
        print(
            f"SPATIAL FOLD {fold + 1} OF {n_folds}"
        )
        print("=" * 60)

        train_mask = (
            df["spatial_block"] != fold
        )

        test_mask = (
            df["spatial_block"] == fold
        )

        train_df = df.loc[
            train_mask
        ].copy()

        test_df = df.loc[
            test_mask
        ].copy()

        y_train = (
            train_df["deposit_present"]
            .astype(int)
            .values
        )

        y_test = (
            test_df["deposit_present"]
            .astype(int)
            .values
        )

        print(
            f"Train: {len(train_df)} | "
            f"Deposits: {y_train.sum()}"
        )

        print(
            f"Test:  {len(test_df)} | "
            f"Deposits: {y_test.sum()}"
        )

        if len(np.unique(y_train)) < 2:
            print(
                "Skipping fold: "
                "training set has one class."
            )
            continue

        if len(np.unique(y_test)) < 2:
            print(
                "Skipping fold: "
                "test set has one class."
            )
            continue

        # ----------------------------------------------------
        # Training-only scaling
        # ----------------------------------------------------

        train_df, test_df = prepare_fold_data(
            train_df,
            test_df,
            continuous_features
        )

        # ----------------------------------------------------
        # Training-only rock selection
        # ----------------------------------------------------

        valid_rocks = get_valid_rocks(
            train_df,
            y_train,
            all_rock_features
        )

        # ----------------------------------------------------
        # M7: Global model
        # ----------------------------------------------------

        model_features["M7_All_Global"] = [
            "bouguer_z",
            "distance_to_fault_z",
            "distance_to_fault_z_sq",
            "distance_to_lithology_contact_z",
            "distance_to_lithology_contact_z_sq",
        ] + valid_rocks

        # ----------------------------------------------------
        # Fit M1-M7
        # ----------------------------------------------------

        for model_name, features in model_features.items():

            X_train = (
                train_df[features]
                .astype(float)
                .values
            )

            X_test = (
                test_df[features]
                .astype(float)
                .values
            )

            clf = LogisticRegression(
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )

            clf.fit(
                X_train,
                y_train
            )

            predictions = (
                clf.predict_proba(X_test)[:, 1]
            )

            # Put predictions back at their original
            # dataframe positions.
            oof_predictions[
                model_name
            ][test_df.index] = predictions

            fold_auc = safe_auc(
                y_test,
                predictions
            )

            print(
                f"  {model_name:<20} "
                f"AUC={fold_auc:.3f}"
            )

    # ========================================================
    # ATTACH OOF PREDICTIONS
    # ========================================================

    prediction_df = df[
        [
            "deposit_present",
            "daly_domain",
            "spatial_block",
            "centroid_x",
            "centroid_y",
        ]
    ].copy()

    for model_name, predictions in oof_predictions.items():

        prediction_df[
            f"{model_name}_Prediction"
        ] = predictions

    # ========================================================
    # DOMAIN-LEVEL PERFORMANCE
    # ========================================================

    print(
        "\n--- 3. Evaluating Predictor Utility "
        "Within Daly Domains ---"
    )

    records = []

    for domain in unique_domains:

        domain_mask = (
            prediction_df["daly_domain"]
            == domain
        )

        domain_data = prediction_df.loc[
            domain_mask
        ].copy()

        y_domain = (
            domain_data["deposit_present"]
            .values
        )

        n_cells = len(domain_data)
        n_deposits = int(y_domain.sum())
        n_non_deposits = (
            n_cells - n_deposits
        )

        record = {
            "Domain": domain,
            "Cells": n_cells,
            "Deposits": n_deposits,
            "Non_Deposits": n_non_deposits,
            "Deposit_Rate": (
                n_deposits / n_cells
                if n_cells > 0
                else np.nan
            ),
        }

        for model_name in oof_predictions:

            pred_col = (
                f"{model_name}_Prediction"
            )

            valid_mask = (
                domain_data[pred_col]
                .notna()
            )

            y = (
                domain_data.loc[
                    valid_mask,
                    "deposit_present"
                ].values
            )

            p = (
                domain_data.loc[
                    valid_mask,
                    pred_col
                ].values
            )

            record[
                f"{model_name}_AUC"
            ] = safe_auc(y, p)

            record[
                f"{model_name}_PR_AUC"
            ] = safe_pr_auc(y, p)

        records.append(record)

    domain_results = pd.DataFrame(
        records
    )

    # ========================================================
    # MODEL IMPROVEMENT RELATIVE TO BASELINE
    # ========================================================

    # M1-M7 deltas relative to 0.5 no-discrimination.
    model_names = list(
        oof_predictions.keys()
    )

    for model_name in model_names:

        auc_col = (
            f"{model_name}_AUC"
        )

        domain_results[
            f"{model_name}_DeltaAUC"
        ] = (
            domain_results[auc_col] - 0.5
        )

    # ========================================================
    # PAIRWISE FEATURE COMPARISONS
    # ========================================================

    # Directly useful comparisons for interpretation.

    domain_results[
        "Fault_minus_Gravity_AUC"
    ] = (
        domain_results["M2_Fault_AUC"]
        - domain_results["M1_Gravity_AUC"]
    )

    domain_results[
        "Lithology_minus_Gravity_AUC"
    ] = (
        domain_results["M3_Lithology_AUC"]
        - domain_results["M1_Gravity_AUC"]
    )

    domain_results[
        "Lithology_minus_Fault_AUC"
    ] = (
        domain_results["M3_Lithology_AUC"]
        - domain_results["M2_Fault_AUC"]
    )

    domain_results[
        "GravLith_minus_Global_AUC"
    ] = (
        domain_results["M5_Grav_Lith_AUC"]
        - domain_results["M7_All_Global_AUC"]
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    domain_path = (
        OUTPUT_DIR
        / "phase4_feature_by_daly_domain.csv"
    )

    prediction_path = (
        OUTPUT_DIR
        / "phase4_oof_predictions.csv"
    )

    domain_results.to_csv(
        domain_path,
        index=False
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    # ========================================================
    # TERMINAL DISPLAY
    # ========================================================

    auc_display_cols = [
        "Domain",
        "Cells",
        "Deposits",
        "Non_Deposits",
        "M1_Gravity_AUC",
        "M2_Fault_AUC",
        "M3_Lithology_AUC",
        "M4_Grav_Fault_AUC",
        "M5_Grav_Lith_AUC",
        "M6_Fault_Lith_AUC",
        "M7_All_Global_AUC",
    ]

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PHASE 4 — FEATURE UTILITY BY DALY DOMAIN"
    )

    print(
        "=" * 100
    )

    print(
        domain_results[
            auc_display_cols
        ]
        .round(3)
        .to_string(index=False)
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "KEY PAIRWISE AUC DIFFERENCES"
    )

    print(
        "=" * 100
    )

    comparison_cols = [
        "Domain",
        "Fault_minus_Gravity_AUC",
        "Lithology_minus_Gravity_AUC",
        "Lithology_minus_Fault_AUC",
        "GravLith_minus_Global_AUC",
    ]

    print(
        domain_results[
            comparison_cols
        ]
        .round(3)
        .to_string(index=False)
    )

    print(
        "\nSaved:"
    )
    print(
        domain_path
    )
    print(
        prediction_path
    )


if __name__ == "__main__":
    run_phase4_domain_analysis()