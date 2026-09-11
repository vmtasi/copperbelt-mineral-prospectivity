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


def run_sensitivity_analysis():

    print("--- 1. Loading Data & Preparing Folds ---")

    data_path = ROOT / 'data' / 'copperbelt_training_v5_with_tectonic_domain.csv'
    df = pd.read_csv(data_path)

    features = [
        'distance_to_fault',
        'distance_to_lithology_contact',
        'bouguer'
    ]

    df = df.dropna(
        subset=[
            'centroid_x',
            'centroid_y',
            'domain',
            'litho_contact_litho_class'
        ] + features
    ).copy()

    df['spatial_block'] = get_along_belt_folds(df, n_folds=4)

    n_iterations = 500

    # Initialize a seeded RNG for strict reproducibility
    rng = np.random.default_rng(42)

    axis_a_records = []
    axis_b_records = []

    print(
        f"--- 2. Executing Sensitivity Analysis "
        f"(Iterations = {n_iterations}) ---"
    )

    for fold in range(4):

        train_idx = df['spatial_block'] != fold
        test_idx = df['spatial_block'] == fold

        train_full = df[train_idx].copy()
        test_full = df[test_idx].copy()

        # ============================================================
        # M5 STANDARDIZATION
        # Scaling is fitted strictly on the training data.
        # ============================================================

        for feat in features:

            scaler = StandardScaler().fit(train_full[[feat]])

            train_full[f'{feat}_z'] = scaler.transform(
                train_full[[feat]]
            )

            test_full[f'{feat}_z'] = scaler.transform(
                test_full[[feat]]
            )

        # Non-linear lithology-distance term
        for feat in ['distance_to_lithology_contact']:

            train_full[f'{feat}_z_sq'] = (
                train_full[f'{feat}_z'] ** 2
            )

            test_full[f'{feat}_z_sq'] = (
                test_full[f'{feat}_z'] ** 2
            )

        feats_m5 = [
            'bouguer_z',
            'distance_to_lithology_contact_z',
            'distance_to_lithology_contact_z_sq'
        ]

        # ============================================================
        # SPLIT POSITIVE AND NEGATIVE OBSERVATIONS
        # ============================================================

        train_pos = train_full[
            train_full['deposit_present'] == 1
        ]

        train_neg = train_full[
            train_full['deposit_present'] == 0
        ]

        test_pos = test_full[
            test_full['deposit_present'] == 1
        ]

        test_neg = test_full[
            test_full['deposit_present'] == 0
        ]

        max_test_pos = len(test_pos)

        if max_test_pos == 0:
            continue

        # ============================================================
        # FULL-TRAINING AUC ANCHOR
        #
        # This is the observed performance when the model receives
        # the complete available training region.
        # ============================================================

        clf_full = LogisticRegression(
            max_iter=1000,
            random_state=42
        ).fit(
            train_full[feats_m5].values,
            train_full['deposit_present'].values
        )

        full_preds = clf_full.predict_proba(
            test_full[feats_m5].values
        )[:, 1]

        full_auc = roc_auc_score(
            test_full['deposit_present'].values,
            full_preds
        )

        # ============================================================
        # AXIS A: TRAINING-INFORMATION SENSITIVITY
        #
        # Tests whether regional predictive performance improves
        # as the number of positive training examples increases.
        #
        # The TEST REGION remains completely fixed.
        # ============================================================

        # Candidate numbers of positive deposits used for training.
        # The final value is always the actual maximum available
        # positive training sample for the current fold.
        candidate_steps_a = [
            20,
            40,
            60,
            80,
            100,
            120,
            140,
            len(train_pos)
        ]

        train_steps = sorted(
            set(
                s for s in candidate_steps_a
                if 10 <= s <= len(train_pos)
            )
        )

        # Preserve the original positive:negative ratio of the
        # complete training region at each training size.
        ratio_neg_pos = len(train_neg) / len(train_pos)

        X_test_full = test_full[feats_m5].values
        y_test_full = test_full['deposit_present'].values

        fold_a_record = {
            'Test_Fold': f"Fold_{fold + 1}",
            'Observed_Full_Training_Set_AUC': round(full_auc, 3)
        }

        for size in train_steps:

            step_aucs = []

            # Maintain approximately the same class ratio as the
            # complete training set.
            n_neg = min(
                int(round(size * ratio_neg_pos)),
                len(train_neg)
            )

            for _ in range(n_iterations):

                # Repeated SUBSAMPLING, not bootstrap sampling.
                # Observations are sampled without replacement.
                pos_idx = rng.choice(
                    train_pos.index,
                    size=size,
                    replace=False
                )

                neg_idx = rng.choice(
                    train_neg.index,
                    size=n_neg,
                    replace=False
                )

                boot_train = pd.concat([
                    train_pos.loc[pos_idx],
                    train_neg.loc[neg_idx]
                ])

                clf = LogisticRegression(
                    max_iter=1000
                ).fit(
                    boot_train[feats_m5].values,
                    boot_train['deposit_present'].values
                )

                preds = clf.predict_proba(
                    X_test_full
                )[:, 1]

                step_aucs.append(
                    roc_auc_score(
                        y_test_full,
                        preds
                    )
                )

            fold_a_record[
                f'AUC_nTrain={size}'
            ] = (
                f"{np.median(step_aucs):.3f} "
                f"[{np.percentile(step_aucs, 2.5):.3f}, "
                f"{np.percentile(step_aucs, 97.5):.3f}]"
            )

        axis_a_records.append(fold_a_record)

        # ============================================================
        # AXIS B: EVALUATION-SAMPLE SENSITIVITY
        #
        # Tests whether the observed regional AUC is sensitive to
        # the number of positive deposits available for evaluation.
        #
        # The TRAINED MODEL remains completely fixed.
        # ============================================================

        candidate_steps_b = [
            max_test_pos,
            50,
            40,
            30,
            20,
            10
        ]

        test_steps = sorted(
            set(
                s for s in candidate_steps_b
                if 10 <= s <= max_test_pos
            ),
            reverse=True
        )

        fold_b_record = {
            'Test_Fold': f"Fold_{fold + 1}",
            'Observed_Full_Test_AUC': round(full_auc, 3)
        }

        for size in test_steps:

            step_aucs = []

            for _ in range(n_iterations):

                # Repeatedly subsample positive deposits from the
                # fixed test region. Test negatives remain fixed.
                pos_idx = rng.choice(
                    test_pos.index,
                    size=size,
                    replace=False
                )

                test_subset = pd.concat([
                    test_pos.loc[pos_idx],
                    test_neg
                ])

                preds = clf_full.predict_proba(
                    test_subset[feats_m5].values
                )[:, 1]

                step_aucs.append(
                    roc_auc_score(
                        test_subset['deposit_present'].values,
                        preds
                    )
                )

            fold_b_record[
                f'AUC_nTest={size}'
            ] = (
                f"{np.median(step_aucs):.3f} "
                f"[{np.percentile(step_aucs, 2.5):.3f}, "
                f"{np.percentile(step_aucs, 97.5):.3f}]"
            )

        axis_b_records.append(fold_b_record)

    # ================================================================
    # SAVE RESULTS
    # ================================================================

    df_axis_a = pd.DataFrame(axis_a_records)
    df_axis_b = pd.DataFrame(axis_b_records)

    output_dir = ROOT / 'figures' / 'audit'
    os.makedirs(output_dir, exist_ok=True)

    df_axis_a.to_csv(
        output_dir / 'phase7_6_axis_a_training_info.csv',
        index=False
    )

    df_axis_b.to_csv(
        output_dir / 'phase7_6_axis_b_test_positives.csv',
        index=False
    )

    # ================================================================
    # PRINT RESULTS
    # ================================================================

    print(
        "\n=============================================================================================================================="
    )

    print(
        "AXIS A: TRAINING-INFORMATION SENSITIVITY "
        "(Fixed Test Region, Varying Training Size)"
    )

    print(
        "=============================================================================================================================="
    )

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(
        df_axis_a.to_string(index=False)
    )

    print(
        "\n=============================================================================================================================="
    )

    print(
        "AXIS B: EVALUATION-SAMPLE SENSITIVITY "
        "(Fixed Training Model, Varying Test Positives)"
    )

    print(
        "=============================================================================================================================="
    )

    print(
        df_axis_b.to_string(index=False)
    )

    print(
        "=============================================================================================================================="
    )

    print(
        f"\n[+] Sensitivity matrices saved to {output_dir}"
    )


if __name__ == "__main__":
    run_sensitivity_analysis()