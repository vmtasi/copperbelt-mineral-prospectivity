import pandas as pd
from pathlib import Path

from src.validation_strategies import (
    get_along_belt_folds,
    get_kmeans_spatial_folds,
    get_random_folds,
    validate_fold_counts,
)
from src.spatial_cv import run_spatial_cv


def compare_validation_strategies():
    ROOT = Path(__file__).resolve().parents[1]
    data_root = ROOT / 'data'
    df = pd.read_csv(data_root / 'copperbelt_training_v5.csv')

    strategies = [
        ('random', get_random_folds),
        ('kmeans', get_kmeans_spatial_folds),
        ('along_belt', get_along_belt_folds),
    ]

    results = []
    for name, strategy in strategies:
        print(f"\n=== Running validation strategy: {name} ===")
        spatial_blocks = strategy(df, n_folds=4)
        fold_summary = validate_fold_counts(df, spatial_blocks, n_folds=4)
        print(fold_summary)

        metrics = run_spatial_cv(
            df=df,
            spatial_blocks=spatial_blocks,
            n_folds=4,
            output_prefix=f"{name}_",
            plot_folds=True,
        )

        results.append({
            'strategy': name,
            'mean_auc': metrics['mean_auc'],
            'pr_auc': metrics['pr_auc'],
            'recall_at_5': metrics['recall_at_5'],
            'n_cells': metrics['n_cells'],
            'total_deposits': metrics['total_deposits'],
            'calibration_path': metrics['calibration_path'],
            'qgis_output_path': metrics['qgis_output_path'],
            'posterior_csv': metrics['posterior_csv'],
        })

    summary_df = pd.DataFrame(results)
    summary_path = data_root / 'validation_strategy_comparison.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved validation strategy comparison: {summary_path}")


if __name__ == "__main__":
    compare_validation_strategies()
