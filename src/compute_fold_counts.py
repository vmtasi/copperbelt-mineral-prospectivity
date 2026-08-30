from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

CSV = DATA / 'copperbelt_training_v5.csv'
if not CSV.exists():
    raise SystemExit(f"Required data file not found: {CSV}")

df = pd.read_csv(CSV)

from src.validation_strategies import (
    get_random_folds,
    get_kmeans_spatial_folds,
    get_along_belt_folds,
)

STRATEGIES = [
    ('random', get_random_folds),
    ('kmeans', get_kmeans_spatial_folds),
    ('along_belt', get_along_belt_folds),
]

def summarize_folds(df, blocks, n_folds=4):
    df2 = df.copy()
    df2['spatial_block'] = blocks
    rows = []
    for i in range(n_folds):
        grp = df2[df2['spatial_block'] == i]
        cells = int(len(grp))
        deposits = int(grp['deposit_present'].sum()) if 'deposit_present' in grp.columns else int(grp.get('deposit', pd.Series()).sum())
        # Use exact column name 'Mining_count' as requested
        if 'Mining_count' not in grp.columns:
            raise SystemExit("Required column 'Mining_count' not found in CSV")
        mining_count = float(grp['Mining_count'].sum())
        deposit_density = deposits / mining_count if mining_count != 0 else float('nan')
        rows.append({
            'Fold': i + 1,
            'Cells': cells,
            'Deposits': deposits,
            'Mining_count': mining_count,
            'Deposit_density': deposit_density,
        })
    return pd.DataFrame(rows)


def main(n_folds=4):
    outputs = {}
    for name, fn in STRATEGIES:
        blocks = fn(df, n_folds=n_folds)
        summary_df = summarize_folds(df, blocks, n_folds=n_folds)
        outputs[name] = summary_df

    # Print combined summary — no files are written
    for name, table in outputs.items():
        print(f"\n=== Strategy: {name} ===")
        print(table.to_string(index=False))


if __name__ == '__main__':
    main(n_folds=4)
