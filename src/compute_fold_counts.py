from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'

CSV = DATA / 'copperbelt_training_v13_litho_contact.csv'
if not CSV.exists():
    raise SystemExit(f"Required data file not found: {CSV}")

df = pd.read_csv(CSV)
MODEL_COLUMNS = [
    'centroid_x',
    'centroid_y',
    'litho_contact_litho_class',
    'domain',
    'distance_to_fault',
    'distance_to_lithology_contact',
    'bouguer',
    'deposit_present',
]
df = df.dropna(subset=MODEL_COLUMNS).copy()

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
        deposits = int(grp['deposit_present'].sum())
        deposit_rate = deposits / cells if cells else float('nan')
        rows.append({
            'Fold': i + 1,
            'Cells': cells,
            'Deposits': deposits,
            'Deposit_rate': deposit_rate,
        })
    return pd.DataFrame(rows)


def main(n_folds=4):
    outputs = {}
    for name, fn in STRATEGIES:
        blocks = fn(df, n_folds=n_folds)
        summary_df = summarize_folds(df, blocks, n_folds=n_folds)
        outputs[name] = summary_df

    # Print combined summary â€” no files are written
    for name, table in outputs.items():
        print(f"\n=== Strategy: {name} ===")
        print(table.to_string(index=False))


if __name__ == '__main__':
    main(n_folds=4)

