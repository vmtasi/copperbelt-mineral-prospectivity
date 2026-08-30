from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIGURES = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)

CSV = DATA / 'copperbelt_training_v5.csv'
if not CSV.exists():
    raise SystemExit(f"Data file not found: {CSV}")

df = pd.read_csv(CSV)

from src.validation_strategies import (
    get_random_folds,
    get_kmeans_spatial_folds,
    get_along_belt_folds,
)

strategies = [
    ('random', get_random_folds),
    ('kmeans', get_kmeans_spatial_folds),
    ('along_belt', get_along_belt_folds),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

for ax, (name, fn) in zip(axes, strategies):
    blocks = fn(df, n_folds=4)
    df_plot = df.copy()
    df_plot['block'] = blocks

    # marker size from Mining_count (add 1 to avoid zero)
    sizes = (np.sqrt(df_plot['Mining_count'].fillna(0) + 1) * 6).clip(6, 80)

    sc = ax.scatter(
        df_plot['centroid_x'], df_plot['centroid_y'],
        c=df_plot['block'], cmap='tab10', s=sizes,
        alpha=0.7, edgecolor='k', linewidth=0.2
    )

    # highlight deposits with black X
    deposits = df_plot[df_plot['deposit_present'] == 1]
    if not deposits.empty:
        ax.scatter(deposits['centroid_x'], deposits['centroid_y'],
                   facecolors='none', edgecolors='black', marker='X', s=80, label='Deposit')

    ax.set_title(f"{name} folds (size ~ Mining_count)")
    ax.set_aspect('equal')
    ax.set_xlabel('centroid_x')
    ax.set_ylabel('centroid_y')

# single colorbar for fold index is not strictly meaningful; add legend for deposits
axes[2].legend(loc='upper right')

out_path = FIGURES / 'mining_locations_by_fold_strategies.png'
plt.suptitle('Mining locations and folds (marker size ~ Mining_count)')
plt.savefig(out_path, dpi=200)
plt.close()

print(f"Saved plot: {out_path}")
