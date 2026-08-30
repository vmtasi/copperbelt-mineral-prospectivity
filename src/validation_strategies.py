import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def get_random_folds(df, n_folds=4, random_state=42):
    """Assign random cross-validation folds to each row."""
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(df))
    fold_sizes = np.full(n_folds, len(df) // n_folds)
    fold_sizes[: len(df) % n_folds] += 1

    spatial_blocks = np.empty(len(df), dtype=int)
    start = 0
    for fold in range(n_folds):
        stop = start + fold_sizes[fold]
        spatial_blocks[order[start:stop]] = fold
        start = stop

    return spatial_blocks


def get_kmeans_spatial_folds(df, n_folds=4, random_state=42):
    """Assign spatial folds using KMeans clustering on centroids."""
    coords = df[['centroid_x', 'centroid_y']].values
    kmeans = KMeans(n_clusters=n_folds, random_state=random_state, n_init=10)
    return kmeans.fit_predict(coords)


def get_along_belt_folds(df, n_folds=4):
    """Assign contiguous folds along the principal axis of the Copperbelt."""
    coords = df[['centroid_x', 'centroid_y']].values
    pca = PCA(n_components=1)
    projection = pca.fit_transform(coords).flatten()

    order = np.argsort(projection)
    spatial_blocks = np.empty(len(df), dtype=int)
    fold_sizes = np.full(n_folds, len(df) // n_folds)
    fold_sizes[: len(df) % n_folds] += 1

    start = 0
    for fold in range(n_folds):
        stop = start + fold_sizes[fold]
        spatial_blocks[order[start:stop]] = fold
        start = stop

    return spatial_blocks


def plot_spatial_folds(df, spatial_blocks, output_path, title=None):
    """Save a scatter plot showing spatial fold assignment on centroid coordinates."""
    plt.figure(figsize=(8, 8))
    plt.scatter(df.centroid_x, df.centroid_y, c=spatial_blocks, cmap='tab10', s=8)
    plt.axis('equal')
    plt.title(title or 'Spatial Folds')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def validate_fold_counts(df, spatial_blocks, n_folds=4):
    """Return the number of rows and deposit counts per fold."""
    return pd.DataFrame(
        {
            'fold': np.arange(n_folds),
            'n_cells': [np.sum(spatial_blocks == fold) for fold in range(n_folds)],
            'n_deposits': [np.sum(df.loc[spatial_blocks == fold, 'deposit_present'] == 1) for fold in range(n_folds)],
        }
    )
