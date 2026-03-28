import os
import pytensor

pytensor.config.compiledir = "C:/pytensor_cache"

import pandas as pd
import numpy as np
import pymc as pm

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist

def distances(target_coords, known_coords):
    if len(known_coords) == 0:
        return np.full(len(target_coords), np.inf)
    dists = cdist(target_coords, known_coords, metric='euclidean')
    dists[dists==0.0] = np.inf
    return dists.min(axis=1)

def run_spatial_cv():
    print("--- 1. Loading Data & Creating Spatial Blocks ---")
    df = pd.read_csv("C:/Users/vanmu/copperbelt-mineral-prospectivity/data/copperbelt_dataset_clean.csv")
    df = df.dropna(subset=['id', 'distance_to_tract_boundary', 'centroid_x', 'centroid_y'])

    # Slicing the map in 4 geographic regions (North, South, East, West)

    coords = df[['centroid_x', 'centroid_y']].values
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['spatial_block'] = kmeans.fit_predict(coords)

    fold_metrics = []

    print("\n ---2. Starting Spatial Block Cross-Validation ---")
    for fold in range(4):
        print(f"\m Evaluating Block {fold + 1} of 4...")

        train_df = df[df['spatial_block'] != fold].copy()
        test_df = df[df['spatial_block'] == fold].copy()

        # LEAKAGE PREVENTION
        train_positives = train_df[train_df['deposit_present'] == 1][['centroid_x', 'centroid_y']].values
        
        train_df['dist_to_deposit'] = distances(train_df[['centroid_x', 'centroid_y']].values, train_positives)
        test_df['dist_to_deposit'] = distances(test_df[['centroid_x', 'centroid_y']].values, train_positives)
        features = ['dist_to_deposit', 'distance_to_tract_boundary']

        # scale Based on training  block
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[features])
        X_test_scaled = scaler.transform(test_df[features])

        y_train = train_df['deposit_present'].values
        y_test = test_df['deposit_present'].values

        if sum(y_test) ==0:
            print(f"skipping AUC for Block {fold+1}: No deposits in this region to test against")
            continue 

        # Bayesian Sampler
        with pm.Model() as prospectivity_model:
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            beta = pm.Normal('beta', mu=0, sigma=10, shape=X_train_scaled.shape[1])
            mu = alpha + pm.math.dot(X_train_scaled, beta)
            p = pm.math.invlogit(mu)
            y_obs = pm.Bernoulli('y_obs', p=p, observed=y_train)
            trace = pm.sample(draws=1000, tune=1000, cores=1, target_accept=0.9, progressbar=False)

        # Predict on the unseen Block
        alpha_smaples = trace.posterior['alpha'].values.flatten()
        beta_samples = trace.posterior['beta'].values.reshape(-1, 2)

        mean_probs = []

        for i in range(len(X_test_scaled)):
            cell_features = X_test_scaled[i]
            cell_probs = 1 /(1+np.exp(-(alpha_smaples + np.dot(beta_samples, cell_features))))
            mean_probs.append(np.mean(cell_probs))

        # Calculate AUC

        fold_auc = roc_auc_score(y_test, mean_probs)
        fold_metrics.append(np.mean(cell_probs))
        print(f"Block {fold + 1} Spatial AUC: {fold_auc:.3f}")
    print(f"FINAL SPATIAL AUC (Mean): {np.mean(fold_metrics):.3f}")

if __name__ == "__main__":
    run_spatial_cv()