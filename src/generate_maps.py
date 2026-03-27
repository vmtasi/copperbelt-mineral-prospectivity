import os
import pytensor
pytensor.config.compilerdir = "C:/pytensor_cache"

import pandas as pd 
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

def get_distances(target_coords, known_coords):
    dists = cdist(target_coords, known_coords, metric='euclidean')
    dists[dists==0.0] = np.inf
    return dists.min(axis=1)

def generate_exploration_maps():
    print("--- 1. Loading Data & Engineering Features ---")
    df = pd.read_csv("C:/Users/vanmu/copperbelt-mineral-prospectivity/data/copperbelt_dataset_clean.csv")
    df =df.dropna(subset=['id', 'distance_to_tract_boundary', 'centroid_x', 'centroid_y'])

    # Get all known deposits to calculate honest distances
    all_positives = df[df['deposit_present'] == 1][['centroid_x', 'centroid_y']].values

    # Calculate distance to nearest deposit (preventing 0.0 leakage)
    df['dist_to_deposit'] = get_distances(df[['centroid_x', 'centroid_y']].values, all_positives)

    features = ['dist_to_deposit', 'distance_to_tract_boundary']

    # Scale the entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    y_obs_data = df['deposit_present'].values

    print("\n--- 2. Running Fast MCMC Sampler ---")
    with pm.Model() as prospectivity_model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=X_scaled.shape[1])
        
        mu = alpha + pm.math.dot(X_scaled, beta)
        p = pm.math.invlogit(mu)
        
        y_obs = pm.Bernoulli('y_obs', p=p, observed=y_obs_data)
        
        # Using your lightning-fast C++ compiled settings
        trace = pm.sample(draws=1000, tune=1000, cores=1, target_accept=0.9, progressbar=True)

    print("\n--- 3. Calculating Grid Probabilities & Uncertainty ---")

    alpha_samples = trace.posterior['alpha'].values.flatten()
    beta_samples = trace.posterior['beta'].values.reshape(-1,2)

    #Probs for all 1900+ cells
    mean_probs = []
    uncertainties = []

    for i in range(len(X_scaled)):
        cell_features = X_scaled[i]
        cell_probs = 1 / (1 + np.exp(-(alpha_samples + np.dot(beta_samples, cell_features))))

        mean_probs.append(np.mean(cell_probs))
        uncertainties.append(np.std(cell_probs))

    df['mean_probability'] = mean_probs
    df['uncertainty'] = uncertainties

    print("\n--- 4. Generating High-Res Exploration Maps ---")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))

    # Known deposit coordinates \
    deposits_x = df[df['deposit_present'] == 1]['centroid_x']
    deposits_y = df[df['deposit_present'] == 1]['centroid_y']
    # MAP 1: PROSPECTIVITY (Mean Probability)
    sc1 = ax1.scatter(df['centroid_x'], df['centroid_y'], c=df['mean_probability'], 
                      cmap='viridis', s=50, alpha=0.8, edgecolors='none')
    ax1.scatter(deposits_x, deposits_y, c='red', marker='*', s=100, label='Known Deposits', edgecolors='black')
    ax1.set_title('Bayesian Mineral Prospectivity (Probability)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Longitude / Centroid X')
    ax1.set_ylabel('Latitude / Centroid Y')
    ax1.legend(loc='upper right')
    plt.colorbar(sc1, ax=ax1, label='Probability of Deposit')

    # MAP 2: UNCERTAINTY (Standard Deviation)
    sc2 = ax2.scatter(df['centroid_x'], df['centroid_y'], c=df['uncertainty'], 
                      cmap='plasma', s=50, alpha=0.8, edgecolors='none')
    ax2.scatter(deposits_x, deposits_y, c='white', marker='*', s=100, label='Known Deposits', edgecolors='black')
    ax2.set_title('Model Uncertainty (Standard Deviation)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Longitude / Centroid X')
    ax2.set_ylabel('Latitude / Centroid Y')
    ax2.legend(loc='upper right')
    plt.colorbar(sc2, ax=ax2, label='Uncertainty (Std Dev)')

    # Save and show
    plt.tight_layout()
    plt.savefig("C:/Users/vanmu/copperbelt-mineral-prospectivity/bayesian_exploration_maps.png", dpi=300)
    print("Maps successfully saved to your project folder as 'bayesian_exploration_maps.png'!")
    # Save the final predictions back to CSV
    df.to_csv("C:/Users/vanmu/copperbelt-mineral-prospectivity/data/copperbelt_predictions_for_GIS.csv", index=False)
    plt.show()

if __name__ == "__main__":
    generate_exploration_maps()