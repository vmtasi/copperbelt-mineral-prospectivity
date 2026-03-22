import os
os.environ["PYTENSOR_FLAGS"] = "cxx="

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

def get_distances(target_coords, known_coords):
    dists = cdist(target_coords, known_coords, metric='euclidean')
    dists[dists==0.0] = np.inf
    return dists.min(axis=1)

def run_bayesian_model():
    print("----1.loading & Preparing Data ---")
    df = pd.read_csv("C:/Users/vanmu/copperbelt-mineral-prospectivity/data/copperbelt_dataset_clean.csv")
    
    # FIXED: Reassigned df so the dropna actually applies
    df = df.dropna(subset=['id', 'distance_to_tract_boundary', 'centroid_x', 'centroid_y'])
    
    train_df, test_df = train_test_split(df, test_size=.2, random_state=42)

    #dynamic spatial
    train_positives = train_df[train_df['deposit_present'] == 1][['centroid_x', 'centroid_y']].values

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['dist_to_deposit'] = get_distances(train_df[['centroid_x', 'centroid_y']].values, train_positives)
    test_df['dist_to_deposit'] = get_distances(test_df[['centroid_x', 'centroid_y']].values, train_positives)

    features = ['dist_to_deposit', 'distance_to_tract_boundary']

    # SCALING for PyMC to converge properly

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[features])
    X_test_scaled = scaler.transform(test_df[features])

    y_train = train_df['deposit_present'].values

    print("\n ---2. Building the Bayesian Logistic Regression ---")

    #Probabilistic model
    with pm.Model() as prospectivity_model:
        #Priors: Assume coeff are normally distributed ~0
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=X_train_scaled.shape[1])

        #The log link Function
        mu = alpha + pm.math.dot(X_train_scaled, beta)
        p = pm.math.invlogit(mu)

        y_obs = pm.Bernoulli('y_obs', p=p, observed=y_train)
        print("\n--- 3. Sampling the Posterior (MCMC) ---")
        
        trace = pm.sample(draws=1000, tune=1000, cores=1, target_accept=0.9, progressbar=True)

    print("\n--- 4. Extracting Uncertainty for the Test Set ---")
    #Extract formulas the MCMC enine generated
    alpha_samples = trace.posterior['alpha'].values.flatten()
    beta_samples = trace.posterior['beta'].values.reshape(-1, 2)

    print("\nPredictions for first 5 test cells (Probability ± Uncertainty):")

    for i in range(5):
        cell_features = X_test_scaled[i]

        #P(deposit) for all 4000 sampled parameter sets
        cell_probs = 1 / (1 + np.exp(-(alpha_samples + np.dot(beta_samples, cell_features))))

        #Mean is prediction, sd is uncertainity
        mean_prob = np.mean(cell_probs)
        uncertainity = np.std(cell_probs)

        actual = test_df.iloc[i]['deposit_present']
        
        # FIXED: Typo in the format string syntax from 3.f to .3f
        print(f"Cell {i+1} | Actual: {actual} | Prediction: {mean_prob:.3f} +/- {uncertainity:.3f}")

if __name__ == "__main__":
    run_bayesian_model()