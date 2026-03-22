import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

def run_baseline():
    print("--- 1. Loading & Cleaning Data ---")
    df = pd.read_csv("C:/Users/vanmu/copperbelt-mineral-prospectivity/data/copperbelt_dataset_clean.csv")
    df = df.dropna(subset=['id', 'distance_to_tract_boundary', 'centroid_x', 'centroid_y'])

    # 1. SPLIT FIRST to prevent leakage
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print("\n--- 2. Dynamic Spatial Feature Engineering ---")
    # Extract the coordinates of the KNOWN deposits from the TRAINING set only
    train_positives = train_df[train_df['deposit_present'] == 1][['centroid_x', 'centroid_y']].values

    def get_honest_distances(target_coords, known_coords):
        """Calculates distance to nearest deposit, masking exact 0.0 matches to avoid cheating."""
        dists = cdist(target_coords, known_coords, metric='euclidean')
        # Preventing the model from learning "distance = 0 means deposit"
        dists[dists == 0.0] = np.inf
        return dists.min(axis=1)

    # Compute distances for Train and Test
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['honest_dist_to_deposit'] = get_honest_distances(
        train_df[['centroid_x', 'centroid_y']].values, train_positives
    )
    test_df['honest_dist_to_deposit'] = get_honest_distances(
        test_df[['centroid_x', 'centroid_y']].values, train_positives
    )

    print("\n--- 3. Training Logistic Regression ---")
    features = ['honest_dist_to_deposit', 'distance_to_tract_boundary']
    
    X_train = train_df[features]
    y_train = train_df['deposit_present']
    X_test = test_df[features]
    y_test = test_df['deposit_present']

    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("\n--- 4. Rigorous Baseline Results ---")
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_baseline()