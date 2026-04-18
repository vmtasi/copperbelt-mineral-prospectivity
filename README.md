# Copperbelt Mineral Prospectivity Mapping: A Bayesian Spatial Approach

## Project Overview
This project builds a **physics-informed Machine Learning pipeline** to identify **Greenfield (unexplored) copper deposits** in the Central African Copperbelt.

Rather than relying on standard black-box algorithms and random cross-validation—which notoriously inflate performance metrics in spatial data due to autocorrelation—this project utilizes **Bayesian Logistic Regression via Markov Chain Monte Carlo (MCMC)** paired with strict **Spatial Block Cross-Validation**.

The core narrative of this project is the **rigorous evolution of its spatial features**, documenting the transition from naive geometric proximity to true geological proxies in order to eliminate data leakage and build a commercially viable exploration tool.


## Feature Engineering

Building a model capable of predicting mineral deposits in completely unexplored frontier zones requires systematically stripping away all hindsight bias.

This pipeline evolved through **three distinct phases** to achieve true geological intelligence.


### Phase 1: The Proximal Bias Trap

The initial iteration of the model utilized the **Euclidean distance from a grid cell to the nearest known copper deposit**.

Creating a classic **data leakage**.

In a true Greenfield exploration scenario, the locations of existing deposits are fundamentally unknown.

Because the model required the coordinates of old mines to predict new ones, it suffered from **proximal bias** and was therefore unviable for frontier discovery.

This feature was entirely **scrapped to preserve mathematical integrity**.

---

### Phase 2: Overfitting to Human Constructs

To resolve the leakage, the model shifted to measuring the **distance to the edge of regional geological tracts**.

However, when subjected to **K-Means Spatial Block Cross-Validation**—where the model was trained on the southern regions of the basin and forced to predict the unseen northern region—the performance collapsed to an **AUC of 0.14**.

Tract boundaries are often **arbitrary, human-drawn polygons**.

The model had learned a spatial rule in the south that was physically meaningless in the north, exposing that it was **overfit to local geography rather than universal geology**.

---

### Phase 3: The True Geological Proxy

The final, production-ready model relies entirely on **exogenous geological proxies**.

Using **QGIS spatial engineering**, grid cells were mapped to continuous lithological polygons to extract:

- the **primary host rock**
- the **exact distance to the nearest lithological contact zone**

A lithological contact is the physical boundary where two distinct rock layers meet and where fluid chemistry and mineralization pathways often change.

To prevent the Bayesian MCMC sampler from becoming unstable when encountering unfamiliar rock types in frontier blocks, **L2 regularization** was introduced via tightened normal priors:

$$
\mathcal{N}(0,1)
$$

Furthermore, **exact spatial coordinates were strictly removed** from the training features to prevent the model from memorizing the map.

The result is a **stable, non-leaky model** capable of predicting copper mineralization in completely unseen geographic blocks based purely on the physical plumbing system of the earth.

---

## Methodology & Tech Stack

### Spatial Data Engineering
- **:contentReference[oaicite:0]{index=0}**
- vector joins
- polygon intersections
- polygon-to-line conversion
- centroid / polygon-to-line nearest distance computation
- lithological contact extraction
- spatial block generation

---

### Probabilistic Modeling
- **:contentReference[oaicite:1]{index=1}**
- **:contentReference[oaicite:2]{index=2}**
- Bayesian Logistic Regression
- **NUTS (No-U-Turn Sampler)**
- posterior predictive inference

---

### Validation Framework
- **:contentReference[oaicite:3]{index=3}**
- KMeans spatial clustering
- strict geographic train-test separation
- spatial block cross-validation
- AUC / ROC evaluation

---

## Conclusion

This project demonstrates that in **geoscientific machine learning**, the **physical reality of a feature** and the **prevention of spatial data leakage** matter infinitely more than algorithmic complexity.

A modest AUC achieved through **strict spatial blocking** and **pristine geological contact features** provides far more value for real-world drilling capital deployment than a near-perfect score driven by proximal bias.

The central lesson is clear:

> **geological truth beats statistical illusion**
