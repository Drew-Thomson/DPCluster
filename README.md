# Density Peak Clustering (dpcluster)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn compatible](https://img.shields.io/badge/scikit--learn-compatible-F7931E.svg)](https://scikit-learn.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://docs.pytest.org/)

A high-performance, vectorized implementation of **Density Peak Clustering** designed for structural biology, molecular dynamics (MD) trajectories, and general-purpose N-dimensional data analysis.

Based on the algorithm described in:
> **"Clustering by fast search and find of density peaks"**
> Alex Rodriguez and Alessandro Laio. *Science*, 2014, 344(6191), 1492-1496. 
> DOI: [10.1126/science.1242072](https://doi.org/10.1126/science.1242072)

This is a modified version of code originally used in my research group to cluster transformed Ramachandran coordinates for cyclic peptides, but the algorithm will work for any data. It has been tested for dimensions up to 4D, but in principle should work for higher.

---

## 🧬 Overview
Density Peak Clustering intuitively defines cluster centers as points that are surrounded by neighbors with lower local density and are at a relatively large distance from any points with a higher local density.

Unlike traditional implementations restricted to 2D or 3D datasets, `dpcluster` is built to handle **N-dimensional** conformations (e.g., flattened distance matrices, PCA-reduced dihedral angles) out-of-the-box.

## ✨ Key Engineering Features
- **Vectorized Performance:** Heavy calculations are vectorized via NumPy. The original $O(N^2)$ assignment bottleneck has been completely replaced with a faster $O(N)$ lookup step.
- **Scikit-Learn API:** Inherits from `BaseEstimator` and `ClusterMixin`. This makes the model fully plug-and-play with the scikit-learn ecosystem.
- **Defaults:** If left unspecified, the cutoff distance (`cutoff_dist`) is dynamically auto-estimated using the 2nd percentile of the pairwise distance matrix as recommended in the original publication.
- **Decoupled Visualization:** The plotting module dynamically adapts to 2D or 3D data. Matplotlib is an optional dependency, ensuring the core package is safe for headless HPC cluster execution.

---

## 📦 Installation

To install the core package:
```bash
pip install -e .
```

To enable visualization tools, use the optional `matplotlib` dependency:
```bash
pip install matplotlib
```

---

## 🚀 Quickstart

The API mirrors standard scikit-learn conventions (`.fit()`, `.predict()`, `.fit_predict()`):

```python
import numpy as np
from dpcluster import DensityPeakClustering
from dpcluster.plotting import plot_decision_graph, plot_clusters

# 1. Load your N-dimensional data
X = np.random.rand(500, 10) 

# 2. Initialize the model 
# (cutoff_dist is auto-estimated if set to None)
model = DensityPeakClustering(cutoff_dist=None, delta_cut=0.1, rho_cut=0.1)

# 3. Fit the model and extract cluster labels (-1 indicates noise)
labels = model.fit_predict(X)

# 4. Optional: Visualize the results
# The plotting module automatically generates 2D or 3D scatter plots depending on the data
fig1, ax1 = plot_decision_graph(model)
fig2, ax2 = plot_clusters(model, X)
```

---

## 🧪 Testing & Validation

This project maintains strict test coverage for algorithmic correctness and edge-case handling. The test suite uses synthetic datasets and scikit-learn's **Adjusted Rand Index (ARI)** to validate clustering accuracy against ground-truth labels.

To run the test suite:
```bash
pip install pytest
pytest tests/
```
