# Density Peak Clustering (dpcluster)

A Python module for performing Density Peak Clustering. 

Based on the paper: "Clustering by fast search and find of density peaks"
Alex Rodriguez and Alessandro Laio, Science ,2014, 344(6191), 1492-1496
DOI: 10.1126/science.1242072

This was written for clustering data from MD simulations. For the actual clustering a series of phi and psi dihedral angles were converted to their cos and sin, then a PCA was run to reduce to three dimensions. This module was therefore coded up to work exclusively with 3D data. It should still work with 2D data just by adding a zero. Input data should be an array of length 3 arrays.

The clustering returns the clusters as a list of indices corresponding to the input data. Cluster centres are also returned, as are the indices for the unclustered points.

## Installation

```bash
pip install -e .
```

## Usage

```python
from dpcluster import Densitycluster
from dpcluster.plotting import plot_decision_graph, plot_clusters_3d

# Assuming 'data' is your dataset (e.g., NumPy array or list of lists)
model = Densitycluster(data)
model.build_dist_matrix()
model.assign_rho_delta(cutoff_dist=1.5)
plot_decision_graph(model)

model.run_clustering()
plot_clusters_3d(model)
```
