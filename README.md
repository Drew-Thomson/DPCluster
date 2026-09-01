# Density Peak Clustering (dpcluster)

A Python module for performing Density Peak Clustering.

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
