"""Fast, vectorized implementation of Density Peak Clustering (Rodriguez & Laio, 2014)
compatible with scikit-learn conventions.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted

logger = logging.getLogger(__name__)


class DensityPeakClustering(BaseEstimator, ClusterMixin):
    """Clustering by fast search and find of density peaks.

    Parameters
    ----------
    cutoff_dist : float, default=None
        Cutoff distance (d_c) for kernel density estimation. 
        If None, automatically estimated as the 2nd percentile of pairwise distances.
    delta_cut : float, default=0.1
        Fractional threshold for minimum distance delta to select cluster peaks.
    rho_cut : float, default=0.1
        Fractional threshold for local density rho to select cluster peaks.
    grad : float, default=0.0
        Slope of the decision line separating cluster centers.
    metric : str, default='euclidean'
        Metric supported by `scipy.spatial.distance.pdist`.
    kernel : {'gaussian', 'cutoff'}, default='gaussian'
        Kernel type used for local density calculation.
    """

    def __init__(
        self,
        cutoff_dist: Optional[float] = None,
        delta_cut: float = 0.1,
        rho_cut: float = 0.1,
        grad: float = 0.0,
        metric: str = "euclidean",
        kernel: Literal["gaussian", "cutoff"] = "gaussian",
    ) -> None:
        self.cutoff_dist = cutoff_dist
        self.delta_cut = delta_cut
        self.rho_cut = rho_cut
        self.grad = grad
        self.metric = metric
        self.kernel = kernel

    def _compute_density(self, distances: np.ndarray, cutoff_dist: float) -> np.ndarray:
        """Vectorized computation of local density rho."""
        if self.kernel == "cutoff":
            return np.sum(distances < cutoff_dist, axis=1) - 1.0  # Exclude self
        elif self.kernel == "gaussian":
            # Continuous Gaussian kernel
            return np.sum(np.exp(-((distances / cutoff_dist) ** 2)), axis=1)
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_delta(self, distances: np.ndarray, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized computation of delta and nearest higher density neighbors."""
        n_samples = distances.shape[0]
        delta = np.zeros(n_samples, dtype=np.float64)
        nearest_higher_indices = np.full(n_samples, -1, dtype=int)

        # Order points by density descending
        ord_rho = np.argsort(-rho)

        # Point with maximum density has delta = max distance to any other point
        max_idx = ord_rho[0]
        delta[max_idx] = np.max(distances[max_idx])
        nearest_higher_indices[max_idx] = -1

        # Vectorized lookup for remaining points
        for i in range(1, n_samples):
            curr_pt = ord_rho[i]
            higher_pts = ord_rho[:i]
            
            # Distances to all points with higher density
            dists_to_higher = distances[curr_pt, higher_pts]
            
            min_dist_idx = np.argmin(dists_to_higher)
            delta[curr_pt] = dists_to_higher[min_dist_idx]
            nearest_higher_indices[curr_pt] = higher_pts[min_dist_idx]

        return delta, nearest_higher_indices

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> DensityPeakClustering:
        """Fit the clustering model to input coordinates or precomputed distances."""
        # 1. Validation and Setup
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        
        n_samples = X.shape[0]
        if n_samples < 2:
            raise ValueError("Clustering requires at least 2 data points.")

        # 2. Pairwise Distances
        if self.metric == "precomputed":
            self.distances_ = X
        else:
            self.distances_ = squareform(pdist(X, metric=self.metric))

        # Determine cutoff distance if not provided
        if self.cutoff_dist is None:
            self.cutoff_dist_ = np.percentile(self.distances_[self.distances_ > 0], 2)
            logger.info("Auto-estimated cutoff_dist: %f", self.cutoff_dist_)
        else:
            self.cutoff_dist_ = self.cutoff_dist

        # 3. Vectorized Rho & Delta
        self.rho_ = self._compute_density(self.distances_, self.cutoff_dist_)
        self.delta_, self.nearest_higher_indices_ = self._compute_delta(self.distances_, self.rho_)

        # 4. Peak Identification (Decision Line)
        max_rho = np.max(self.rho_)
        max_delta = np.max(self.delta_)

        rho_threshold = max_rho * self.rho_cut
        delta_threshold = max_delta * self.delta_cut

        # Linear decision boundary: delta > (grad * (max_delta / max_rho) * rho + delta_cutoff)
        cutoff_line = (self.grad * (max_delta / max_rho) * self.rho_) + delta_threshold
        is_center = (self.delta_ > cutoff_line) & (self.rho_ > rho_threshold)
        self.cluster_centers_indices_ = np.flatnonzero(is_center)

        n_clusters = len(self.cluster_centers_indices_)
        logger.info("Found %d cluster centers.", n_clusters)

        self.labels_ = np.full(n_samples, -1, dtype=int)

        if n_clusters == 0:
            logger.warning("No cluster centers met the cutoff criteria. All points marked as noise.")
            return self

        # 5. Assignment via Fast O(N) Lookup
        # Every non-center point takes the cluster label of its closest higher-density neighbor
        for cluster_id, center_idx in enumerate(self.cluster_centers_indices_):
            self.labels_[center_idx] = cluster_id

        # Iterate in descending density order to ensure parent labels are assigned before children
        ord_rho = np.argsort(-self.rho_)
        for idx in ord_rho:
            if self.labels_[idx] == -1:
                nearest_higher = self.nearest_higher_indices_[idx]
                if nearest_higher != -1:
                    self.labels_[idx] = self.labels_[nearest_higher]

        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the model and return cluster labels."""
        self.fit(X, y)
        return self.labels_
