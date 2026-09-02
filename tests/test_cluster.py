import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist, squareform
from dpcluster.cluster import DensityPeakClustering

def test_density_peak_recovery():
    """Test standard execution with synthetic blobs and check ARI."""
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    model = DensityPeakClustering(cutoff_dist=1.0, delta_cut=0.2, rho_cut=0.2)
    labels = model.fit_predict(X)
    
    assert len(model.cluster_centers_indices_) == 3
    assert labels.shape == (300,)
    
    # Check that clustering accuracy is high using Adjusted Rand Index
    ari = adjusted_rand_score(y, labels)
    assert ari > 0.95

def test_auto_cutoff_dist():
    """Test that cutoff_dist is automatically estimated if None."""
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)
    model = DensityPeakClustering(cutoff_dist=None, delta_cut=0.2, rho_cut=0.2)
    model.fit(X)
    
    assert hasattr(model, 'cutoff_dist_')
    assert model.cutoff_dist_ > 0

def test_kernels():
    """Test that both gaussian and cutoff kernels execute properly."""
    X, _ = make_blobs(n_samples=50, centers=2, random_state=42)
    
    model_gaussian = DensityPeakClustering(kernel="gaussian", cutoff_dist=1.0)
    labels_g = model_gaussian.fit_predict(X)
    assert len(set(labels_g)) > 0
    
    model_cutoff = DensityPeakClustering(kernel="cutoff", cutoff_dist=1.0)
    labels_c = model_cutoff.fit_predict(X)
    assert len(set(labels_c)) > 0

def test_precomputed_metric():
    """Test using a precomputed distance matrix."""
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)
    distances = squareform(pdist(X, metric="euclidean"))
    
    model = DensityPeakClustering(metric="precomputed", cutoff_dist=1.0)
    labels = model.fit_predict(distances)
    
    ari = adjusted_rand_score(y, labels)
    assert ari > 0.95

def test_noise_handling():
    """Test that strict cutoffs label points as noise (-1)."""
    X, _ = make_blobs(n_samples=50, centers=1, random_state=42)
    # Very high thresholds so no point qualifies as a center
    model = DensityPeakClustering(delta_cut=1.1, rho_cut=1.1)
    labels = model.fit_predict(X)
    
    assert len(model.cluster_centers_indices_) == 0
    assert np.all(labels == -1)

def test_input_validation():
    """Test error handling for bad inputs."""
    model = DensityPeakClustering(cutoff_dist=1.0)
    
    # 1D array should fail
    with pytest.raises(ValueError):
        model.fit(np.array([1, 2, 3]))
        
    # Less than 2 samples should fail
    with pytest.raises(ValueError):
        model.fit(np.array([[1, 2]]))
