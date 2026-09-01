import pytest
import numpy as np
import random
from dpcluster.cluster import Densitycluster

def test_clustering_basic():
    # Generate fake data
    centres = [[1, 0, 4], [-2, 5, -3], [6, -2, 0], [-5, 4, -3]]
    sigmas = [[0.5, 1, 0.1], [0.3, 2, 1], [0.5, 0.5, 0.5], [1, 0.1, 0.5]]
    
    fake_data = []
    for _ in range(1000):
        index = random.randint(0, 3)
        entry = [random.gauss(centres[index][x], sigmas[index][x]) for x in range(len(centres[0]))]
        fake_data.append(entry)
    
    # Run clustering
    model = Densitycluster(fake_data)
    model.build_dist_matrix()
    model.assign_rho_delta(cutoff_dist=1.5, delta_cut=0.1, rho_cut=0.1)
    model.run_clustering()
    
    # Assertions
    assert model.clusters is not None
    assert len(model.clusters) > 0
    assert len(model.cluster_centre_idx) > 0
