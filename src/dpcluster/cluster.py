import time
import numpy as np
import scipy.spatial.distance

class Densitycluster:
    def __init__(self, data):
        self.data = np.array(data)
        self.idx_list = list(range(len(self.data)))
        self.distances = None
        self.rho = None
        self.delta = None
        self.cluster_centre_idx = None
        self.clusters = None
        self.grad = None
        self.delta_cutoff = None
        self.rho_cutoff = None
        self.cut_dist = None

    def _dens(self, dist, cutoff_dist):
        return np.exp(-(dist / cutoff_dist))

    def build_dist_matrix(self, method='euclidean'):
        t1 = time.time()
        self.distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(self.data, metric=method))
        t2 = time.time()
        print(f"distance matrix built in {t2 - t1:.4f} seconds")

    def assign_rho_delta(self, cutoff_dist, delta_cut=0.1, rho_cut=0.1, grad=0.0, kernel='gaussian'):
        t1 = time.time()
        self.grad = grad
        self.cut_dist = cutoff_dist
        
        if kernel == 'cutoff':
            self.rho = [sum([i < cutoff_dist for i in j]) for j in self.distances]
        elif kernel == 'gaussian':
            self.rho = []
            for i in range(len(self.data)):
                self.rho.append(sum([self._dens(d, cutoff_dist) for d in self.distances[i]]))

        self.delta = []
        for i in range(len(self.data)):
            # Find distances to points with higher density
            higher_rho_dist = [d for d, r in zip(self.distances[i], self.rho) if r > self.rho[i]]
            if higher_rho_dist:
                self.delta.append(min(higher_rho_dist))
            else:
                self.delta.append(max(self.distances[i]))
        
        self.delta_cutoff = max(self.delta) * delta_cut
        self.rho_cutoff = max(self.rho) * rho_cut
        
        t2 = time.time()
        print(f"delta and rho assigned in {t2 - t1:.4f} seconds")

    def cutoff(self, xval):
        return self.grad * max(self.delta) * xval / max(self.rho) + self.delta_cutoff

    def run_clustering(self):
        t2 = time.time()

        rho_delta_array = np.array([self.rho, self.delta]).T
        
        # Mask 1: Points above the cutoff line (decision graph)
        mask1 = np.array([i[1] > self.cutoff(i[0]) for i in rho_delta_array])
        # Mask 2: Points with rho above rho_cutoff
        mask2 = np.array([i > self.rho_cutoff for i in self.rho])
        
        mask3 = np.logical_and(mask1, mask2)
        self.cluster_centre_idx = np.array(self.idx_list)[mask3]
        
        unclustered = np.array(self.idx_list)[np.logical_and(mask2, np.logical_not(mask1))]
        self.excluded = np.array(self.idx_list)[np.logical_and(np.logical_not(mask2), np.array([i > self.delta_cutoff for i in self.delta]))]
        
        print(f"there were {len(unclustered)} points to cluster")
        print(f"there were {len(self.excluded)} points excluded")

        self.pairs = []
        for unc in unclustered:
            # Find neighbour with higher density, sorted by distance
            # Get indices sorted by distance
            neighbours = np.argsort(self.distances[unc])
            hd_neighbours = [x for x in neighbours if self.rho[x] > self.rho[unc]]
            if hd_neighbours:
                self.pairs.append((unc, hd_neighbours[0]))
            else:
                # Should not happen for non-cluster-center points if density is defined
                pass

        self.clusters = [[x] for x in self.cluster_centre_idx]
        
        count = 0
        while count < len(self.pairs):
            for i, pair in enumerate(self.pairs):
                if pair[0] is None: continue
                for j, cluster in enumerate(self.clusters):
                    if pair[1] in cluster:
                        self.clusters[j].append(pair[0])
                        count += 1
                        self.pairs[i] = (None, None)
                        break

        t3 = time.time()
        print(f"clusters assigned in {t3 - t2:.4f} seconds")
        print(f"found {len(self.clusters)} clusters")
        print(f"clusters contain {[len(x) for x in self.clusters]} entries")
