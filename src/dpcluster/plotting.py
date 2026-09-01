import matplotlib.pyplot as plt
import numpy as np

def plot_decision_graph(cluster_obj):
    """Plot the rho-delta decision graph."""
    if cluster_obj.rho is None or cluster_obj.delta is None:
        raise ValueError("rho and delta not assigned. Call assign_rho_delta() first.")
    
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.scatter(cluster_obj.rho, cluster_obj.delta, s=100, alpha=0.3, edgecolors='none')
    
    x_points = np.linspace(0, max(cluster_obj.rho), 200)           
    y_points = [cluster_obj.cutoff(x) for x in x_points]
    
    ax1.plot(x_points, y_points, c='green')
    ax1.plot([cluster_obj.rho_cutoff, cluster_obj.rho_cutoff], [0, max(cluster_obj.delta)], c='green')

    ax1.grid(True)
    ax1.set_xlabel('rho') 
    ax1.set_ylabel('delta')
    plt.xlim(-max(cluster_obj.rho) / 10, max(cluster_obj.rho) + 1)
    plt.ylim(-max(cluster_obj.delta) / 10, max(cluster_obj.delta) + 1)
    plt.show()

def plot_clusters_3d(cluster_obj):
    """Plot the 3D clustered data."""
    if cluster_obj.clusters is None:
        raise ValueError("Clusters not assigned. Call run_clustering() first.")
        
    colours = plt.cm.rainbow([x for x in np.linspace(0, 1, len(cluster_obj.clusters))])
    
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    for c, col in zip(cluster_obj.clusters, colours):
        pts = np.array([cluster_obj.data[c1] for c1 in c])
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[col], s=100, alpha=0.2)
    
    # Excluded points
    if len(cluster_obj.excluded) > 0:
        ex_pts = np.array([cluster_obj.data[ex] for ex in cluster_obj.excluded])
        ax.scatter(ex_pts[:, 0], ex_pts[:, 1], ex_pts[:, 2], marker='s', c='grey', s=100, alpha=0.5)
        
    # Cluster centers
    cen_pts = np.array([cluster_obj.data[cen] for cen in cluster_obj.cluster_centre_idx])
    ax.scatter(cen_pts[:, 0], cen_pts[:, 1], cen_pts[:, 2], marker='*', c='black', s=1000)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()
