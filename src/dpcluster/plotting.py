import logging
import numpy as np

logger = logging.getLogger(__name__)

# Optional matplotlib dependency
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _check_matplotlib():
    """Verify that matplotlib is installed before plotting."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting functions. "
            "Install it with `pip install matplotlib`."
        )


def plot_decision_graph(model):
    """
    Plot the rho-delta decision graph.
    
    Parameters
    ----------
    model : DensityPeakClustering
        The fitted clustering model containing rho_ and delta_ attributes.
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """
    _check_matplotlib()
    
    if not hasattr(model, 'rho_') or model.rho_ is None:
        raise ValueError("Model is not fitted. Call fit() or fit_predict() first.")
        
    fig, ax = plt.subplots(figsize=(8, 8))
    
    rho = model.rho_
    delta = model.delta_
    max_rho = np.max(rho)
    max_delta = np.max(delta)
    
    # Scatter all points
    ax.scatter(rho, delta, s=50, alpha=0.4, edgecolors='none', c='blue')
    
    # Plot thresholds
    rho_threshold = max_rho * model.rho_cut
    delta_threshold = max_delta * model.delta_cut
    
    x_vals = np.linspace(0, max_rho, 200)
    # Decision boundary: y = grad * (max_delta / max_rho) * rho + delta_cutoff
    cutoff_line = (model.grad * (max_delta / max_rho) * x_vals) + delta_threshold
    
    ax.plot(x_vals, cutoff_line, c='red', linestyle='--', label='Cutoff Line')
    ax.axvline(x=rho_threshold, color='green', linestyle=':', label='Rho Threshold')
    
    # Highlight cluster centers
    if hasattr(model, 'cluster_centers_indices_') and model.cluster_centers_indices_ is not None:
        centers = model.cluster_centers_indices_
        ax.scatter(rho[centers], delta[centers], s=100, c='black', marker='*', label='Cluster Centers')
    
    ax.set_title('Density Peak Decision Graph')
    ax.set_xlabel('Local Density (rho)')
    ax.set_ylabel('Minimum Distance (delta)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_clusters(model, X):
    """
    Plot the clustered data points in 2D or 3D.
    
    Parameters
    ----------
    model : DensityPeakClustering
        The fitted clustering model containing labels_.
    X : array-like of shape (n_samples, n_features)
        The input data coordinates. If >3 dimensions, only the first 3 are plotted.
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """
    _check_matplotlib()
    
    if not hasattr(model, 'labels_') or model.labels_ is None:
        raise ValueError("Model is not fitted. Call fit() or fit_predict() first.")
        
    X = np.asarray(X)
    n_features = X.shape[1]
    
    is_3d = n_features >= 3
    if n_features > 3:
        logger.warning(
            "Input data has %d dimensions. Plotting only the first 3 dimensions.", 
            n_features
        )
        
    fig = plt.figure(figsize=(10, 8))
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
        
    labels = model.labels_
    unique_labels = set(labels)
    
    # Generate colormap for clusters (excluding noise label -1)
    n_clusters = len(unique_labels - {-1})
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, n_clusters)))
    color_idx = 0
    
    for label in unique_labels:
        mask = (labels == label)
        
        if label == -1:
            c = 'grey'
            alpha = 0.3
            marker = 's'
            s = 30
            label_name = 'Noise'
        else:
            c = [colors[color_idx]]
            color_idx += 1
            alpha = 0.6
            marker = 'o'
            s = 50
            label_name = f'Cluster {label}'
            
        pts = X[mask]
        
        if is_3d:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, alpha=alpha, marker=marker, s=s, label=label_name)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], c=c, alpha=alpha, marker=marker, s=s, label=label_name)
            
    # Plot cluster centers prominently
    if hasattr(model, 'cluster_centers_indices_') and model.cluster_centers_indices_ is not None:
        centers_idx = model.cluster_centers_indices_
        centers = X[centers_idx]
        if is_3d:
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', marker='*', s=300, edgecolor='white')
        else:
            ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='*', s=300, edgecolor='white')
            
    ax.set_title('Density Peak Clustering Results')
    
    if is_3d:
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
    else:
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        
    # Legend formatting
    if len(unique_labels) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
    return fig, ax
