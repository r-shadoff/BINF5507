from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import numpy as np

def plot_clusters(method, X, labels, centers=None):
    """
    Plots clustering results.
    
    Parameters:
    method: clustering method used
    X: array like, shape (n_samples, n_features). Input data points.
    labels: array like, shape (n_samples)
    centers: array like, shape (n_clusters, n_features)
   
    Returns: 
    None
    """
    plt.figure(figsize=(8,8))
    
    # Scatter plot of clusters
    plt.scatter(X[:, 0], c=labels, s=50, cmap='viridis')
    
    if centers is not None:
        plt.scatter(centers[:,0], c='red', s=200, alpha=0.75, marker='X', label='Centers')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{method} Clustering Result')
    plt.legend()
    plt.show()

def plot_silhouette(method, X, labels, silhouette_avg, n_clusters=4):
    """
    Plots the silhouette scores for each sample.

    Parameters:
    method: method used to generate clusters and their silhouette score.
    X : array-like, shape (n_samples, n_features)
    labels : array-like, shape (n_samples,)
    silhouette_avg : float, The average silhouette score for all the samples.
    n_clusters : int, optional, default=4
    
    Returns:
    None
    """
    plt.figure(figsize=(8, 8))

    # Compute silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, labels)
    
    y_lower = 10
    for i in range(n_clusters):
        # Extract and sort silhouette scores for cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, 
                          facecolor=color, edgecolor=color, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # Adjust y_lower for the next cluster

    plt.title(f"Silhouette Plot for {method} Clustering")
    plt.xlabel("Silhouette Coefficient Values")
    plt.ylabel("Cluster Label")

    # Draw a vertical red line for the average silhouette score
    plt.axvline(x=silhouette_avg, color="red", linestyle="--", label="Avg Silhouette")
    plt.legend()

    plt.yticks([])
    plt.xticks(np.arange(-0.1, 1.1, 0.2))
    plt.show()