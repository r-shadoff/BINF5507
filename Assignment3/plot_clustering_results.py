from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import numpy as np


def plot_clustering_results(method, X, labels, centers, silhouette_avg, n_clusters=4):
    """
    Plots the clustering results and the silhouette scores for each sample.

    Parameters:
    method: string, name of clustering method used
    X : array-like, shape (n_samples, n_features)
        The input data points.
    labels : array-like, shape (n_samples,)
        The cluster labels for each data point.
    centers : array-like, shape (n_clusters, n_features), optional
        The coordinates of the cluster centers. If None, no centers are plotted.
    silhouette_avg : float
        The average silhouette score for all the samples.
    n_clusters : int, optional, default=4
        The number of clusters.

    Returns:
    None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    # Define a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    cluster_colors = {i: colors[i] for i in range (n_clusters)}
    
    # Plot each cluster separately to ensure legend is accurate
    for i in range(n_clusters):
        ax1.scatter(X[labels == i, 0], X[labels == i, 1], 
                    color=cluster_colors[i], label=f'Cluster {i}', s=50)

    # Plot cluster centers
    if centers is not None:
        ax1.scatter(centers[:, 0], centers[:, 1], c='red', s=200, 
                    alpha=0.75, marker='X', label='Cluster Centers')

    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_title(f'{method} Clustering Result')
    ax1.legend()

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, 
                          ith_cluster_silhouette_values, facecolor=cluster_colors[i], 
                          edgecolor=cluster_colors[i], alpha=1)

        ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax2.set_title(f"{method} Clustering Silhouette Plot")
    ax2.set_xlabel("Silhouette coefficient")
    ax2.set_ylabel("Cluster label")

    ax2.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax2.set_yticks([])
    ax2.set_xticks(np.arange(-0.1, 1.1, 0.2))

    plt.show()