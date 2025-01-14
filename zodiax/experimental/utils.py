from jax import numpy as np, Array
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def cov_to_corr(cov: Array) -> Array:
    """
    Calculate the correlation matrix from the covariance matrix.
    """

    # Calculate the standard deviations
    stds = np.sqrt(np.diag(cov))

    # Calculate the correlation matrix
    return cov / np.outer(stds, stds)


def get_group_inds(corr: Array) -> Array:
    """
    Group indices of correlated variables.
    """
    # Compute the distance between each pair of variables
    d = sch.distance.pdist(corr)  # vector of ('55' choose 2) pairwise distances

    # Perform hierarchical/agglomerative clustering
    L = sch.linkage(d, method="complete")

    # Given a distance threshold, return indices of clusters
    # i.e. array of cluster indices corresponding to each variable
    ind = sch.fcluster(L, 0.5 * d.max(), "distance")

    return ind


def plot_corr_matrix(cov: Array, param_labels: list):
    """
    Plot the sorted correlation matrix for a given covariance matrix.

    Args:
    cov: Array
        Covariance matrix.
    params: list
        List of parameter names (strings).
    """

    # Convert covariance matrix to correlation matrix
    corr = cov_to_corr(cov)

    # Sorting
    ind = get_group_inds(corr)
    args = np.argsort(ind)
    sorted_labels = [param_labels[i] for i in args]

    # Plotting
    plt.imshow(corr[args, :][:, args], origin="upper")
    plt.xticks(range(len(param_labels)), sorted_labels, rotation=45)
    plt.yticks(range(len(param_labels)), sorted_labels)
    plt.colorbar()
    plt.show()
