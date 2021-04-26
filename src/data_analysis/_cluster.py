import hdbscan
from sklearn.cluster import *
from scipy.spatial.distance import cdist
import numpy as np


def _exemplars(cluster_id, condensed_tree):
    raw_tree = condensed_tree._raw_tree
    # Just the cluster elements of the tree, excluding singleton points
    cluster_tree = raw_tree[raw_tree['child_size'] > 1]
    # Get the leaf cluster nodes under the cluster we are considering
    leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster_id)
    # Now collect up the last remaining points of each leaf cluster (the heart of the leaf)
    result = np.array([])
    for leaf in leaves:
        max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
        points = raw_tree['child'][(raw_tree['parent'] == leaf) &
                                   (raw_tree['lambda_val'] == max_lambda)]
        result = np.hstack((result, points))
    return result.astype(np.int)


def _min_dist_to_exemplar(point, cluster_exemplars, data):
    dists = cdist([data[point]], data[cluster_exemplars.astype(np.int32)])
    return dists.min()


def _dist_vector(point, exemplar_dict, data):
    result = {}
    for cluster in exemplar_dict:
        result[cluster] = _min_dist_to_exemplar(point, exemplar_dict[cluster], data)
    return np.array(list(result.values()))


def _dist_membership_vector(point, exemplar_dict, data, softmax=False):
    if softmax:
        result = np.exp(1. / _dist_vector(point, exemplar_dict, data))
        result[~np.isfinite(result)] = np.finfo(np.double).max
    else:
        result = 1. / _dist_vector(point, exemplar_dict, data)
        result[~np.isfinite(result)] = np.finfo(np.double).max
    result /= result.sum()
    return result


def k_means(X, n_clusters=10):
    """
    K-Means
    """
    km = KMeans(n_clusters=n_clusters)
    y = km.fit_predict(X)

    return y


def dbscan(X, eps=0.35, min_samples=4):
    """
    DBSCAN
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    y = db.fit_predict(X)

    return y


def hdbscan_method(X, eps=0.1, min_samples=4):
    """
    HDBSCAN
    """
    clusterer = hdbscan.HDBSCAN(cluster_selection_epsilon=eps, min_cluster_size=min_samples)
    clusterer.fit(X)

    tree = clusterer.condensed_tree_
    exemplar_dict = {c: _exemplars(c, tree) for c in tree._select_clusters()}

    y = []

    for x in range(X.shape[0]):
        membership_vector = _dist_membership_vector(x, exemplar_dict, X)
        y.append(np.argmax(membership_vector))

    return y


def agglomerative(X, n_clusters=10, linkage='ward'):
    """
    Hierarchical Clustering
    """
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    y = agg.fit_predict(X)

    return y
