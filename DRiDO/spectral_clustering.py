from functools import reduce
import jax.numpy as jnp
from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph
import time


def affinity_to_laplacian_eigenvectors(affinity_matrix):
    graph_laplacian = csgraph.laplacian(affinity_matrix, normed=False)
    eigenvalues, eigenvectors = np.linalg.eig(graph_laplacian)
    ordering = np.argsort(eigenvalues)
    return eigenvalues[ordering], eigenvectors[ordering]


# Vanilla spectral clustering
def make_knn_graph(theta, k=10, ensure_connected=False):
    def pick_top_k(row, k=7):
        order = np.argpartition(-row, kth=k)
        new_row = row.copy()[order]
        new_row.iloc[k:] = 0
        return new_row

    abs_theta = theta.abs()
    n, m = theta.shape
    k = np.min([k, m - 1])
    k_sparse_abs_theta = abs_theta.apply(pick_top_k, axis=1, k=k)
    k_sparse_abs_theta = (k_sparse_abs_theta + k_sparse_abs_theta.T) / 2
    G = nx.Graph(k_sparse_abs_theta)
    if ensure_connected:
        assert nx.is_connected(G)
    return k_sparse_abs_theta


def run_spectral_clustering(abs_theta, k=20):
    spectral_clustering = SpectralClustering(
        n_clusters=k,
        affinity='precomputed',
        assign_labels='discretize',
        random_state=0).fit(abs_theta.values)

    cluster_df = pd.DataFrame({'var': abs_theta.index,
                               'cluster': spectral_clustering.labels_}).sort_values('cluster')
    return cluster_df


# Eigengap heuristic spectral clustering
def get_eigengap_spectral_clustering(affinity_matrix):
    num_best_clusters = eigengap_heuristic(affinity_matrix, k=10)
    print('Num clusters: ', num_best_clusters)
    clustering = SpectralClustering(
        n_clusters=sorted(num_best_clusters)[0],
        affinity='precomputed',
        assign_labels='discretize',
        random_state=0).fit(affinity_matrix)
    return clustering.labels_


def eigengap_heuristic(affinity_matrix, k=5):
    """Reference: https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
                  http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    min_components = 3
    eigenvalues, _ = affinity_to_laplacian_eigenvectors(affinity_matrix)

    plt.title('Largest eigen values of input matrix')
    plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
    plt.grid()
    index_largest_gap = np.argsort(
        np.diff(eigenvalues)[min_components:])[::-1][:k]
    nb_clusters = index_largest_gap + min_components + 1
    return nb_clusters


# Self-tuning spectral clustering
def generate_Givens_rotation(i, j, theta, size):
    g = jnp.eye(size)
    c = jnp.cos(theta)
    s = jnp.sin(theta)

    g = g.at[i, i].set(0)
    g = g.at[i, j].set(0)
    g = g.at[j, i].set(0)
    g = g.at[j, j].set(0)

    ii_mat = jnp.zeros_like(g)
    ii_mat = ii_mat.at[i, i].set(1)

    jj_mat = jnp.zeros_like(g)
    jj_mat = jj_mat.at[j, j].set(1)

    ji_mat = jnp.zeros_like(g)
    ji_mat = ji_mat.at[j, i].set(1)

    ij_mat = jnp.zeros_like(g)
    ij_mat = ij_mat.at[i, j].set(1)

    return g + c * ii_mat + c * jj_mat + s * ji_mat - s * ij_mat


def make_ij_coords(C):
    """Make i,j coord list so that coords are ordered by max(i, j).
       This ensures that when we warm start, we can just append 0s to the new
       Givens rotations so that we get faster convergence (hopefully)."""
    ij_list = [(i, j) for i in range(C) for j in range(C) if i < j]
    return sorted(ij_list, key=lambda x: (np.max(x), x[0], x[1]))


def generate_U_list(ij_list, theta_list, size):
    """Generate Givens rotations for ij tuples given theta."""
    return [generate_Givens_rotation(ij[0], ij[1], theta, size)
            for ij, theta in zip(ij_list, theta_list)]


def get_rotation_matrix(X, C, initialization=None):
    X = jnp.array(X)
    ij_list = make_ij_coords(C)

    def cost(theta_list):
        U_list = generate_U_list(ij_list, theta_list, C)
        R = reduce(jnp.dot, U_list, jnp.eye(C))
        Z = X.dot(R)
        M = jnp.clip(jnp.max(Z, axis=1, keepdims=True), 1e-9, jnp.inf)
        return jnp.sum((Z / M) ** 2)

    if initialization is not None:
        assert len(initialization) <= int(C * (C - 1) / 2)
        pad_length = int(C * (C - 1) / 2) - len(initialization)
        theta_init = jnp.pad(initialization, (0, pad_length))
    else:
        theta_init = jnp.array([0.0] * int(C * (C - 1) / 2))
    opt = minimize(cost, x0=theta_init, method='BFGS')
    return opt.fun, reduce(jnp.dot, generate_U_list(ij_list, opt.x, C), jnp.eye(C)), opt.x


def get_self_tuning_spectral_clustering(
        affinity, min_n_cluster=20, max_n_cluster=50, use_warm_start=True):
    w, v = affinity_to_laplacian_eigenvectors(affinity)
    order = np.argsort(w)
    w = w[order]
    v = v[order]
    results = []
    prev_result = None
    start_time = time.time()
    for c in range(min_n_cluster, max_n_cluster + 1):
        x = v[:, -c:]
        cost, r, opt_var = get_rotation_matrix(
            x, c, initialization=prev_result)
        if use_warm_start:
            prev_result = opt_var
        results.append((cost, x.dot(r)))
        elapsed = int(time.time() - start_time)
        print('n_cluster: %d \t cost: %f (%d s)' % (c, cost, elapsed))
    cost, z = sorted(results, key=lambda x: x[0])[0]
    return z
