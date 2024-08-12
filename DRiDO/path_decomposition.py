import networkx as nx
import numpy as np
import pandas as pd


def submatrix(mat, i):
    return np.delete(np.delete(mat, i, 0), i, 1)


def path_subscore(pcorr, path):
    # Reference: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04542-5
    src = path[0]
    dst = path[-1]
    A = -pcorr
    np.fill_diagonal(A, 1)
    A_i_det = np.linalg.det(submatrix(A, src))
    A_j_det = np.linalg.det(submatrix(A, dst))
    mask = ~np.in1d(np.arange(A.shape[0]), path)
    A_sub_det = np.linalg.det(A[mask][:, mask])

    edges = list(zip(path[:-1], path[1:]))
    tau = np.prod([pcorr[edge] for edge in edges])

    return tau * A_sub_det / np.sqrt(A_i_det * A_j_det)


def compute_all_paths_scores(adj_mat, src, dst, threshold=0, max_path_len=5):
    adj_mat = adj_mat.copy()
    adj_mat[np.abs(adj_mat) < threshold] = 0
    G = nx.Graph(adj_mat)
    paths = list(nx.all_simple_paths(
        G, source=src, target=dst, cutoff=max_path_len))
    print('Computing scores for ', len(paths), ' paths')
    path_scores = [path_subscore(adj_mat, path) for path in paths]
    path_scores_df = pd.DataFrame({'path': paths, 'score': path_scores})
    path_scores_df['normalized_score'] = (
        path_scores_df['score'].abs() / path_scores_df['score'].abs().sum())
    return path_scores_df


def generate_path_matrix(path_scores, node_labels=None):
    edge_weights = dict()
    for _, row in path_scores.iterrows():
        path = row['path']
        if node_labels is not None:
            path = [node_labels[i] for i in path]
        normalized_score = row['normalized_score']
        edges = list(zip(path[:-1], path[1:]))
        for edge in edges:
            if edge in edge_weights:
                edge_weights[edge] += normalized_score
            else:
                edge_weights[edge] = normalized_score
    edge_weights = [(edge[0], edge[1], {'weight': value})
                    for edge, value in edge_weights.items()]
    return nx.from_edgelist(edge_weights, create_using=nx.DiGraph()), edge_weights


def path_decomposition(pcorr_df, src, dst, edge_threshold=0, max_path_len=5):
    src_idx = np.where(pcorr_df.columns == src)[0][0]
    dst_idx = np.where(pcorr_df.columns == dst)[0][0]
    path_subscores = compute_all_paths_scores(
        pcorr_df.values, src_idx, dst_idx, threshold=edge_threshold,
        max_path_len=max_path_len)
    path_subscores['readable_path'] = path_subscores['path'].apply(
        lambda x: ' → '.join([pcorr_df.columns[i] for i in x]))
    return path_subscores
