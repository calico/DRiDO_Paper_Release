import numpy as np
import pandas as pd
from DRiDO.covariance_utils import normalize_diag


def average_affinity_matrix(
    network,
    clustering,
    feature_key="var",
    cluster_key="cluster",
    should_remove_diag=True,
):
    """Summarize cluster connectivity by averaging the outgoing edges
    of the precision matrix. This tends to preserve sparsity better
    but does not in general result in a valid correlation matrix
    even if the original network is a correlation matrix.
    """

    cluster_ids = (
        clustering.set_index(feature_key)
        .loc[network.columns.values][cluster_key]
        .values
    )
    reduced_affinity = network.groupby(cluster_ids).mean()
    reduced_affinity = reduced_affinity.groupby(cluster_ids, axis=1).mean()
    if should_remove_diag:
        reduced_affinity = reduced_affinity - \
            np.diag(np.diag(reduced_affinity.values))
    return reduced_affinity


def summarize_affinity_matrix(
    network, clustering, feature_key="var", cluster_key="cluster",
    exemplars=None
):
    """Summarize cluster connectivity by picking exemplar from cluster
    with largest average correlation with other cluster vars.

    Alternatively, we can average the outgoing edges of the precision matrix
    which would also capture the connectivity of the network, but this
    has the advantage that it gives us a proper covariance matrix.

    You can optionally include a pre-selected list of exemplars as a dictionary
    mapping from cluster names to exemplars. Otherwise, exemplars are selectedd
    by maximizing the average correlation to other features in the cluster.
    """
    network = network.copy()
    network = normalize_diag(network)
    cluster_ids = (
        clustering.set_index(
            feature_key).loc[network.columns.values][cluster_key].values
    )

    def _pick_exemplar(group_network):
        group_vars = group_network.index
        average_within_group_network = group_network[group_vars].abs().mean()
        return group_vars[np.argmax(average_within_group_network)]

    if exemplars is None:
        exemplars = {}
    
    cluster2exemplars = pd.Series(
        exemplars.values(), index=exemplars.keys())
    valid_mask = np.in1d(cluster2exemplars.index, cluster_ids)
    cluster2exemplars = cluster2exemplars[valid_mask]
    auto_selected_cluster2exemplars = network.groupby(cluster_ids).apply(_pick_exemplar)
    cluster2exemplars = cluster2exemplars.combine_first(auto_selected_cluster2exemplars)
    
    exemplars = cluster2exemplars.values
    exemplar_network_matrix = network.loc[exemplars].loc[:, exemplars]
    exemplar_network_matrix.index = cluster2exemplars.index
    exemplar_network_matrix.columns = cluster2exemplars.index
    return exemplar_network_matrix, cluster2exemplars


def get_var_cluster_connectivity(
        network, clustering, var, feature_key="var", cluster_key="cluster"
):
    cluster_ids = (
        clustering.set_index(
            feature_key).loc[network.columns.values][cluster_key].values
    )
    sub_clustering = network.loc[var]
    return sub_clustering.groupby(cluster_ids).max()
