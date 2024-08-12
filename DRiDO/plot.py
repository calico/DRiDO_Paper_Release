import jinja2
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
import numpy as np
import pandas as pd
import scipy.spatial as sp
import scipy.cluster.hierarchy as hc
import seaborn as sns
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import MDS

from DRiDO.network_summary import average_affinity_matrix, get_var_cluster_connectivity

environment = jinja2.Environment(
    loader=jinja2.FileSystemLoader("DRiDO/templates"))


def mds_layout(distances):
    mds_embedding = MDS(n_components=2, dissimilarity='precomputed')
    layout = mds_embedding.fit_transform(distances)
    return layout


def spectral_linkage(network, num_components=10):
    """Hierarchical clusteirng linkage with spectral distance."""
    spec_embedding = SpectralEmbedding(
        n_components=num_components, affinity='precomputed').fit_transform(network)
    network_distance = sp.distance.squareform(
        sp.distance.pdist(spec_embedding))
    network = network - np.diag(np.diag(network))
    linkage = hc.linkage(network_distance, method='average', metric='cosine')
    return linkage


def affinity_to_distance(X, gamma=1.0):
    return np.log(np.max(X) / (X + 1e-5))


def make_network_viz(sparse_precision_matrix,
                     node_locations=None,
                     cluster_labels=None,
                     highlighted_nodes=None,
                     write_to_file=True,
                     make_legend=False,
                     filename=None):
    if cluster_labels is None:
        clusters = range(sparse_precision_matrix.shape[0])
    else:
        clusters = cluster_labels.loc[sparse_precision_matrix.columns]['cluster'].values

    if node_locations is None:
        distances = affinity_to_distance(sparse_precision_matrix.abs().values)
        coords = mds_layout(distances)
        coord_min = np.min(coords, axis=0)
    else:
        coords = node_locations.loc[sparse_precision_matrix.columns].values
        coord_min = [0, 0]

    graph_nodes = {}

    for col, cluster, coord in zip(sparse_precision_matrix.columns, clusters, coords):
        feature_metadata = {'id': col, 'name': col, 'group': int(cluster) if isinstance(cluster, float) else cluster,
                            'x': float(coord[0] - coord_min[0]),
                            'y': float(coord[1] - coord_min[1])}
        if highlighted_nodes is not None:
            if col in highlighted_nodes:
                feature_metadata['highlight'] = True
            else:
                feature_metadata['highlight'] = False
        graph_nodes[col] = feature_metadata

    graph_links = []

    normalized_values = sparse_precision_matrix.values
    for c1, c2 in zip(*np.where(sparse_precision_matrix.abs().values > 1e-5)):
        if c1 != c2:
            x = sparse_precision_matrix.columns[c1]
            y = sparse_precision_matrix.columns[c2]
            graph_links.append({
                'source': x,
                'target': y,
                'value': normalized_values[c1, c2] / 20
            })

    graph_dict = {'nodes': list(graph_nodes.values()), 'links': graph_links}
    if make_legend:
        legend_template = environment.get_template("network_visualization_legend.jinja")
        rendered_html = legend_template.render({'graph_data': graph_dict})
    else:
        template = environment.get_template("network_visualization.jinja")
        rendered_html = template.render({'graph_data': graph_dict})

    if write_to_file and filename is not None:
        with open(filename, 'w') as f:
            f.write(rendered_html)
            print("Wrote to %s" % filename)
    else:
        return rendered_html


def make_chord_diagram(network,
                       cluster_labels,
                       write_to_file=True,
                       filename=None):

    matrix = average_affinity_matrix(
        network, cluster_labels)

    template = environment.get_template("chord_visualization.jinja")
    rendered_html = template.render({'graph_data': matrix})

    if write_to_file:
        with open(filename, 'w') as f:
            f.write(rendered_html)
            print("Wrote to %s" % filename)
    else:
        return rendered_html


def spectral_clustermap(network, **kwargs):
    """Clustermap with spectral linkage (see above)."""
    linkage = spectral_linkage(network)
    return sns.clustermap(
        network,
        row_linkage=linkage,
        col_linkage=linkage,
        **kwargs)


# Waterfall plots
def spread_labels(group, width=3000, jitter=0):
    padding = 0
    labels = group.index
    group = group.sort_values()
    estimated_width = np.array([len(label) for label in labels])
    estimated_intervals = (estimated_width[:-1] + estimated_width[1:]) / 2
    estimated_intervals = np.insert(
        estimated_intervals, 0, estimated_width[0] / 2)
    denom = 2 * padding + np.sum(estimated_width)
    estimated_intervals = np.cumsum(estimated_intervals / denom) * width
    x_pos = (width * (padding / denom)) + estimated_intervals
    if jitter > 0:
        x_pos += np.random.randn(len(x_pos)) * jitter * (width / len(x_pos))
    return x_pos


def compute_max_flow_layout(G, src, dst, jitter=0):
    _, flow_dict = nx.maximum_flow(
        G, src, dst, capacity='weight', flow_func=shortest_augmenting_path)
    flow_graph_df = pd.DataFrame(flow_dict).T.fillna(0)

    missing_cols = [
        c for c in flow_graph_df.index if c not in flow_graph_df.columns]
    for col in missing_cols:
        flow_graph_df.loc[:, col] = 0

    missing_rows = [
        c for c in flow_graph_df.columns if c not in flow_graph_df.index]
    for row in missing_rows:
        flow_graph_df.loc[row] = 0

    G = nx.from_pandas_adjacency(flow_graph_df, create_using=nx.DiGraph)
    layout = nx.nx_agraph.graphviz_layout(G, prog='dot', root=src)
    layout_df = pd.DataFrame(layout, index=['x', 'y']).T

    def _spread_labels(group, width=3000, jitter=0):
        # Fix position of source and destination in the middle of plot
        if src in group.index or dst in group.index:
            jitter = 0
        return spread_labels(group, width=width, jitter=jitter)

    layout_df['x'] = layout_df.groupby(
        'y').transform(_spread_labels, jitter=jitter)
    layout_df = layout_df.copy().fillna(0)
    return G, layout_df


def make_waterfall_plot(
        graph,
        src,
        dst,
        layout_df=None,
        signs=None,
        edge_width=5e1,
        figsize=(25, 20),
        font_size=15,
        jitter=0,
        edge_threshold=0,
        return_layout=True,
        normalize_weights=None,
):
    waterfall_fig = plt.figure(figsize=figsize)
    spec = mpl.gridspec.GridSpec(ncols=2, nrows=1,
                                 width_ratios=[25, 1], wspace=0.01)
    ax_diagram = waterfall_fig.add_subplot(spec[0])

    if jitter > 0:
        np.random.seed(0)

    if layout_df is None:
        max_flow_graph, layout_df = compute_max_flow_layout(
            graph, src, dst, jitter=jitter)
    else:
        max_flow_graph, _ = compute_max_flow_layout(
            graph, src, dst, jitter=jitter)
    layout = dict(zip(layout_df.index, layout_df.values))
    median_y = layout_df['y'].max() / 2

    nx.draw_networkx_labels(max_flow_graph, pos=layout, font_size=font_size, ax=ax_diagram)
    valid_edges = [(s, d) for s, d in max_flow_graph.edges()
                   if graph[s][d]['weight'] > edge_threshold]
    # max_edge_width = np.max([np.log(1 + graph[u][v]['weight']) * edge_width
    #                          for u, v in valid_edges])
    max_edge_width = np.max([graph[u][v]['weight'] * edge_width
                             for u, v in valid_edges])
    top_edges = [(s, d) for s, d in valid_edges
                 if layout_df.loc[d, 'y'] > median_y or s == src]
    # weights = [np.log(1 + graph[u][v]['weight']) * edge_width
    #            for u, v in top_edges]
    # weights = np.clip(weights, 0.01, np.inf)
    weights = [graph[u][v]['weight'] * edge_width
               for u, v in top_edges]

    if signs is not None:
        edge_vmax = max_edge_width
        edge_vmin = -max_edge_width
        weights = weights * np.array([signs[s][d] for s, d in top_edges])
        edge_cmap = sns.color_palette('coolwarm_r', as_cmap=True)
    else:
        edge_vmax = max_edge_width
        edge_vmin = 0
        edge_cmap = sns.color_palette('viridis', as_cmap=True)

    _ = nx.draw_networkx_edges(max_flow_graph,
                               pos=layout,
                               arrowstyle='->',
                               edgelist=top_edges,
                               arrowsize=20,
                               width=np.abs(weights),
                               edge_color=weights,
                               alpha=0.7,
                               min_source_margin=30,
                               min_target_margin=35,
                               edge_cmap=edge_cmap,
                               edge_vmax=edge_vmax,
                               edge_vmin=edge_vmin,
                               connectionstyle='angle3,angleA=0,angleB=90',
                               ax=ax_diagram)

    bottom_edges = [(s, d) for s, d in valid_edges
                    if layout_df.loc[d, 'y'] <= median_y and s != src]
    # weights = [np.log(1 + graph[u][v]['weight']) * edge_width
    #            for u, v in bottom_edges]
    # weights = np.clip(weights, 0.05, np.inf)
    weights = [graph[u][v]['weight'] * edge_width
               for u, v in bottom_edges]

    if signs is not None:
        edge_vmax = max_edge_width
        edge_vmin = -max_edge_width
        weights = weights * np.array([signs[s][d] for s, d in bottom_edges])
        edge_cmap = sns.color_palette('coolwarm_r', as_cmap=True)
    else:
        edge_vmax = max_edge_width
        edge_vmin = 0
        edge_cmap = sns.color_palette('viridis', as_cmap=True)

    _ = nx.draw_networkx_edges(max_flow_graph,
                               pos=layout,
                               arrowstyle='->',
                               edgelist=bottom_edges,
                               arrowsize=20,
                               width=np.abs(weights),
                               edge_color=weights,
                               alpha=0.7,
                               min_source_margin=30,
                               min_target_margin=70,
                               edge_cmap=edge_cmap,
                               edge_vmax=edge_vmax,
                               edge_vmin=edge_vmin,
                               connectionstyle='angle3,angleA=90,angleB=0',
                               ax=ax_diagram)
    
    sns.despine(left=True, bottom=True, ax=ax_diagram)

    # Plot colorbar
    ax_cbar = waterfall_fig.add_subplot(spec[1])
    if normalize_weights is None:
        normalize_weights = 1.0
        cbar_label = 'Prop. of covariance'
    else:
        cbar_label = 'Partial correlation'        
    norm = mpl.colors.Normalize(vmin=edge_vmin / edge_width / normalize_weights, vmax=edge_vmax / edge_width / normalize_weights)
    cb1 = mpl.colorbar.Colorbar(ax_cbar, cmap=edge_cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.ax.set_ylabel(cbar_label, rotation=270)
    if return_layout:
        return waterfall_fig, layout_df
    else:
        return waterfall_fig


def make_cluster_summary(network, clustering, target_vars, order=None, ignore=None, return_associations=True):
    eps = 1e-2
    # clustering_by_var = clustering.set_index('var')['cluster']
    cluster_associations = pd.concat([get_var_cluster_connectivity(
        network.abs(), clustering, var) for var in target_vars], axis=1)
    cluster_associations.columns = target_vars
    p95 = np.percentile(network.abs().loc[target_vars].values.flatten(), q=95)
    p5 = np.percentile(network.abs().loc[target_vars].values.flatten(), q=5)

    if order is None:
        # Number of non-zeros
        nnz_vals = (cluster_associations > eps).astype(int)
        score = sum([nnz_vals[var] for var in target_vars])
        raw_vals = sum([cluster_associations[var] for var in target_vars])
        score += 0.01 * raw_vals
        order = score.sort_values().index.values[::-1]

    if ignore is not None:
        order = [x for x in order if x not in ignore]

    results = [sns.clustermap(cluster_associations.loc[order],
                              square=True,
                              annot=cluster_associations.loc[order],
                              yticklabels=True,
                              vmin=p5,
                              vmax=p95,
                              row_cluster=False,
                              col_cluster=False,
                              annot_kws={"size": 80 /
                                         np.sqrt(len(cluster_associations))},
                              cbar_pos=None,
                              cmap='viridis',
                              figsize=(12, 16))]
    if return_associations:
        results.append(cluster_associations.loc[order])

    return tuple(results)
