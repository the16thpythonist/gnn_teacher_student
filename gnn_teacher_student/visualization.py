from typing import Callable, List, Any

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx


# == GRAPH PLOTTING =========================================================================================
# ===========================================================================================================

def draw_colors_graph(g: dict,
                      ax: plt.Axes,
                      alpha: float = 0.6,
                      layout_func: Callable = nx.drawing.layout.kamada_kawai_layout,
                      node_size: int = 500,
                      edge_size: int = 1,
                      include_labels: bool = False):

    nodes_dict = {int(i): {'color': g['node_attributes'][i]}
                  for i in g['node_indices']}

    graph = nx.Graph()
    graph.add_nodes_from(nodes_dict)
    graph.add_edges_from(g['edge_indices'])

    pos = layout_func(graph)
    for i in g['node_indices']:
        graph.nodes[i]['pos'] = pos[i]

    for i in g['node_indices']:
        x, y = graph.nodes[i]['pos']
        ax.scatter(
           x, y,
           color='white',
           s=node_size * 0.9,
           zorder=0
        )

    nx.draw_networkx_nodes(
        graph,
        ax=ax,
        pos=pos,
        node_size=node_size,
        node_color=g['node_attributes'],
        alpha=alpha,
        edgecolors='black',
        linewidths=edge_size
    )

    for (i, j) in g['edge_indices']:
        x1, y1 = graph.nodes[i]['pos']
        x2, y2 = graph.nodes[j]['pos']

        ax.plot(
            (x1, x2),
            (y1, y2),
            color='black',
            lw=edge_size,
            zorder=-1
        )

    if include_labels:
        labels = {int(i): str(i) for i in g['node_indices']}

        nx.draw_networkx_labels(
            graph,
            ax=ax,
            pos=pos,
            labels=labels
        )

    return ax, graph


def draw_graph_node_importances(node_importances: np.ndarray,
                                graph: nx.Graph,
                                ax: plt.Axes,
                                cmap=plt.get_cmap('Greys'),
                                size=1000,
                                vmin: float = 0,
                                vmax: float = 1,
                                threshold: float = 0.1):
    normalize_value = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i in graph:
        x, y = graph.nodes[i]['pos']
        value = normalize_value(node_importances[i])
        color = cmap(value)

        if value > threshold:
            ax.scatter(
                x, y,
                s=size,
                color=color,
                zorder=-1
            )

    return ax


def draw_graph_edge_importances(edge_indices: np.ndarray,
                                edge_importances: np.ndarray,
                                graph: nx.Graph,
                                ax: plt.Axes,
                                cmap=plt.get_cmap('Greys'),
                                size=6,
                                vmin: float = 0,
                                vmax: float = 1,
                                threshold: float = 0.1):
    normalize_value = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for (i, j), importance in zip(edge_indices, edge_importances):
        x1, y1 = graph.nodes[i]['pos']
        x2, y2 = graph.nodes[j]['pos']

        value = normalize_value(importance)
        color = cmap(value)

        if value > threshold:
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                lw=size,
                zorder=-1
            )

    return ax


# == PLOTTING ===============================================================================================
# ===========================================================================================================

def plot_average_with_uncertainty(ax: plt.Axes,
                                  xs: List[float],
                                  yss: List[List[float]],
                                  color: Any = 'black',
                                  bounds_alpha: float = 0.3,
                                  fill_alpha: float = 0.1):

    avgs = [np.mean(ys) for ys in yss]
    stds = [np.std(ys) for ys in yss]
    ns = [len(ys) for ys in yss]

    uncertainty_lower = [avg - (np.abs(std) / np.sqrt(n)) for n, avg, std in zip(ns, avgs, stds)]
    uncertainty_upper = [avg + (np.abs(std) / np.sqrt(n)) for n, avg, std in zip(ns, avgs, stds)]

    ax.plot(
        xs,
        avgs,
        color=color,
        alpha=1.0,
    )

    ax.plot(
        xs,
        uncertainty_lower,
        color=color,
        alpha=bounds_alpha
    )

    ax.plot(
        xs,
        uncertainty_upper,
        color=color,
        alpha=bounds_alpha
    )

    ax.fill_between(
        xs,
        uncertainty_lower,
        uncertainty_upper,
        color=color,
        alpha=fill_alpha
    )


