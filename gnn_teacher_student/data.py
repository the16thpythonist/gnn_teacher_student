import random
import itertools
from typing import List, Tuple, Optional, Dict, Callable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# == GLOBAL VARIABLES =======================================================================================
# ===========================================================================================================

COLORS_DESCRIPTION = """
COLORS DATASET
==============
The "COLORS" dataset is created by randomly generating graphs of different sizes. Each node has three float
features (R, G, B) corresponding to RGB color values. The graphs are undirected and all edge weights are 1.
"""

COLOR_PAIRS_DESCRIPTION = """
COLOR PAIRS TASK
================
"COUNT COLOR PAIRS" is a possible task which is defined on the "COLORS" dataset. it is possible to define
two colors and the task to be performed will be to predict an integer value for the number of nodes of the
first color inside the graph, which are connected to at least one node of the second color.
"""


# == BASIC GRAPH GENERATION =================================================================================
# ===========================================================================================================

def generate_colors_graph(node_count: int,
                          additional_edge_count: int,
                          colors: List[Tuple[float, float, float]],
                          color_weights: Optional[List[float]] = None,) -> Dict[str, np.ndarray]:
    # If no explicit weights for the random color choice are provided, we assume that all colors should
    # occur with equal probability.
    if color_weights is None:
        color_weights = [1] * len(colors)

    node_indices = list(range(node_count))
    node_attributes = [] + random.choices(colors, weights=color_weights, k=1)
    edge_indices = []

    node_indices_inserted = [0]
    node_indices_remaining = list(range(1, node_count))
    random.shuffle(node_indices_remaining)

    # ~ Generating basic graph structure
    while len(node_indices_remaining) != 0:
        node_1 = random.choice(node_indices_inserted)
        node_2 = node_indices_remaining.pop()

        # Generating the color attributes for the newly inserted node
        node_attributes += random.choices(colors, weights=color_weights, k=1)

        # Adding an undirected edge between the two nodes to the edge list (= two directed edges)
        edge_indices += [(node_1, node_2), (node_2, node_1)]

        # In the end we also have to actually add the node to the list of inserted nodes
        node_indices_inserted.append(node_2)

    # ~ Adding the additional edges
    inserted_edge_count = 0
    while inserted_edge_count != additional_edge_count:
        node_1, node_2 = random.sample(node_indices, 2)

        # If no edge between these two nodes already exists, we add one
        if (node_1, node_2) not in edge_indices and (node_2, node_1) not in edge_indices:
            edge_indices += [(node_1, node_2), (node_2, node_1)]
            inserted_edge_count += 1

    # ~ Creating edge weights and adjancency matrix
    edge_attributes = np.ones(shape=(len(edge_indices), 1))
    node_adjacency = [[1 if (i, j) in edge_indices else 0 for i in node_indices] for j in node_indices]

    # ~ Converting to numpy arrays
    return {
        'node_indices':             np.array(node_indices, dtype=np.int32),
        'node_attributes':          np.array(node_attributes, dtype=np.float32),
        'node_adjacency':           np.array(node_adjacency, dtype=np.int32),
        'edge_indices':             np.array(edge_indices, dtype=np.int32),
        'edge_attributes':          np.array(edge_attributes, dtype=np.float32)
    }


# == DETERMINISTIC GRAPH ALGORITHMS =========================================================================
# ===========================================================================================================

def find_color_pairs(node_colors: np.ndarray,
                     node_adjacency: np.ndarray,
                     color1: Tuple[float, float, float],
                     color2: Tuple[float, float, float]) -> List[Tuple[int, int]]:

    n = node_colors.shape[0]
    assert node_colors.shape[1] == 3, 'second dimension of node_colors needs to be 3! one float value for (R, G, B)'
    assert node_adjacency.shape == (n, n), 'node_adjacency matrix dimension needs to match the first dim of node_colors'

    # We convert the colors from tuples to numpy arrays here because later on we need to compare these
    # values with the values in the node_colors numpy array.
    color1 = np.array(color1)
    color2 = np.array(color2)

    # The following algorithm is my attempt at finding a somewhat efficient implementation for this
    # problem... It is probably more complicated than it needs to be.


    node_indices = np.arange(n)
    is_visited = np.zeros(shape=(n,))
    node_partner = np.zeros(shape=(n,))
    node_partner.fill(np.nan)
    partner_tuples = []

    stack = [0]

    while len(stack) != 0:
        i = stack.pop(0)

        if np.isnan(node_partner[i]) and not is_visited[i]:

            for c1, c2 in [(color1, color2), (color2, color1)]:
                if np.all(node_colors[i] == c1):

                    for j in node_indices:
                        if node_adjacency[i][j] \
                           and i != j \
                           and np.all(node_colors[j] == c2):
                            node_partner[i] = j
                            node_partner[j] = i
                            partner_tuples.append((i, j))

        # Adding all the connected nodes to the stack
        for j in node_indices:
            if node_adjacency[i][j] and not is_visited[j]:
                stack.append(j)

        # Marking the node as visited
        is_visited[i] = 1

    return partner_tuples


# == ENTIRE DATASET GENERATION ==============================================================================
# ===========================================================================================================

def generate_color_pairs_dataset(length: int,
                                 node_count_cb: Callable[[], int] = lambda: random.randint(3, 10),
                                 additional_edge_count_cb: Callable[[], int] = lambda: random.randint(1, 3),
                                 colors: List[Tuple[float, float, float]] = ((1, 0, 0), (0, 1, 0)),
                                 color_weights: Optional[List[int]] = None,
                                 color1: Tuple[float, float, float] = (1, 0, 0),
                                 color2: Tuple[float, float, float] = (0, 1, 0),
                                 exclude_empty: bool = False) -> List[dict]:
    color1 = np.array(color1)
    color2 = np.array(color2)

    dataset = []

    while len(dataset) != length:
        # ~ Generating the base graph
        g = generate_colors_graph(
            node_count=node_count_cb(),
            additional_edge_count=additional_edge_count_cb(),
            colors=colors,
            color_weights=color_weights,
        )

        # ~ Generating label and explanation
        color_pairs = find_color_pairs(
            node_colors=g['node_attributes'],
            node_adjacency=g['node_adjacency'],
            color1=color1,
            color2=color2
        )
        # This statement here will create a list of all node indices which are part of at least one color
        # pair - without duplicates (this is a concern, since a node may be part of multiple pairs)
        color_pair_nodes = list(set(itertools.chain(*color_pairs)))

        # At the core, this statement will create a list which contains the node indices for every node of
        # the given color1 and which is at the same time part of a color pair with a node of color2. The
        # round truth label for this problem will then be the total number of all such nodes which can be
        # found in the graph.
        g['graph_labels'] = np.array(len([
            i
            for i in color_pair_nodes
            if np.all(g['node_attributes'][i] == color1)
        ]), dtype=np.float32)

        # If the corresponding flag was passed as an argument, this section will make sure that only graphs
        # which contain AT LEAST ONE color pair are actually added to the dataset
        if exclude_empty and g['graph_labels'] == 0:
            continue

        # For the ground truth node explanations, every node of the color1 which is at the same time part of
        # a color pair with a node of color2 will be marked with importance "1" and every other kind of node
        # will be marked as
        # "0"
        g['node_importances'] = np.array([
            int(i in color_pair_nodes and np.all(g['node_attributes'][i] == color1))
            for i in g['node_indices']
        ])

        # For the ground truth edge explanations, every edge which connects a color pair (node of color1 and
        # node of color2) will be marked with importance "1" and every other kind of node will be marked "0"
        g['edge_importances'] = np.array([
            int((i, j) in color_pairs or (j, i) in color_pairs)
            for i, j in g['edge_indices']
        ])

        # ~ Adding to dataset
        dataset.append(g)

    return dataset
