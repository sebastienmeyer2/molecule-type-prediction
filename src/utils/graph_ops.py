"""Operations on graphs."""


from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx import Graph

from tqdm import tqdm


def flatten_labels(graph_list: List[Graph]) -> List[Graph]:
    """Simple preprocessing to remove list in labels.

    The raw data contains list of labels and our implementation expects floats.

    Parameters
    ----------
    graph_list : list of networkx.Graph
        List of input graphs.

    Returns
    -------
    graph_list : list of networkx.Graph
        Preprocessed list of graphs.
    """
    for graph in graph_list:

        for node in graph.nodes():
            graph.nodes[node]["labels"] = graph.nodes[node]["labels"][0]

        for edge in graph.edges():
            graph.edges[edge]["labels"] = graph.edges[edge]["labels"][0]

    return graph_list


def extract_largest_connected_component(graph_list: List[Graph]) -> List[Graph]:
    """Simple preprocessing to remove isolated nodes.

    The raw data contains some molecules with multiple connected components and our implementation
    expects connected components.

    Parameters
    ----------
    graph_list : list of networkx.Graph
        List of input graphs.

    Returns
    -------
    list of networkx.Graph
        Preprocessed list of graphs.
    """
    return [g.subgraph(max(nx.connected_components(g), key=len)).copy() for g in graph_list]


def get_nodes_labels(graph: Graph) -> Dict[int, int]:
    """Compute the correspondence between nodes and their labels.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.

    Returns
    -------
    dict of {int: int}
        Correspondence between nodes and their labels.
    """
    return {node: graph.nodes[node]["labels"] for node in graph.nodes()}


def get_edges_labels(graph: Graph) -> Dict[int, int]:
    """Compute the correspondence between edges and their labels.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.

    Returns
    -------
    dict of {int: int}
        Correspondence between edges and their labels.
    """
    return {edge: graph.edges[edge]["labels"] for edge in graph.edges()}


def morgan_one_step(graph_list: List[Graph]) -> List[Graph]:
    """One step label enrichment with Morgan indices.

    Label enrichment with Morgan indices consists in setting as new label the sum of all
    neighboring labels.

    Parameters
    ----------
    graph_list : list of networkx.Graph
        List of input graphs.

    Returns
    -------
    graph_list : list of networkx.Graph
        Preprocessed list of graphs.
    """
    for graph in graph_list:

        # For each node, compute the sum of its neighbors labels
        new_labels = {}

        for node in graph.nodes():

            neighbors = graph.neighbors(node)

            new_label = 0
            for neighbor in neighbors:
                new_label += graph.nodes[neighbor]["labels"]

            new_labels[node] = new_label

        # Relabel nodes
        for node in graph.nodes():

            graph.nodes[node]["labels"] = new_labels[node]

    return graph_list


def label_enrichment_morgan(graph_list: List[Graph], morgan_steps: int = 0) -> List[Graph]:
    """Label enrichment with Morgan indices.

    See `morgan_one_step()` for more information.

    Parameters
    ----------
    graph_list : list of networkx.Graph
        List of input graphs.

    morgan_steps : int, default=0
        Number of enrichment steps. If nonpositive, initial labels are kept.

    Returns
    -------
    graph_list : list of networkx.Graph
        Preprocessed list of graphs.
    """
    # If no step, return original list of graphs
    if morgan_steps <= 0:

        return graph_list

    # Initialize all labels to 1
    for graph in graph_list:

        for n in graph.nodes():

            graph.nodes[n]["labels"] = 1

    # Enrich labels for steps
    for _ in range(morgan_steps):

        graph_list = morgan_one_step(graph_list)

    return graph_list


def assign_wl_labels(
    graph_list: List[Graph], graph_hashes: List[Dict[int, List[str]]], labels_idx: int
) -> List[Graph]:
    """Assignment of Weisfeiler-Lehman labels.

    Parameters
    ----------
    graph_list : list of networkx.Graph
        List of input graphs.

    graph_hashes : list of dict of {int: list of str}
        For each graph, the corresponding list of labels hashes produced by the Weisfeiler-Lehman
        procedure.

    labels_idx : int
        For this given assignment, the index of the labels to assign.

    Returns
    -------
    labeled_graph_list : list of networkx.Graph
        Labeled list of graphs.
    """
    labeled_graph_list = []

    for graph, graph_hash in zip(graph_list, graph_hashes):

        labeled_graph = graph.copy()

        for node in labeled_graph.nodes():

            labeled_graph.nodes[node]["labels"] = graph_hash[node][labels_idx]

        labeled_graph_list.append(labeled_graph)

    return labeled_graph_list


def label_enrichment_wl(graph_list: List[Graph], wl_steps: int = 0) -> List[List[Graph]]:
    """Label enrichment with Weisfeiler-Lehman procedure.

    Parameters
    ----------
    graph_list : list of networkx.Graph
        List of input graphs.

    wl_steps : int, default=0
        Number of enrichment steps. If nonpositive, initial labels are kept.

    Returns
    -------
    graph_lists : list of list of networkx.Graph
        Preprocessed lists of graphs for each iteration of the procedure.
    """
    # If no step, return original list of graphs
    if wl_steps <= 0:

        graph_lists = [graph_list]

        return graph_lists

    # Compute the hashes for all graphs and nodes
    graph_hashes = [
        nx.weisfeiler_lehman_subgraph_hashes(
            graph, edge_attr="labels", node_attr="labels", iterations=wl_steps
        )
        for graph in graph_list
    ]

    # Create the list of list of graphs
    graph_lists = []

    for step in range(wl_steps):

        graph_lists.append(assign_wl_labels(graph_list, graph_hashes, step))

    return graph_lists


def extract_node_histograms(
    graph_list: List[Graph], verbose: bool = False
) -> List[Dict[int, int]]:
    """Extract node histograms from a list of graph.

    Parameters
    ----------
    graph_list : list of networkx.Graph
        List of graphs.

    verbose : bool, default=False
        Verbosity.

    Returns
    -------
    X_nodes : list of dict of {int: int}
        For each graph, a dictionary counting the number of each node label.
    """
    # Progress bar
    n = len(graph_list)

    if verbose:
        pbar = tqdm(range(n), desc="Computing nodes features")
    else:
        pbar = range(n)

    # For each graph we compute the dictionary counting the number of each node label
    X_nodes = []

    for i in pbar:

        X_nodes_i = {}

        graph_nodes_list = [graph_list[i].nodes[n]["labels"] for n in graph_list[i].nodes()]

        for node_label in graph_nodes_list:

            if node_label in X_nodes_i:

                X_nodes_i[node_label] += 1

            else:

                X_nodes_i[node_label] = 1

        X_nodes.append(X_nodes_i)

    return X_nodes


def extract_edge_histograms(
    graph_list: List[Graph], verbose: bool = False
) -> List[Dict[int, int]]:
    """Extract edge histograms from a list of graph.

    Parameters
    ----------
    graph_list : list of networkx.Graph
        List of graphs.

    verbose : bool, default=False
        Verbosity.

    Returns
    -------
    X_edges : list of dict of {int: int}
        For each graph, a dictionary counting the number of each edge label.
    """
    # Progress bar
    n = len(graph_list)

    if verbose:
        pbar = tqdm(range(n), desc="Computing edge histograms")
    else:
        pbar = range(n)

    # For each graph we compute the dictionary counting the number of each edge label
    X_edges = []

    for i in pbar:

        X_edges_i = {}

        graph_edges_list = [graph_list[i].edges[e]["labels"] for e in graph_list[i].edges()]

        for edge_label in graph_edges_list:

            if edge_label in X_edges_i:

                X_edges_i[edge_label] += 1

            else:

                X_edges_i[edge_label] = 1

        X_edges.append(X_edges_i)

    return X_edges


def find_node_walks_rec(
    graph: Graph, node: int, order: int, exclude_set: Optional[Set[int]] = None
) -> List[List[int]]:
    """Recursive walk construction starting at a specified node.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph.

    node : int
        Starting node for the walks.

    order : int
        Depth of the walks.

    exclude_set : optional set of int, default=None
        Keep visited nodes in memory.

    Returns
    -------
    walks : list of list of int
        Enumerate the possible walks from starting node **node** with depth **order**.
    """
    if order == 0:
        return [[node]]

    walks = [
        [node] + walk
        for neighbor in graph.neighbors(node)
        for walk in find_node_walks_rec(graph, neighbor, order-1, exclude_set=exclude_set)
    ]

    return walks


def find_nodes_walks(graph_list: List[Graph], order: int = 1) -> List[List[List[List[int]]]]:
    """Recursive walk construction for all graphs.

    Parameters
    ----------
    graph_list : list of networkx.Graph
        Input list of graph.

    order : int, default=1
        Depth of the walks.

    Returns
    -------
    graphs_walks : list of list of list of list of int
        Enumerate the possible walks from any node with depth **order**.
    """
    graphs_walks = []

    for graph in graph_list:

        graph_walks = []

        for node in graph:

            graph_walks.extend(find_node_walks_rec(graph, node, order))

        graphs_walks.append(graph_walks)

    return graphs_walks


def compute_graphs_sequences(
    graph_list: List[Graph], graphs_walks: List[List[List[List[int]]]], verbose: bool = False
) -> List[Set[Tuple[int]]]:
    """Path labels sequences for a specified graph.

    Parameters
    ----------
    graph_list : list of networkx.Graph
        Input list of graph.

    graphs_walks : list of list of list of list of int
        For each graph, the list of walks.

    order : int, default=1
        Depth of the walks.

    verbose : bool, default=False
        Verbosity.

    Returns
    -------
    graphs_sequences : list of set of tuple of int
        For each graph, contains the unique sequences of labels (nodes and edges).
    """
    # Progress bar
    n = len(graph_list)

    if verbose:
        pbar = tqdm(range(n), desc="Computing labels sequences")
    else:
        pbar = range(n)

    # For each graph, compute the walks labels sequences
    graphs_sequences = []

    for i in pbar:

        graph = graph_list[i]
        graph_walks = graphs_walks[i]

        # Build the corresponding sequence of labels (nodes and edges)
        graph_sequences = set()

        for walk in graph_walks:

            walk_length = len(walk)

            n_cur = walk[0]
            n_next = walk[1]
            edge = (n_cur, n_next)

            label_sequence = [
                graph.nodes[n_cur]["labels"],
                graph.edges[edge]["labels"],
                graph.nodes[n_next]["labels"]
            ]

            cnt = 1

            while cnt < walk_length - 1:

                cnt += 1

                n_cur = n_next
                n_next = walk[cnt]
                edge = (n_cur, n_next)

                label_sequence.append(graph.edges[edge]["labels"])
                label_sequence.append(graph.nodes[n_next]["labels"])

            graph_sequences.add(tuple(label_sequence))

        graphs_sequences.append(graph_sequences)

    return graphs_sequences


def compute_shortest_path_sequences(
    graph_list: List[Graph], shortest_path_list: List[Graph]
) -> Tuple[
    List[Set[Tuple[int, ...]]], List[Set[Tuple[int, ...]]], List[Dict[Tuple[int, ...], int]],
    List[Dict[Tuple[int, ...], int]], List[Dict[Tuple[int, int], int]]
]:
    """Shortest path labels and edges sequences.

    Parameters
    ----------
    graph_list : list of networkx.Graph
        Input list of graph.

    shortest_path_list : list of networkx.Graph
        For each graph, the corresponding shortest path graph.

    Returns
    -------
    labels : list of set of tuple of int
        For each graph, the set of shortest paths labels.

    labels_inv : list of set of tupel of int
        For each graph, the set of inversed shortest path labels.

    labels_dict : list dict of {tuple of int: int}
        For each graph, the number of shortest paths labels.

    labels_inv_dict : list of dict of {tuple of int: int}
        For each graph, the number of inversed shortest path labels.

    costs_dict : list of dict of {tuple of int: int}
        For each graph, the cost of shortest paths.
    """
    labels = []
    labels_inv = []
    labels_dict = []
    labels_inv_dict = []
    costs_dict = []

    for graph, shortest_path in zip(graph_list, shortest_path_list):

        l_shortest_path = set()
        l_shortest_path_inv = set()
        l_shortest_path_dict = {}
        l_shortest_path_inv_dict = {}
        e_shortest_path_dict = {}

        for node1 in graph:
            for node2 in graph:

                cost = shortest_path[node1][node2]

                l_t = (
                    graph.nodes[node1]["labels"],
                    graph.nodes[node2]["labels"],
                    cost  # delta kernel
                )

                l_shortest_path.add(l_t)

                if l_t in l_shortest_path_dict:
                    l_shortest_path_dict[l_t] += 1
                else:
                    l_shortest_path_dict[l_t] = 1

                e_shortest_path_dict[l_t] = cost  # brownian bridge kernel

                l_inv = (
                    graph.nodes[node2]["labels"],
                    graph.nodes[node1]["labels"],
                    cost  # delta kernel
                )

                l_shortest_path_inv.add(l_inv)

                if l_inv in l_shortest_path_inv_dict:
                    l_shortest_path_inv_dict[l_inv] += 1
                else:
                    l_shortest_path_inv_dict[l_inv] = 1

        labels.append(l_shortest_path)
        labels_inv.append(l_shortest_path_inv)
        labels_dict.append(l_shortest_path_dict)
        labels_inv_dict.append(l_shortest_path_inv_dict)
        costs_dict.append(e_shortest_path_dict)

    return labels, labels_inv, labels_dict, labels_inv_dict, costs_dict
