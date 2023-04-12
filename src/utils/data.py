"""Data getter."""


from typing import List, Tuple

import pickle

import numpy as np

from networkx import Graph


from utils.graph_ops import flatten_labels, extract_largest_connected_component


def load_graph_data(data_dir: str = "data") -> Tuple[List[Graph], np.ndarray, List[Graph]]:
    """Retrieve local data files.

    Parameters
    ----------
    data_dir : str, default="data"
        Directory where data is stored.

    Returns
    -------
    graph_list_train : list of networkx.Graph
        List of training graphs.

    y_train : numpy.ndarray
        Training classes.

    graph_list_test : list of networkx.Graph
        List of test graphs.
    """
    # Read files
    path_to_train_data = f"{data_dir}/training_data.pkl"
    path_to_test_data = f"{data_dir}/test_data.pkl"
    path_to_train_labels = f"{data_dir}/training_labels.pkl"

    graph_list_train = pickle.load(open(path_to_train_data, "rb"))
    graph_list_test = pickle.load(open(path_to_test_data, "rb"))
    y_train = pickle.load(open(path_to_train_labels, "rb"))

    # Flatten graph labels
    graph_list_train = flatten_labels(graph_list_train)
    graph_list_test = flatten_labels(graph_list_test)

    # Extract largest connected component (expected to represent one molecule)
    graph_list_train = extract_largest_connected_component(graph_list_train)
    graph_list_test = extract_largest_connected_component(graph_list_test)

    # Convert classes in {-1, 1}
    if 0 in np.unique(y_train):

        y_train = 2 * y_train - 1

    return graph_list_train, y_train, graph_list_test
