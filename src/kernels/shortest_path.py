"""Define the shortest path graph kernel."""


from typing import Any, Dict, List, Optional

import numpy as np

import networkx as nx
from networkx import Graph

from tqdm import tqdm


from kernels.base import Kernel
from utils.graph_ops import (
    label_enrichment_morgan, label_enrichment_wl, compute_shortest_path_sequences
)


class ShortestPathKernel(Kernel):
    """Shortest path graph kernel."""

    name = "shortest_path"

    def __init__(self, **kwargs: Dict[str, Any]):
        """Create the graph kernel.

        Parameters
        ----------
        kwargs : dict of {str: any}
            Parameters for the graph kernel.
        """
        # Verbosity
        self.verbose = kwargs.get("verbose", False)

        # Specific parameters
        self.morgan_steps = kwargs.get("morgan_steps", 0)
        self.wl_steps = kwargs.get("wl_steps", 0)
        self.margin = kwargs.get("margin", 2)

    def kernel(self, X: List[Graph], Y: Optional[List[Graph]] = None) -> np.ndarray:
        """Compute the kernel between two list of graphs.

        Parameters
        ----------
        X : list of networkx.Graph
            First list of graphs.

        Y : optional list of networkx.Graph, default=None
            Second list of graphs. If not specified, **X** is used as the second list of graphs.
            This behavior allows to avoid computing features twice when the kernel is computed
            on the same lists.

        Returns
        -------
        K : np.ndarray
            Evaluation of the kernel for each graph in **X** and **Y**.
        """
        # Initialize kernel
        n = len(X)
        m = len(Y) if Y is not None else n
        K = np.zeros((n, m))

        # Label enrichment with Morgan indices
        X_morgan = label_enrichment_morgan(X, morgan_steps=self.morgan_steps)
        if Y is not None:
            Y_morgan = label_enrichment_morgan(Y, morgan_steps=self.morgan_steps)
        else:
            Y_morgan = X_morgan

        # Label enrichment with Weisfeiler-Lehman procedure
        X_wl = label_enrichment_wl(X_morgan, wl_steps=self.wl_steps)
        if Y is not None:
            Y_wl = label_enrichment_wl(Y_morgan, wl_steps=self.wl_steps)
        else:
            Y_wl = X_wl

        for Xi, Yi in zip(X_wl, Y_wl):

            # Progress bar
            if self.verbose:
                pbar = tqdm(range(n), desc="Computing shortest path kernel")
            else:
                pbar = range(n)

            visited_graphs = set()

            # Compute shortest paths
            X_shortest = [nx.floyd_warshall(graph) for graph in Xi]
            if Y is not None:
                Y_shortest = [nx.floyd_warshall(graph) for graph in Yi]
            else:
                Y_shortest = X_shortest

            # Compute the labels sequences
            X_l, X_l_inv, X_l_dict, X_l_inv_dict, _ = compute_shortest_path_sequences(
                Xi, X_shortest
            )
            if Y is not None:
                Y_l, _, Y_l_dict, _, _ = compute_shortest_path_sequences(Yi, Y_shortest)
            else:
                Y_l = X_l
                Y_l_dict = X_l_dict
                # Y_costs_dict = X_costs_dict

            # Compute the kernel
            for i in pbar:
                for j in range(m):

                    if (j, i) in visited_graphs:

                        K[i, j] = K[j, i]

                    else:

                        k_i_j = 0

                        common_labels = X_l[i] & Y_l[j]

                        for ld in common_labels:

                            # Delta kernel
                            k_i_j += X_l_dict[i][ld] * Y_l_dict[j][ld]

                            # Brownian bridge kernel
                            # c_diff = np.abs(X_costs_dict[i][ld] - Y_costs_dict[j][ld])
                            # k_i_j += max(0, self.margin - e_diff)

                        common_labels_inv = X_l_inv[i] & Y_l[j]

                        for li in common_labels_inv:

                            # Delta kernel
                            k_i_j += X_l_inv_dict[i][li] * Y_l_dict[j][li]

                            # Brownian bridge kernel
                            # e_diff = np.abs(X_costs_dict[i][li[::-1]] - Y_costs_dict[j][li[::-1]])
                            # k_i_j += max(0, self.margin - e_diff)

                        K[i, j] += k_i_j

                    visited_graphs.add((i, j))

        return K
