"""Define the counting graph kernel."""


from typing import Any, Dict, List, Optional

import numpy as np

from networkx import Graph


from kernels.base import Kernel
from utils.kernel_ops import gaussian_kernel


class CountingKernel(Kernel):
    """Counting graph kernel."""

    name = "count"

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
        self.sigma = kwargs.get("sigma", 1.)

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
        # Count nodes
        X_nodes_count = np.array([graph.number_of_nodes() for graph in X])[:, np.newaxis]
        if Y is not None:
            Y_nodes_count = np.array([graph.number_of_nodes() for graph in Y])[:, np.newaxis]
        else:
            Y_nodes_count = X_nodes_count.copy()

        # Count edges
        X_edges_count = np.array([graph.number_of_edges() for graph in X])[:, np.newaxis]
        if Y is not None:
            Y_edges_count = np.array([graph.number_of_edges() for graph in Y])[:, np.newaxis]
        else:
            Y_edges_count = X_edges_count.copy()

        # Concatenate
        X = np.hstack((X_nodes_count, X_edges_count))
        Y = np.hstack((Y_nodes_count, Y_edges_count))

        # Compute the kernel
        K = gaussian_kernel(X, Y, sigma=self.sigma)

        return K
