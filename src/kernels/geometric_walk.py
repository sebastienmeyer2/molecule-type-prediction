"""Define the geometric walk graph kernel."""


from typing import Any, Dict, List, Optional

import numpy as np

from networkx import Graph

from tqdm import tqdm


from kernels.base import Kernel
from utils.graph_ops import (
    label_enrichment_morgan, label_enrichment_wl, find_nodes_walks, compute_graphs_sequences
)


class GeometricWalkKernel(Kernel):
    """Geometric walk graph kernel."""

    name = "geometric_walk"

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
        self.order = kwargs.get("order", 3)
        self.beta = kwargs.get("beta", 0.01)

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

        # Compute the kernel
        beta = 1

        for order in range(1, self.order+1):

            # Update beta
            beta *= self.beta

            for Xi, Yi in zip(X_wl, Y_wl):

                # Progress bar
                if self.verbose:
                    pbar = tqdm(range(n), desc=f"Computing {order}-th order walk kernel")
                else:
                    pbar = range(n)

                visited_graphs = set()

                # Compute walks
                X_walks = find_nodes_walks(Xi, order=order)
                if Y is not None:
                    Y_walks = find_nodes_walks(Yi, order=order)
                else:
                    Y_walks = X_walks

                # Compute labels sequences
                X_sequences = compute_graphs_sequences(Xi, X_walks)
                if Y is not None:
                    Y_sequences = compute_graphs_sequences(Yi, Y_walks)
                else:
                    Y_sequences = X_sequences

                # Aggregate results
                for i in pbar:
                    for j in range(m):

                        if (j, i) in visited_graphs:
                            K[i, j] = K[j, i]
                        else:
                            K[i, j] += beta * len(X_sequences[i] & Y_sequences[j])

                        visited_graphs.add((i, j))

        return K
