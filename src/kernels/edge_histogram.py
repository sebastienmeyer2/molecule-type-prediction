"""Define the edge histogram graph kernel."""


from typing import Any, Dict, List, Optional

import numpy as np

from networkx import Graph


from kernels.base import Kernel
from utils.graph_ops import extract_edge_histograms
from utils.kernel_ops import gaussian_kernel
from utils.misc import assemble_histograms


class EdgeHistogramKernel(Kernel):
    """Edge histogram graph kernel."""

    name = "edge_histogram"

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
        # Extract edge histograms
        X = extract_edge_histograms(X, verbose=self.verbose)
        if Y is not None:
            Y = extract_edge_histograms(Y, verbose=self.verbose)
        else:
            Y = X.copy()

        # Create arrays containing the edge features
        X, Y = assemble_histograms(X, Y)

        # Compute the kernel
        K = gaussian_kernel(X, Y, sigma=self.sigma)

        return K
