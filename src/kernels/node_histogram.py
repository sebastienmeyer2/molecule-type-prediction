"""Define the node histogram graph kernel."""


from typing import Any, Dict, List, Optional

import numpy as np

from networkx import Graph


from kernels.base import Kernel
from utils.graph_ops import label_enrichment_morgan, label_enrichment_wl, extract_node_histograms
from utils.kernel_ops import gaussian_kernel
from utils.misc import assemble_histograms


class NodeHistogramKernel(Kernel):
    """Node histogram graph kernel."""

    name = "node_histogram"

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

        # For each enriched graphs, add the kernel
        for Xi, Yi in zip(X_wl, Y_wl):

            # Extract node histograms
            X_nodes = extract_node_histograms(Xi, verbose=self.verbose)
            if Y is not None:
                Y_nodes = extract_node_histograms(Yi, verbose=self.verbose)
            else:
                Y_nodes = X_nodes

            # Create arrays containing the node features
            X_hist, Y_hist = assemble_histograms(X_nodes, Y_nodes)

            # Compute the kernel
            K += gaussian_kernel(X_hist, Y_hist, sigma=self.sigma)

        return K
