"""Define the sum graph kernel."""


from typing import Any, Dict, List, Optional

import numpy as np

from networkx import Graph


from kernels import (
    CountingKernel, NodeHistogramKernel, EdgeHistogramKernel, GeometricWalkKernel,
    ShortestPathKernel
)
from kernels.base import Kernel
from utils.kernel_ops import normalize_kernel, center_kernel


class SumKernel(Kernel):
    """Sum graph kernel."""

    name = "sum"

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
        self.sigma = kwargs.get("sigma", 4.96)
        self.sigma_m = kwargs.get("sigma_m", 0.04)
        self.sigma_wl = kwargs.get("sigma_wl", 1.18)
        self.beta = kwargs.get("beta", 2.34)
        self.beta_m = kwargs.get("beta_m", 1.89)
        self.beta_wl = kwargs.get("beta_wl", 0.23)

        # Subkernels
        self.kernels = [
            # No label enrichment
            CountingKernel(verbose=self.verbose, sigma=self.sigma),
            EdgeHistogramKernel(verbose=self.verbose, sigma=self.sigma),
            NodeHistogramKernel(
                verbose=self.verbose, morgan_steps=0, wl_steps=0, sigma=self.sigma
            ),
            GeometricWalkKernel(
                verbose=self.verbose, morgan_steps=0, wl_steps=0, order=4, beta=self.beta
            ),
            ShortestPathKernel(verbose=self.verbose, morgan_steps=0, wl_steps=0),

            # Morgan label enrichment
            NodeHistogramKernel(
                verbose=self.verbose, morgan_steps=1, wl_steps=0, sigma=self.sigma_m
            ),
            GeometricWalkKernel(
                verbose=self.verbose, morgan_steps=1, wl_steps=0, order=4, beta=self.beta_m
            ),
            ShortestPathKernel(verbose=self.verbose, morgan_steps=1, wl_steps=0),

            # Weisfeiler-Lehman label enrichment
            NodeHistogramKernel(
                verbose=self.verbose, morgan_steps=0, wl_steps=1, sigma=self.sigma_wl
            ),
            GeometricWalkKernel(
                verbose=self.verbose, morgan_steps=0, wl_steps=1, order=4, beta=self.beta_wl
            ),
            ShortestPathKernel(verbose=self.verbose, morgan_steps=0, wl_steps=1),
        ]

        self.n_kernels = len(self.kernels)
        self.eta = np.ones(self.n_kernels) / self.n_kernels

        self.normalize = False
        self.center = False

    def get_eta(self):
        """Return the weights of the kernels."""
        return self.eta

    def set_eta(self, eta):
        """Set the weights of the kernels."""
        assert len(eta) == len(self.eta), "Length mismatch in the weights of the kernels!"
        self.eta = eta

    def compute_kernel_list(
        self, X: List[Graph], Y: Optional[List[Graph]] = None
    ) -> List[np.ndarray]:
        """Compute all the kernels.

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
        K_list : list of np.ndarray
            Evaluation of the kernels for each graph in **X** and **Y**.
        """
        # Compute all the kernels
        K_list = []

        for kernel in self.kernels:

            X_kernel = [graph.copy() for graph in X]
            if Y is not None:
                Y_kernel = [graph.copy() for graph in Y]
            else:
                Y_kernel = None

            K_list.append(kernel.kernel(X_kernel, Y=Y_kernel))

        # Normalize all the kernels
        if self.normalize:

            K_list = [normalize_kernel(k) for k in K_list]

        # Center all the kernels
        if self.center:

            K_list = [center_kernel(k) for k in K_list]

        return K_list

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
        # Compute all the kernels
        K_list = self.compute_kernel_list(X, Y=Y)

        # Compute the sum
        K = np.sum([eta * k for eta, k in zip(self.eta, K_list)], axis=0)

        return K
