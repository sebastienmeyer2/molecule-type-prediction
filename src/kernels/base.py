"""Define the abstract class for kernels."""


from abc import ABC, abstractmethod

from typing import List, Optional

import numpy as np

from networkx import Graph


class Kernel(ABC):
    """Abstract class for kernels."""

    @property
    @abstractmethod
    def name(self):
        """Name of the kernel for instantiation."""

    @abstractmethod
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
