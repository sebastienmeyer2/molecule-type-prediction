"""Define the kernel support vector classifier model."""


from typing import Any, Dict, Optional

import numpy as np
from cvxopt import matrix, solvers


from models.base import Model


class KernelSVC(Model):
    """Kernel support vector classifier model."""

    name = "svc"

    def __init__(self, **kwargs: Dict[str, Any]):
        """Create the model.

        Parameters
        ----------
        kwargs : dict of {str: any}
            Parameters for the model.
        """
        super().__init__()

        # Verbosity
        self.verbose = kwargs.get("verbose", False)

        # Specific parameters
        self.C = kwargs.get("C", 1.)
        self.max_iter = kwargs.get("max_iter", 50)

        # Internal parameters
        self.alpha = None

    def fit(self, K: np.ndarray, y: np.ndarray, W: Optional[np.ndarray] = None):
        """Fit the model.

        Parameters
        ----------
        K : np.ndarray
            Kernel matrix between training samples and themselves.

        y : np.ndarray
            Training labels in the format {-1, 1}.

        W : optional np.ndarray, default=None
            Sample weights.
        """
        n = len(y)

        # Compute the Jacobian
        P = 2.0 * K
        q = - 2.0 * y

        # Compute the inequality constraints
        G = 1.0 * np.vstack([-np.diagflat(y), np.diagflat(y)])
        h = np.concatenate((np.zeros(n), self.C*np.ones(n)))

        # Run quadratric programming solver
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)

        solvers.options["maxiters"] = self.max_iter
        solvers.options["show_progress"] = self.verbose
        sol = solvers.qp(P, q, G, h)

        self.alpha = np.array(sol["x"]).flatten()
