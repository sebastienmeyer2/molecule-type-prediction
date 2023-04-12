"""Define the kernel ridge regression model."""


from typing import Any, Dict, Optional

import warnings

import numpy as np


from models.base import Model


class KernelRidgeRegression(Model):
    """Kernel ridge regression model."""

    name = "linreg"

    def __init__(self, **kwargs: Dict[str, Any]):
        """Create the model.

        Parameters
        ----------
        kwargs : dict of {str: any}
            Parameters for the model.
        """
        super().__init__()

        # Specific parameters
        self.lambd = kwargs.get("lambd", 0.01)

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

        if W is None:

            W = np.ones(n)

        # Solve closed form kernel ridge regression
        W_sqrt = np.sqrt(np.diag(W))

        weighted_K = W_sqrt @ K @ W_sqrt + n * self.lambd * np.eye(n)

        with warnings.catch_warnings():

            warnings.simplefilter("ignore", category=RuntimeWarning)

            self.alpha = W_sqrt @ np.linalg.solve(weighted_K, W_sqrt @ y)
