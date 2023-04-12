"""Define the abstract class for models."""


from abc import ABC, abstractmethod

from typing import Any, Dict, List, Optional

import numpy as np

from networkx import Graph

from utils.functions import sigmoid


class Model(ABC):
    """Abstract class for models."""

    def __init__(self, **kwargs: Dict[str, Any]):
        """Create the model.

        Parameters
        ----------
        kwargs : dict of {str: any}
            Parameters for the model.
        """
        # Verbosity
        self.verbose = kwargs.get("verbose", False)

        # Internal parameters
        self.alpha = None
        self.margin_mask = None
        self.margin_points = None

    @property
    @abstractmethod
    def name(self):
        """Name of the model for instantiation."""

    def get_alpha(self):
        """Return alpha."""
        return self.alpha

    @abstractmethod
    def fit(self, K: List[Graph], y: np.ndarray, W: Optional[np.ndarray] = None):
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

    def predict_logit(self, K: np.ndarray) -> np.ndarray:
        """Predict logits.

        Parameters
        ----------
        K : np.ndarray
            Kernel matrix between training and evaluation samples.

        Returns
        -------
        np.ndarray
            Predicted logits.
        """
        return K @ self.alpha

    def predict_proba(self, K: np.ndarray) -> np.ndarray:
        """Predict probabilities.

        Parameters
        ----------
        K : np.ndarray
            Kernel matrix between training and evaluation samples.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        return sigmoid(self.predict_logit(K))

    def predict(self, K: np.ndarray) -> np.ndarray:
        """Predict classes.

        Parameters
        ----------
        K : np.ndarray
            Kernel matrix between training and evaluation samples.

        Returns
        -------
        np.ndarray
            Predicted classes.
        """
        return 2 * (self.predict_logit(K) > 0) - 1
