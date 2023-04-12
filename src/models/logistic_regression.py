"""Define the kernel logistic regression model."""


from typing import Any, Dict, Optional

import warnings

import numpy as np
from scipy import optimize

from tqdm import tqdm


from models import KernelRidgeRegression
from models.base import Model
from utils.functions import logistic_loss, logistic_loss_p, logistic_loss_pp


class KernelLogisticRegression(Model):
    """Kernel logistic regression model."""

    name = "logreg"

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
        self.lambd = kwargs.get("lambd", 1.73e-5)
        self.method = kwargs.get("method", "irls")
        assert self.method in {"irls", "newton"}, f"Unknown optimization method {self.method}."
        self.max_iter = kwargs.get("max_iter", 50)
        self.tol = kwargs.get("tol", 1e-6)

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

        # Solve kernel logistic regression by iterated reweighted least-square
        if self.method == "irls":

            alpha_t = np.zeros(n)

            kernel_ridge_regression = KernelRidgeRegression(lambd=self.lambd)

            nb_iter = 0
            if self.verbose:
                pbar = tqdm(total=self.max_iter)

            while nb_iter < self.max_iter:

                # Update parameters
                m = K @ alpha_t
                P_t = logistic_loss_p(- y * m)
                W_t = logistic_loss_pp(y * m)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    z_t = m - P_t * y / W_t

                # Solve kernel ridge regression problem
                kernel_ridge_regression.fit(K, z_t, W=W_t)

                alpha_new = kernel_ridge_regression.get_alpha()

                delta = np.linalg.norm(alpha_new - alpha_t)
                alpha_t = alpha_new

                # Stopping criteria
                if delta < self.tol:
                    break

                nb_iter += 1
                if self.verbose:
                    pbar.update(1)

            if self.verbose:
                pbar.close()

            self.alpha = alpha_t

        # Solve kernel logistic regression with Newton's method
        elif self.method == "newton":

            # Lagrange primal problem
            def loss(alpha):
                K_alpha = K @ alpha
                return logistic_loss(y * K_alpha).mean() + 0.5 * self.lambd * alpha.T @ K_alpha

            # Partial derivative of Lagrange primal problem wrt alpha
            def grad_loss(alpha):
                K_alpha = K @ alpha
                P_alpha = np.diag(logistic_loss_p(y * K_alpha))
                return K @ P_alpha @ y / n + self.lambd * K_alpha

            # Second partial derivative of Lagrange primal problem wrt alpha
            def hessian_loss(alpha):
                K_alpha = K @ alpha
                W = np.diag(logistic_loss_pp(y * K_alpha))
                return K @ W @ K / n + self.lambd * K

            with warnings.catch_warnings():

                warnings.simplefilter("ignore", category=RuntimeWarning)

                optimization_results = optimize.minimize(
                    loss, np.ones(n), method="Newton-CG", jac=grad_loss, hess=hessian_loss,
                    options={"maxiter": 20*self.max_iter, "disp": self.verbose}
                )

            self.alpha = optimization_results.x
