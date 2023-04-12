"""Operations on kernel matrices."""


import numpy as np


def linear_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute linear kernel of two datasets.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape n x d.

    Y : np.ndarray
        Input array of shape m x d.

    Returns
    -------
    np.ndarray
        Linear kernel evaluation of shape n x m.
    """
    return np.einsum("nd,md->nm", X, Y)


def gaussian_kernel(X: np.ndarray, Y: np.ndarray, sigma: float = 1.) -> np.ndarray:
    """Compute gaussian or rbf kernel of two datasets.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape n x d.

    Y : np.ndarray
        Input array of shape m x d.

    sigma : float, default=1.0
        Bandwidth parameter.

    Returns
    -------
    np.ndarray
        Gaussian or RBF kernel evaluation of shape n x m.
    """
    return np.exp(
        (
            np.einsum("nd,md->nm", X, Y) - 0.5 * (
                np.einsum(
                    "nd,nd->n", X, X
                )[:, np.newaxis] + np.einsum(
                    "md,md->m", Y, Y
                )[np.newaxis, :]
            )
        ) / sigma**2
    )


def normalize_kernel(K: np.ndarray) -> np.ndarray:
    r"""Normalize a kernel matrix.

    The formula for normalizing a kernel matrix is:

    .. math::

            \tilde{K}(x_i, x_j) =  \frac{K(x_i, x_j)}{\sqrt{K(x_i, x_i) K(x_j, x_j)}}.

    Parameters
    ----------
    K : np.ndarray
        Input kernel matrix.

    Returns
    -------
    K_scaled : np.ndarray
        Normalized kernel matrix.
    """
    D = np.diag(1.0 / np.sqrt(np.diag(K) + 1e-6))
    K_scaled = D @ K @ D

    return K_scaled


def center_kernel(K: np.ndarray) -> np.ndarray:
    r"""Center a kernel matrix.

    The formula for centering a kernel matrix is:

    .. math::

            \tilde{K} = (\mathbb{I} - \frac{1}{n} H) K (\mathbb{I} - \frac{1}{n} H)

    where the matrix H is full of ones.

    Parameters
    ----------
    K : np.ndarray
        Input kernel matrix.

    Returns
    -------
    K_tilde : np.ndarray
        Centered kernel matrix.
    """
    n = K.shape[0]
    m = K.shape[1]

    identity_left = np.eye(n)
    identity_right = np.eye(m)

    h_left = np.ones((n, n)) / n
    h_right = np.ones((m, m)) / m

    K_tilde = (identity_left - h_left) @ K @ (identity_right - h_right)

    return K_tilde
