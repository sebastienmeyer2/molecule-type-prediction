"""Conversion and loss functions."""


import warnings

import numpy as np


def logit(y: np.ndarray) -> np.ndarray:
    """Compute logits.

    Parameters
    ----------
    y : np.ndarray
        Predicted probabilities.

    Returns
    -------
    np.ndarray
        Predicted logits.
    """
    return np.log(y / (1.0 - y))


def sigmoid(y: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function.

    Parameters
    ----------
    y : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Sigmoid evaluation of **y**.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return 1.0 / (1.0 + np.exp(-y))


def logistic_loss(y: np.ndarray) -> np.ndarray:
    """Compute the logistic loss.

    Parameters
    ----------
    y : np.ndarray
        Predicted classes.

    Returns
    -------
    np.ndarray
        Logistic loss.
    """
    return - np.log(sigmoid(y) + 1e-6)


def logistic_loss_p(y: np.ndarray) -> np.ndarray:
    """Compute the first derivative of the logistic loss.

    Parameters
    ----------
    y : np.ndarray
        Predicted classes.

    Returns
    -------
    np.ndarray
        First derivative of the logistic loss.
    """
    return - sigmoid(y)


def logistic_loss_pp(y: np.ndarray) -> np.ndarray:
    """Compute the second derivative of the logistic loss.

    Parameters
    ----------
    y : np.ndarray
        Predicted classes.

    Returns
    -------
    np.ndarray
        Second derivative of the logistic loss.
    """
    return sigmoid(y) * sigmoid(-y)
