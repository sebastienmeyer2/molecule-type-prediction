"""Kernel models."""


from .base import Model

from .ridge_regression import KernelRidgeRegression
from .logistic_regression import KernelLogisticRegression
from .svc import KernelSVC


__all__ = [
    "Model",

    "KernelRidgeRegression",
    "KernelLogisticRegression",
    "KernelSVC"
]
