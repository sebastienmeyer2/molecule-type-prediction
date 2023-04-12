"""Initialize kernels and models based on their name and parameters."""


from typing import Any, Dict


from kernels import (
    Kernel, CountingKernel, EdgeHistogramKernel, NodeHistogramKernel, OrderWalkKernel,
    GeometricWalkKernel, ShortestPathKernel, SumKernel
)
from models import Model, KernelLogisticRegression, KernelRidgeRegression, KernelSVC


SUPPORTED_KERNELS_NAMES = [
    "count", "edge_histogram", "node_histogram", "order_walk", "geometric_walk", "shortest_path",
    "sum"
]
SUPPORTED_MODELS_NAMES = ["linreg", "logreg", "svc"]


def create_kernel(kernel_name: str, kernel_params: Dict[str, Any]) -> Kernel:
    """Instantiate a kernel.

    Parameters
    ----------
    kernel_name : str
        Name of the kernel following package usage.

    kernel_params : dict of {str: any}
        A dictionary of parameters for chosen **kernel_name**.

    Returns
    -------
    kernel : Kernel
        Corresponding kernel from the catalogue.

    Raises
    ------
    ValueError
        If the **kernel_name** is not supported.
    """
    if kernel_name not in SUPPORTED_KERNELS_NAMES:

        err_msg = f"Unknown kernel {kernel_name}."
        raise ValueError(err_msg)

    # Instantiate kernel
    if kernel_name == "count":

        kernel = CountingKernel(**kernel_params)

    elif kernel_name == "node_histogram":

        kernel = NodeHistogramKernel(**kernel_params)

    elif kernel_name == "edge_histogram":

        kernel = EdgeHistogramKernel(**kernel_params)

    elif kernel_name == "order_walk":

        kernel = OrderWalkKernel(**kernel_params)

    elif kernel_name == "geometric_walk":

        kernel = GeometricWalkKernel(**kernel_params)

    elif kernel_name == "shortest_path":

        kernel = ShortestPathKernel(**kernel_params)

    elif kernel_name == "sum":

        kernel = SumKernel(**kernel_params)

    return kernel


def create_model(model_name: str, model_params: Dict[str, Any]) -> Model:
    """Create a model.

    Parameters
    ----------
    model_name : str
        Name of the model following package usage.

    model_params : dict of {str: any}
        A dictionary of parameters for chosen **model_name**.

    Returns
    -------
    model : Model
        Corresponding model from the catalogue.

    Raises
    ------
    ValueError
        If the **model_name** is not supported.
    """
    if model_name not in SUPPORTED_MODELS_NAMES:

        err_msg = f"Unknown model {model_name}."
        raise ValueError(err_msg)

    # Instantiate model
    if model_name == "linreg":

        model = KernelRidgeRegression(**model_params)

    elif model_name == "logreg":

        model = KernelLogisticRegression(**model_params)

    elif model_name == "svc":

        model = KernelSVC(**model_params)

    return model
