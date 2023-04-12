"""Gridsearch param grid."""


from typing import Any, Dict, Tuple

from optuna import Trial


def optuna_param_grid(
    trial: Trial, kernel_name: str, model_name: str, verbose: bool = False
) -> Dict[str, Any]:
    """Create a param grid for *optuna* usage.

    Parameters
    ----------
    trial : optuna.Trial
        An instance of Trial object from *optuna* package to handle parameters search.

    kernel_name : str
        Name of the kernel following package usage.

    model_name : str
        Name of the model following package usage.

    verbose : bool, default=False
        Verbosity.

    Returns
    -------
    params : dict of {str: any}
        A dictionary of parameters for both the kernel and the model.
    """
    params = {}

    # Initialize kernel params
    params["kernel__verbose"] = trial.suggest_categorical("kernel__verbose", [verbose])

    if kernel_name == "count":

        # Bandwidth
        params["kernel__sigma"] = trial.suggest_float("kernel__sigma", 0.01, 100, log=True)

    elif kernel_name == "node_histogram":

        # Label enrichment
        params["kernel__morgan_steps"] = trial.suggest_categorical("kernel__morgan_steps", [0])
        params["kernel__wl_steps"] = trial.suggest_categorical("kernel__wl_steps", [0])

        # Bandwidth
        params["kernel__sigma"] = trial.suggest_float("kernel__sigma", 0.01, 100, log=True)

    elif kernel_name == "edge_histogram":

        # Bandwidth
        params["kernel__sigma"] = trial.suggest_float("kernel__sigma", 0.01, 100, log=True)

    elif kernel_name == "order_walk":

        # Label enrichment
        params["kernel__morgan_steps"] = trial.suggest_categorical("kernel__morgan_steps", [0])
        params["kernel__wl_steps"] = trial.suggest_categorical("kernel__wl_steps", [0])

        # Walk parameters
        params["kernel__order"] = trial.suggest_int("kernel__order", 1, 4)

    elif kernel_name == "geometric_walk":

        # Label enrichment
        params["kernel__morgan_steps"] = trial.suggest_categorical("kernel__morgan_steps", [0])
        params["kernel__wl_steps"] = trial.suggest_categorical("kernel__wl_steps", [0])

        # Walk parameters
        params["kernel__order"] = trial.suggest_int("kernel__order", 1, 4)
        params["kernel__beta"] = trial.suggest_float("kernel__beta", 1e-3, 10, log=True)

    elif kernel_name == "shortest_path":

        # Label enrichment
        params["kernel__morgan_steps"] = trial.suggest_categorical("kernel__morgan_steps", [0])
        params["kernel__wl_steps"] = trial.suggest_categorical("kernel__wl_steps", [0])

    elif kernel_name == "sum":

        # Bandwidth
        params["kernel__sigma"] = trial.suggest_float("kernel__sigma", 0.01, 100, log=True)
        params["kernel__sigma_m"] = trial.suggest_float("kernel__sigma_m", 0.01, 100, log=True)
        params["kernel__sigma_wl"] = trial.suggest_float("kernel__sigma_wl", 0.01, 100, log=True)

        # Walk parameters
        params["kernel__beta"] = trial.suggest_float("kernel__beta", 1e-3, 10, log=True)
        params["kernel__beta_m"] = trial.suggest_float("kernel__beta_m", 1e-3, 10, log=True)
        params["kernel__beta_wl"] = trial.suggest_float("kernel__beta_wl", 1e-3, 10, log=True)

    # Initialize model params
    params["model__verbose"] = trial.suggest_categorical("model__verbose", [verbose])

    if model_name == "linreg":

        # Regularization
        params["model__lambd"] = trial.suggest_float("model__lambd", 1e-6, 1e-2, log=True)

    elif model_name == "logreg":

        # Regularization
        params["model__lambd"] = trial.suggest_float("model__lambd", 1e-6, 1e-2, log=True)

    elif model_name == "svc":

        # Regularization
        params["model__C"] = trial.suggest_float("model__C", 0.1, 1e6, log=True)

    elif model_name == "mkl":

        # Regularization
        params["model__C"] = trial.suggest_float("model__C", 0.1, 1e6, log=True)

    return params


def extract_kernel_and_model_params(params: Dict[str, Any]) -> Tuple[Dict[str, Any], ...]:
    """Extract parameters linked to the kernel and to the model.

    Parameters
    ----------
    params : dict of {str: any}
        A dictionary of parameters for both the kernel and the model.

    Returns
    -------
    kernel_params : dict of {str: any}
        A dictionary of parameters for the kernel.

    model_params : dict of {str: any}
        A dictionary of parameters for the model.
    """
    kernel_params = {}
    model_params = {}

    for param, param_value in params.items():

        model_or_kernel = param.split("__")

        if len(model_or_kernel) <= 1:

            kernel_params[param] = param_value
            model_params[param] = param_value

        else:

            sub_param = "".join(model_or_kernel[1:])

            if model_or_kernel[0] == "kernel":

                kernel_params[sub_param] = param_value

            else:

                model_params[sub_param] = param_value

    return kernel_params, model_params
