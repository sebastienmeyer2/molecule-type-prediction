"""Main file to run experiments."""


import os

from datetime import datetime

import argparse

import numpy as np
import pandas as pd

import optuna


from engine.objective import Objective
from utils.misc import set_seed, str2bool
from utils.data import load_graph_data


def run(
    seed: int = 42,
    data_dir: str = "data",
    gridsearch_subset: float = 0.2,
    kernel_name: str = "sum",
    model_name: str = "logreg",
    normalize: bool = True,
    center: bool = False,
    n_trials: int = 0,
    n_folds: int = 5,
    eval_metric: str = "roc_auc",
    verbose: int = False,
    submission: int = True,
    results_dir: str = "results"
):
    """Run experiment.

    Parameters
    ----------
    seed : int, default=42
        Seed to use everywhere for reproducibility.

    data_dir : str, default="data"
        Directory where data is stored.

    gridsearch_subset : float, default=0.2
        Select a subset of the training set to fasten cross validation.

    kernel_name : str, default="sum"
        Name of the kernel following package usage.

    model_name : str, default="logreg"
        Name of the model following package usage.

    normalize : bool, default=True
        If True, the kernel matrices are normalized before training and prediction.

    center : bool, default=False
        If True, the kernel matrices are centered before training and prediction.

    n_trials : int, default=0
        Number of hyperparameter optimization trials.

    n_folds : int, default=5
        Number of cross-validation splits.

    eval_metric : {"accuracy", "roc_auc"}, default="roc_auc"
        Evaluation metric for gridsearch.

    verbose : bool, default=False
        Verbosity.

    submission : bool, default=True
        If True, a submission file will be produced after hyperparameter optimization.

    results_dir : str, default="results"
        Directory where submission files are stored.
    """
    # Fix seed
    set_seed(seed)

    # Get data
    graph_list_train, y_train, graph_list_test = load_graph_data(data_dir=data_dir)

    # Research grid parameters
    study_id = datetime.now().strftime("%d-%m-%y_%H-%M-%S")

    # Create Objective
    objective = Objective(
        seed, kernel_name, model_name, graph_list_train, y_train, graph_list_test,
        gridsearch_subset=gridsearch_subset, normalize=normalize, center=center, n_folds=n_folds,
        eval_metric=eval_metric, verbose=verbose
    )

    # Initialize optuna study object
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(study_name=study_id, direction="maximize", sampler=sampler)

    # Run whole gridsearch
    study.optimize(objective, n_trials=n_trials)

    best_params = {}

    if n_trials > 0:

        # Best found parameters
        best_params.update(study.best_trial.params)

        # Print summary
        summary = []

        summary.append("\n===== OPTUNA GRID SEARCH SUMMARY =====\n")
        summary.append(f"Model is {model_name}.\n")
        summary.append("\n")
        summary.append(f"Cross-validation training {eval_metric}: {objective.best_score}.\n")
        summary.append("\n")
        summary.append(f"Current params are:\n {best_params}\n")
        summary.append("=========================================\n")

        print("".join(summary))

    if submission:

        # Create the model with best known parameters and predict logits
        best_params.update({"kernel__verbose": True, "model__verbose": True})
        objective.set_params(best_params)

        y_pred_logit = objective.train_predict_logit()

        y_sub = pd.DataFrame({"Predicted": y_pred_logit})
        y_sub.index += 1  # reset index

        # Save file in the requested format
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        filename = f"{results_dir}/trial{study_id}"
        filename += f"_cv_{eval_metric[:3]}_{np.around(objective.best_score, 3)}.csv"

        y_sub.to_csv(filename, header=True, index_label="Id")

        print(f"=====> A trial file has been created under {filename}.")


if __name__ == "__main__":

    # Command lines
    PARSER_DESC = "Main file to run experiments."
    PARSER = argparse.ArgumentParser(description=PARSER_DESC)

    # Seed
    PARSER.add_argument(
        "--seed",
        default=42,
        type=int,
        help="""
             Seed to use everywhere for reproducibility. Default: 42.
             """
    )

    # Data
    PARSER.add_argument(
        "--data-dir",
        default="data",
        type=str,
        help="""
             Directory where data is stored. Default: "data".
             """
    )

    PARSER.add_argument(
        "--gridsearch-subset",
        default=0.2,
        type=float,
        help="""
             Select a subset of the training set to fasten cross validation. Default: 0.2.
             """
    )

    # Kernel and model
    PARSER.add_argument(
        "--kernel-name",
        default="sum",
        type=str,
        help="""
             Name of the kernel following package usage. Default: "sum".
             """
    )

    PARSER.add_argument(
        "--model-name",
        default="logreg",
        type=str,
        help="""
             Name of the model following package usage. Default: "logreg".
             """
    )

    PARSER.add_argument(
        "--normalize",
        default="True",
        type=str2bool,
        help="""
             If "True", kernels matrices are normalized to zero mean and unit variance.
             Default: "True".
             """
    )

    PARSER.add_argument(
        "--center",
        default="False",
        type=str2bool,
        help="""
             If "True", kernels matrices are centered. Default: "False".
             """
    )

    # Gridsearch
    PARSER.add_argument(
        "--trials",
        default=0,
        type=int,
        help="""
             Choose the number of gridsearch trials. Default: 0.
             """
    )

    PARSER.add_argument(
        "--folds",
        default=5,
        type=int,
        help="""
             Number of cross-validation splits. Default: 5.
             """
    )

    PARSER.add_argument(
        "--eval-metric",
        default="roc_auc",
        type=str,
        choices=["accuracy", "roc_auc"],
        help="""
             Evaluation metric for gridsearch. Default: "roc_auc".
             """
    )

    PARSER.add_argument(
        "--verbose",
        default="False",
        type=str2bool,
        help="""
             Verbosity. Default: "False".
             """
    )

    # Submission
    PARSER.add_argument(
        "--submission",
        default="True",
        type=str2bool,
        help="""
             If "True", a submission file will be produced after hyperparameter optimization.
             Default: "True".
             """
    )

    PARSER.add_argument(
        "--results-dir",
        default="results",
        type=str,
        help="""
             Directory where submission files are stored. Default: "results".
             """
    )

    # End of command lines
    ARGS = PARSER.parse_args()

    # Run the experiment
    run(
        seed=ARGS.seed,
        data_dir=ARGS.data_dir,
        gridsearch_subset=ARGS.gridsearch_subset,
        kernel_name=ARGS.kernel_name,
        model_name=ARGS.model_name,
        normalize=ARGS.normalize,
        center=ARGS.center,
        n_trials=ARGS.trials,
        n_folds=ARGS.folds,
        eval_metric=ARGS.eval_metric,
        verbose=ARGS.verbose,
        submission=ARGS.submission,
        results_dir=ARGS.results_dir
    )
