"""Run optimized cross validation for parameters gridsearch."""


from typing import Any, Dict, List

import numpy as np

from networkx import Graph

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from optuna.trial import Trial


from engine.gridsearch import extract_kernel_and_model_params, optuna_param_grid
from engine.hub import create_kernel, create_model
from utils.kernel_ops import normalize_kernel, center_kernel


class Objective():
    """An Objective class to wrap trials.

    General class that implements call functions for gridsearch algorithms.
    """
    def __init__(
        self, seed: int, kernel_name: str, model_name: str, graph_list_train: List[Graph],
        y_train: np.ndarray, graph_list_test: List[Graph], gridsearch_subset: float = 0.2,
        normalize: bool = False, center: bool = False, n_folds: int = 5,
        eval_metric: str = "roc_auc", verbose: bool = False
    ):
        """Create an objective.

        Parameters
        ----------
        seed : int
            Seed to use everywhere for reproducibility.

        kernel_name : str
            Name of the kernel following package usage.

        model_name : str
            Name of the model following package usage.

        graph_list_train : list of networkx.Graph
            List of training graphs.

        y_train : np.ndarray
            Training classes.

        graph_list_test : list of networkx.Graph
            List of test graphs.

        gridsearch_subset : float, default=0.2
            Select a subset of the training set to fasten cross validation.

        normalize : bool, default=False
            If True, the kernel matrices are normalized before training and prediction.

        center : bool, default=False
            If True, the kernel matrices are centered before training and prediction.

        n_folds : int, default=5
            Number of cross-validation splits.

        eval_metric : {"accuracy", "roc_auc"}, default="roc_auc"
            Evaluation metric for gridsearch.

        verbose : bool, default=False
            Verbosity.
        """
        # Handling randomness
        self.seed = seed

        # Data
        self.graph_list_train = graph_list_train
        self.y_train = y_train
        self.graph_list_test = graph_list_test

        self.nb_training_samples = len(self.y_train)

        # Only return a subset of the dataset
        self.gridsearch_subset = gridsearch_subset
        assert 1. >= gridsearch_subset >= 0., f"Subset {gridsearch_subset} must be in [0, 1]."

        if self.gridsearch_subset < 1.:

            _, self.graph_subset_train, _, self.y_subset_train = train_test_split(
                self.graph_list_train, self.y_train, test_size=self.gridsearch_subset,
                shuffle=True, random_state=self.seed
            )

        else:

            self.graph_subset_train = self.graph_list_train
            self.y_subset_train = self.y_train

        # Model and decision values
        self.kernel_name = kernel_name
        self.model_name = model_name
        self.normalize = normalize
        self.center = center

        # Model parameters
        self.params: Dict[str, Any] = {}

        # Keep best results in memory
        self.n_folds = n_folds
        self.eval_metric = eval_metric
        self.logit = self.eval_metric == "roc_auc"
        self.eval_fn = accuracy_score if self.eval_metric == "accuracy" else roc_auc_score
        self.best_score = 0.

        # Verbosity
        self.verbose = verbose

    def get_params(self) -> Dict[str, Any]:
        """Get parameters.

        Returns
        -------
        params : dict of {str: any}
            A dictionary of parameters for both the kernel and the model.
        """
        return self.params

    def set_params(self, params: Dict[str, Any]):
        """Set parameters.

        Parameters
        ----------
        params : dict of {str: any}
            A dictionary of parameters for both the kernel and the model.
        """
        self.params = params

    def run_cross_val(self) -> float:
        """Initialize a model and run cross-validation on training set.

        Returns
        -------
        mean_score : float
            Mean target metric value of current model during cross-validation.
        """
        # Precompute the kernel
        kernel_params, model_params = extract_kernel_and_model_params(self.params)
        kernel = create_kernel(self.kernel_name, kernel_params)

        K = kernel.kernel(self.graph_subset_train)
        if self.normalize:
            K = normalize_kernel(K)
        if self.center:
            K = center_kernel(K)

        # Cross validation parameters
        mean_score = 0.

        kf = StratifiedKFold(n_splits=self.n_folds)

        for train_indices, eval_indices in kf.split(self.graph_subset_train, self.y_subset_train):

            # Data splitting
            fold_y_train = self.y_subset_train[train_indices]
            fold_y_eval = self.y_subset_train[eval_indices]

            fold_K_train = K[train_indices, :][:, train_indices]
            fold_K_eval = K[eval_indices, :][:, train_indices]

            # Initialize model
            model = create_model(self.model_name, model_params)

            # Fit the model
            model.fit(fold_K_train, fold_y_train)

            # Predict
            if self.logit:
                fold_y_pred = model.predict_logit(fold_K_eval)
            else:
                fold_y_pred = model.predict(fold_K_eval)

            if np.sum(~np.isfinite(fold_y_pred)) > 0 or len(np.unique(fold_y_eval)) <= 1:
                fold_eval_score = np.nan
            else:
                fold_eval_score = self.eval_fn(fold_y_eval, fold_y_pred)

            mean_score += fold_eval_score

        # Compute mean values of selected metrics
        mean_score /= self.n_folds

        # Keep best values in memory
        if mean_score > self.best_score:

            self.best_score = mean_score

        return mean_score

    def train_predict_logit(self):
        """Initialize a kernel, a model and predict logits.

        Returns
        -------
        y_pred_logits : ndarray
            Predicted logits.
        """
        # Precompute the kernel
        kernel_params, model_params = extract_kernel_and_model_params(self.params)
        kernel = create_kernel(self.kernel_name, kernel_params)

        n_train = len(self.graph_list_train)

        graph_list = []
        for graph in self.graph_list_train:
            graph_list.append(graph)
        for graph in self.graph_list_test:
            graph_list.append(graph)

        K = kernel.kernel(graph_list)
        if self.normalize:
            K = normalize_kernel(K)
        if self.center:
            K = center_kernel(K)
        K_train = K[:n_train, :n_train]
        K_eval = K[n_train:, :n_train]

        # Initialize model
        model = create_model(self.model_name, model_params)

        # Fit the model
        model.fit(K_train, self.y_train)

        # Predict
        y_pred_logits = model.predict_logit(K_eval)

        return y_pred_logits

    def __call__(self, trial: Trial) -> float:
        """Run a trial using *optuna* package.

        Parameters
        ----------
        trial : optuna.Trial
            An instance of Trial object from *optuna* package to handle parameters search.

        Returns
        ----------
        trial_target : float
            Target metric value of current model during trial.
        """
        # Initialize parameter grid via optuna
        optuna_params = optuna_param_grid(
            trial, self.kernel_name, self.model_name, verbose=self.verbose
        )

        self.set_params(optuna_params)

        # Run cross val evaluation
        trial_target = self.run_cross_val()

        return trial_target
