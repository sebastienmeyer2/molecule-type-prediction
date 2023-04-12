"""Miscellaneous."""


from typing import Dict, Hashable, List, Tuple

import random

import argparse

import numpy as np


def set_seed(seed: int = 42):
    """Fix seed for current run.

    Parameters
    ----------
    seed : int, default=42
        Seed to use everywhere for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)  # pandas seed is numpy seed


def str2bool(v: str) -> bool:
    """An easy way to handle boolean options.

    Parameters
    ----------
    v : str
        Argument value.

    Returns
    -------
    str2bool(v) : bool
        Corresponding boolean value, if it exists.

    Raises
    ------
    argparse.ArgumentTypeError
        If the entry cannot be converted to a boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def assemble_histograms(
    X: List[Dict[Hashable, int]], Y: List[Dict[Hashable, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble two list of histograms.

    Parameters
    ----------
    X : list of dict of {hashable: int}
        For each graph, a dictionary counting the number of each label.

    Y : list of dict of {hashable: int}
        For each graph, a dictionary counting the number of each label.

    Returns
    -------
    X_arr : np.ndarray
        Corresponding feature array for **X**.

    Y_arr : np.ndarray
        Corresponding feature array for **Y**.
    """
    # Convert the labels to unique integers
    hash_to_int = {}
    max_length = 0

    for d in X:

        for obj in d.keys():

            if obj not in hash_to_int:

                hash_to_int[obj] = max_length
                max_length += 1

    for d in Y:

        for obj in d.keys():

            if obj not in hash_to_int:

                hash_to_int[obj] = max_length
                max_length += 1

    # Retrieve features for X
    X_arr = np.zeros((len(X), max_length))

    for i, d_i in enumerate(X):

        for label, count in d_i.items():

            X_arr[i, hash_to_int[label]] = count

    # Retrieve features for Y
    Y_arr = np.zeros((len(Y), max_length))

    for i, d_i in enumerate(Y):

        for label, count in d_i.items():

            Y_arr[i, hash_to_int[label]] = count

    return X_arr, Y_arr
