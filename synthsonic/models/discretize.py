from typing import Union

import numpy as np


def guess_column_bins(X: np.ndarray, bins="auto", min_bins=5, max_bins=100):
    """
    Args:
        X (np.array): data to guess the number of bins of
        bins (str): mode to guess (see numpy.histogram_bin_edges)
        min_bins (int): minimum number of bins
        max_bins (int): maximum number of bins

    Returns:
        bins (list): bins per column
    """
    def f(x, mode):
        if mode == "knuth":
            # https://docs.astropy.org/en/stable/api/astropy.stats.knuth_bin_width.html
            from astropy.stats import knuth_bin_width
            _, bin_edges = knuth_bin_width(x, return_bins=True)
            return bin_edges
        else:
            return np.histogram_bin_edges(x, bins=mode)

    guessed_bins = [
        max(
            min(
                len(f(X[:, col], mode=bins)) - 1,
                max_bins
            ),
            min_bins
        )
        for col in range(X.shape[1])
    ]

    return guessed_bins


def discretize(X: np.ndarray, n_bins: Union[int, list, tuple, np.ndarray]):
    """Transform

    Args:
        X: (uniform) data
        n_bins: number of bins, or bin per column

    Returns:
        discretized data
    """
    if isinstance(n_bins, int):
        n_bins = np.array([n_bins] * X.shape[1])
    elif isinstance(n_bins, (list, tuple)):
        n_bins = np.array(n_bins)

    bin_widths = 1. / n_bins
    X_num_discrete = np.floor(X / bin_widths[None, :])

    # ensure discretized values are between 0 and n_bins
    X_num_discrete = np.clip(X_num_discrete, a_min=0, a_max=n_bins - 1)
    return X_num_discrete


def inv_discretize(X, n_bins):
    """Reverse transform

    Args:
        X: discretized data
        n_bins: number of bins, or bin per column

    Returns:
        (uniform) data
    """
    if isinstance(n_bins, int):
        n_bins = np.array([n_bins] * X.shape[1])
    elif isinstance(n_bins, (list, tuple)):
        n_bins = np.array(n_bins)

    bin_width = 1. / n_bins
    X_rand = np.random.uniform(low=0., high=bin_width, size=X.shape)
    return X * bin_width + X_rand

