import phik
import numpy as np
import pandas as pd
from tqdm import tqdm

from phik.binning import create_correlation_overview_table, hist2d_from_array
from phik.bivariate import phik_from_chi2
from phik.statistics import estimate_simple_ndof

import itertools
from scipy.stats import power_divergence


def phik_from_hist2d(observed: np.ndarray, expected: np.ndarray, noise_correction: bool = True) -> float:
    """
    correlation coefficient of bivariate gaussian derived from chi2-value

    Chi2-value gets converted into correlation coefficient of bivariate gauss
    with correlation value rho, assuming giving binning and number of records.
    Correlation coefficient value is between 0 and 1.

    Bivariate gaussian's range is set to [-5,5] by construction.

    :param observed: 2d-array observed values
    :param expected: 2d-array expected values
    :param bool noise_correction: apply noise correction in phik calculation
    :returns float: correlation coefficient phik
    """
    if isinstance(observed, pd.DataFrame):
        observed = observed.values
    if isinstance(expected, pd.DataFrame):
        expected = expected.values

    # important to ensure that observed and expected have same normalization
    expected = expected * (np.sum(observed) / np.sum(expected))

    # chi2 contingency test
    chi2 = chi_square(observed, expected, lambda_='pearson')

    # noise pedestal
    endof = estimate_simple_ndof(observed) if noise_correction else 0
    pedestal = endof
    if pedestal < 0:
        pedestal = 0

    # phik calculation adds noise pedestal to theoretical chi2
    return phik_from_chi2(chi2, observed.sum(), *observed.shape, pedestal=pedestal)


def chi_square(observed, expected, correction=True, lambda_=None):
    """ Calculate chi square between observed and expected 2d matrix

    :param observed:
    :param expected:
    :param correction:
    :param lambda_:
    :return:
    """
    observed = np.asarray(observed)
    if np.any(observed < 0):
        raise ValueError("All values in `observed` must be nonnegative.")
    if observed.size == 0:
        raise ValueError("No data; `observed` has size 0.")

    expected = np.asarray(expected)

    terms = np.divide((observed.astype(np.float64) - expected) ** 2, expected,
                      out=np.zeros_like(expected), where=expected != 0)
    return np.sum(terms)


def phik_matrix(X_obs, X_exp):
    """ Calculate phik matrix

    :param X_obs: array of observed data points
    :param X_exp: array of expected data points
    :return: phik matrix
    """
    assert X_obs.shape[1] == X_exp.shape[1]

    n_unique = [len(np.unique(X_obs[:, i])) for i in range(X_obs.shape[1])]

    phik_list = []
    for i, j in tqdm(itertools.combinations_with_replacement(range(X_obs.shape[1]), 2)):
        if i == j:
            phik_list.append((i, j, 1.))
            continue
        elif n_unique[i] == 1 or n_unique[j] == 1:
            phik_list.append((i, j, 0.))
            continue
        expected = hist2d_from_array(X_exp[:, i], X_exp[:, j], interval_cols=[])
        observed = hist2d_from_array(X_obs[:, i], X_obs[:, j], interval_cols=[])
        expected = make_equal_shape(observed, expected)
        phik_list.append((i, j, phik_from_hist2d(observed, expected)))

    phik_overview = create_correlation_overview_table(phik_list)
    return phik_overview


def make_equal_shape(observed, expected):
    """ Sometimes expected histogram shape need filling

    :param observed:
    :param expected:
    :return:
    """
    o_cols = observed.columns.tolist()
    e_cols = expected.columns.tolist()
    o_cols_missing = list(set(e_cols) - set(o_cols))
    e_cols_missing = list(set(o_cols) - set(e_cols))

    o_idx = observed.index.tolist()
    e_idx = expected.index.tolist()
    o_idx_missing = list(set(e_idx) - set(o_idx))
    e_idx_missing = list(set(o_idx) - set(e_idx))

    # make expected columns equal to observed
    for c in o_cols_missing:
        observed[c] = 0.0
    for c in e_cols_missing:
        expected[c] = 0.0
    observed.columns = sorted(observed.columns)
    expected.columns = sorted(expected.columns)
    assert len(observed.columns) == len(expected.columns)

    # make expected index equal to observed
    for i in o_idx_missing:
        observed.loc[i] = np.zeros(len(observed.columns))
    for i in e_idx_missing:
        expected.loc[i] = np.zeros(len(expected.columns))
    assert len(observed.index) == len(expected.index)

    return expected