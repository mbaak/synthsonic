from __future__ import annotations

import itertools
import logging
import random
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pgmpy.estimators import TreeSearch
from pgmpy.inference import BayesianModelProbability
from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
from scipy import interpolate
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from tqdm import tqdm
from xgboost import XGBClassifier

from synthsonic.models.discretize import discretize, guess_column_bins, inv_discretize
from synthsonic.models.kde_quantile_tranformer import KDEQuantileTransformer
from synthsonic.models.phik_utils import phik_matrix


class KDECopulaNNPdf(BaseEstimator):
    """Kernel Density Estimation Copula NN PDF, models any data distribution

    KDECopulaNNPdf applies 7 steps to model any data distribution, where the variables are continuous:
    1. Smoothing of non-unique spikes in the input data
    2. KDE quantile transformation of each variable to a normal distribution
    3. PCA transformation of all normalized variables
    4. KDE quantile transformation of pca variables to uniform distributions
    5. non-linear feature ordering to select top-n non-linear variables
    6. NN classifier to model the copula space of the selected top-n variables.
    7. Recalibration of classifier probabilities.
    """

    def __init__(
        self,
        numerical_columns=[],
        categorical_columns=[],
        distinct_threshold=-1,
        n_quantiles=500,
        mirror_left=None,
        mirror_right=None,
        rho=0.5,
        n_adaptive=1,
        x_min=None,
        x_max=None,
        do_PCA=True,
        ordering="pca",
        min_pca_variance=1.00,
        min_mutual_information=0,
        min_phik_correlation=0,
        n_nonlinear_vars=None,
        force_uncorrelated=False,
        # clf=MLPClassifier(random_state=0, max_iter=1000, early_stopping=True),
        clf=XGBClassifier(
            n_estimators=250,
            reg_lambda=1,
            gamma=0,
            max_depth=9,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=0,
        ),
        random_state=0,
        use_inverse_qt=False,
        use_KDE=False,
        n_uniform_bins: Union[int, list, tuple, np.ndarray, str] = 25,
        n_calibration_bins=100,
        test_size=0.35,
        copy=True,
        clffitkw={},
        estimator_type="tan",
        edge_weights_fn="mutual_info",
        class_node=None,
        root_node=None,
        bm_fit_args={},  # ex: estimator=BayesianEstimator, prior_type='dirichlet', pseudo_counts=0.1
        apply_calibration=True,
        isotonic_increasing="auto",  # or True
        verbose=True,
    ):
        """Parameters of the KDECopulaNNPdf class

        KDECopulaNNPdf applies 7 steps to model any data distribution, where the variables are continuous:
        1. Smoothing of non-unique spikes in the input data
        2. KDE quantile transformation of each variable to a normal distribution
        3. PCA transformation of all normalized variables
        4. KDE quantile transformation of pca variables to uniform distributions
        5. non-linear feature ordering to select top-n non-linear variables
        6. NN classifier to model the copula space of the selected top-n variables.
        7. Recalibration of classifier probabilities.

        :param int n_quantiles: number of quantiles/bins used in output histogram. If greater than number of samples,
            this is reset to number of samples. Default is 1000.
        :param mirror_left: array. Mirror the data on a value on the left to counter signal leakage.
            Default is None, which is no mirroring.
        :param mirror_right: array. Mirror the data on a value on the right to counter signal leakage.
            Default is None, which is no mirroring.
        :param float rho: KDE bandwidth scale parameter. default is 0.5.
        :param int n_adaptive: KDE number of adaptive iterations to be applied to improve the band width. default is 1.
        :param x_min: array. minimum value of pdf's x range. default is None (= - inf)
        :param x_max: array. maximum value of pdf's x range. default is None (= + inf)
        :param do_PCA: if False, do not perform the PCA variable transformation.
        :param ordering: ordering of variables after pca transformation. Based on 'pca' importance or pairs of variables
            with highest mutual information 'mi'. default is 'pca'.
        :param min_pca_variance: minimal pca variance to be explained by non-linear variables. This selects top-n
            non-linear variables that are to be modelled. Used if 'ordering' is 'pca'.
        :param min_mutual_information: minimal mutual information required for selected pairs of variables. This selects
            top-n non-linear variables that are to be modelled. Used if 'ordering' is 'mi'.
        :param min_phik_correlation: minimal phik correlation required for selected pairs of variables. This selects
            top-n non-linear variables that are to be modelled. Used if 'ordering' is 'phik'.
        :param n_nonlinear_vars: number of non-linear variables that are to be modelled. This overrides 'ordering'.
            Default is None.
        :param force_uncorrelated: if True, force that all variables are treated as uncorrelated. So no PCA is applied
            and no modelling of non-linear effects.
        :param clf: classifier used to model non-linear variable dependencies. Default is out-of-the-box
            MLPClassifier() from sklearn.
        :param int random_state: when an integer, the seed given random generator.
        :param bool use_inverse_qt: Default is False. If true, KDE is not used in inverse transformation.
        :param bool use_KDE: Default is True. If false, KDE smoothing is off, using default quantile transformation.
        :param copy: Copy the data before transforming. Default is True.
        :param n_uniform_bins: number of bins to discretize numeric variables. If int, then used of all columns,
            if list/tuple[int] or np.ndarray, then per variable/column. If str, then auto-guessed using numpy's histogram_bin_edges.
            (See https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html for possible values) or
            "knuth" for Knuth [1] (make sure to install astropy)

            [1] K.H. “Optimal Data-Based Binning for Histograms”. arXiv:0605197, 2006
                https://arxiv.org/pdf/physics/0605197.pdf
        """
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.distinct_threshold = distinct_threshold
        self.n_quantiles = n_quantiles
        self.mirror_left = mirror_left
        self.mirror_right = mirror_right
        self.rho = rho
        self.n_adaptive = n_adaptive
        self.x_min = x_min
        self.x_max = x_max
        self.do_PCA = do_PCA
        self.ordering = ordering
        self.min_pca_variance = min_pca_variance
        self.min_mutual_information = min_mutual_information
        self.min_phik_correlation = min_phik_correlation
        self.n_nonlinear_vars = n_nonlinear_vars
        self.force_uncorrelated = force_uncorrelated
        self.copy = copy
        self.random_state = random_state
        self.use_inverse_qt = use_inverse_qt
        self.use_KDE = use_KDE
        self.min_pdf_value = 1e-100
        self.max_scale_value = 5000
        self.clf = clf
        self.logger = logging.getLogger(self.__class__.__name__)
        self.n_uniform_bins = n_uniform_bins
        # Calibration
        self.apply_calibration = apply_calibration
        self.n_calibration_bins = n_calibration_bins
        self.test_size = test_size
        self.clffitkw = clffitkw
        self.isotonic_increasing = isotonic_increasing

        # Bayesian network
        self.estimator_type = estimator_type
        self.edge_weights_fn = edge_weights_fn
        self.class_node = class_node
        self.root_node = root_node
        self.bm_fit_args = bm_fit_args

        # Logging
        self.verbose = verbose

        # basic checks on attributes
        if self.min_pca_variance < 0 or self.min_pca_variance > 1:
            raise ValueError(
                "Invalid value for 'min_pca_variance': %f. Should be a float in (0,1]." % self.min_pca_variance
            )
        if self.min_mutual_information < 0:
            raise ValueError(
                "Invalid value for 'min_mutual_information': %f. Should be greater than zero."
                % self.min_mutual_information
            )
        if self.min_phik_correlation < 0 or self.min_phik_correlation > 1:
            raise ValueError(
                "Invalid value for 'min_phik_correlation': %f. Should be a float in [0,1]." % self.min_phik_correlation
            )
        if self.n_nonlinear_vars is not None:
            if not isinstance(self.n_nonlinear_vars, (int, float, np.number)) or self.n_nonlinear_vars < 1:
                raise ValueError(
                    "Invalid value for 'n_nonlinear_vars': %d. Should be an int greater than zero."
                    % self.n_nonlinear_vars
                )
        if self.ordering not in ["pca", "mi", "phik", ""]:
            raise ValueError("Non-linear variables should be ordered by 'pca', mutual information 'mi', or 'phik'.")

    def fit(self, X, y=None):
        """Fit the Kernel Density Copula NN model on the data.

        :param X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :param y: None
            Ignored. This parameter exists only for compatibility with
            :class:`sklearn.pipeline.Pipeline`.
        :return: self : object
            Returns instance of object.
        """
        # sort columns into numerical and categorical
        for i in range(X.shape[1]):
            if i in self.categorical_columns or i in self.numerical_columns:
                continue
            if len(np.unique(X[:, i])) < self.distinct_threshold:
                self.categorical_columns.append(i)
            else:
                self.numerical_columns.append(i)
        self.logger.info(
            f"Processing {len(self.numerical_columns):d} numerical and {len(self.categorical_columns):d} categorical columns"
        )
        X_cat = X[:, self.categorical_columns]
        X_num = X[:, self.numerical_columns]
        if len(self.numerical_columns) > 0:
            X_num = check_array(X_num, copy=False, dtype=FLOAT_DTYPES, force_all_finite="allow-nan")

        # sample profiles
        n_samples, n_features = X_num.shape

        if self.n_quantiles > n_samples:
            self.logger.warning(
                f"n_quantiles ({self.n_quantiles}) is greater than the total number of samples ({n_samples}). n_quantiles is set to num samples."
            )
            self.n_quantiles = n_samples
        if self.n_quantiles < 1:
            self.logger.warning(f"n_quantiles ({self.n_quantiles}) is lower than 1. n_quantiles is set to 1.")
            self.n_quantiles = 1
        if self.n_nonlinear_vars is not None and self.n_nonlinear_vars > n_features:
            self.logger.warning(
                f"n_nonlinear_vars ({self.n_nonlinear_vars:d}) is greater than the total number of features ({n_features:d}). n_nonlinear_vars is set to num features."
            )
            self.n_nonlinear_vars = n_features
        if not self.do_PCA and self.ordering == "pca":
            self.logger.warning(
                "Cannot order non-linear variables by pca (turned off). Switching to mutual information."
            )
            self.ordering = "mi"

        if self.do_PCA and n_features >= 2 and not self.force_uncorrelated:
            self.pipe_ = make_pipeline(
                KDEQuantileTransformer(
                    n_quantiles=self.n_quantiles,
                    output_distribution="normal",
                    mirror_left=self.mirror_left,
                    mirror_right=self.mirror_right,
                    rho=self.rho,
                    n_adaptive=self.n_adaptive,
                    x_min=self.x_min,
                    x_max=self.x_max,
                    copy=self.copy,
                    random_state=self.random_state,
                    use_inverse_qt=self.use_inverse_qt,
                    use_KDE=self.use_KDE,
                ),
                PCA(n_components=n_features, whiten=False, copy=self.copy, random_state=self.random_state),
                KDEQuantileTransformer(
                    n_quantiles=self.n_quantiles,
                    output_distribution="uniform",
                    rho=max(self.rho, 0.2),
                    n_adaptive=self.n_adaptive,
                    use_inverse_qt=True,
                    copy=self.copy,
                    random_state=self.random_state,
                ),
            )
        elif n_features >= 1:
            self.pipe_ = make_pipeline(
                KDEQuantileTransformer(
                    n_quantiles=self.n_quantiles,
                    output_distribution="uniform",
                    mirror_left=self.mirror_left,
                    mirror_right=self.mirror_right,
                    rho=self.rho,
                    n_adaptive=self.n_adaptive,
                    x_min=self.x_min,
                    x_max=self.x_max,
                    copy=self.copy,
                    random_state=self.random_state,
                    use_inverse_qt=self.use_inverse_qt,
                    use_KDE=self.use_KDE,
                )
            )

        self.logger.info("Transforming numerical variables.")
        X_uniform = self.pipe_.fit_transform(X_num) if n_features >= 1 else np.empty_like(X_num)

        self.logger.info("Configuring Bayesian Network (cat+num).")
        # discretize continuous variables; use these as input to model bayesian network
        if not isinstance(self.n_uniform_bins, (int, list, tuple, np.ndarray)):
            self.n_uniform_bins = guess_column_bins(X_uniform, bins=self.n_uniform_bins)
            self.logger.info(f"automatically set n_uniform_bins = {self.n_uniform_bins}")
        else:
            self.logger.info(f"n_uniform_bins = {self.n_uniform_bins}")

        # discretize numeric variables
        X_num_discrete = discretize(X_uniform, self.n_uniform_bins)

        # joining cat and num-discrete, then reorder to original order
        X_discrete = self._join_and_reorder(X_cat, X_num_discrete, self.categorical_columns, self.numerical_columns)
        self._configure_bayesian_network(X_discrete)

        self.logger.info("Configuring classifier.")
        X_trans = self._join_and_reorder(X_cat, X_uniform, self.categorical_columns, self.numerical_columns)
        self._configure_classifier(X_discrete, X_trans)

        return self

    def _configure_bayesian_network(self, X_discrete, plot_structure=False):
        """Distretize uniform variables and fit Bayesian network

        :param X_cat:
        :param X_uniform:
        :param bins:
        :return: X_discrete
        """
        # first find the bayesian network structure
        df = pd.DataFrame(X_discrete)
        # "tan" bayesian network needs string column names
        df.columns = [str(c) for c in df.columns]
        est = TreeSearch(df, root_node=df.columns[self.root_node] if self.root_node is not None else None)
        dag = est.estimate(
            estimator_type=self.estimator_type,
            class_node=str(self.class_node) if self.class_node is not None else None,
            show_progress=self.verbose,
            edge_weights_fn=self.edge_weights_fn,
        )

        if plot_structure:
            import networkx as nx
            from networkx.drawing.nx_agraph import graphviz_layout

            pos = graphviz_layout(dag, prog="dot")
            nx.draw(dag, pos, with_labels=True, arrowsize=10, node_size=600, alpha=0.8, font_weight="bold")
            plt.show()

        # model the conditional probabilities
        self.bn = BayesianModel(dag.edges())
        self.bn.fit(df, **self.bm_fit_args)
        # initialize sampler with fitted model
        self.bn_sampler = BayesianModelSampling(self.bn)
        self.bn_prob = BayesianModelProbability(self.bn)
        self.bn_ordering = [str(i) for i in range(X_discrete.shape[1])]

    def _join_and_reorder(self, X_cat, X_num, cat_order, num_order):
        X_join = np.concatenate([X_cat, X_num], axis=1)
        permutation = list(cat_order) + list(num_order)
        # permutation = [current_order.index(i) for i in range(len(current_order))]
        index = np.empty_like(permutation)
        index[permutation] = np.arange(len(permutation))
        return X_join[:, index]

    def _sample_bayesian_network(self, size=250000, add_uniform=True, seed=None, show_progress=True, X_bn=None):
        """Sample the Bayesian Network

        :param size:
        :param add_uniform:
        :param seed:
        :param show_progress:
        :param X_bn:
        :return:
        """
        # overwrite size if needed
        if X_bn is not None:
            size = X_bn.shape[0]

        # set random seed
        if seed is not None:
            np.random.seed(seed)

        if show_progress:
            self.logger.info(f"Generating {size} data points.")

        if X_bn is None:
            # generate sample
            df = self.bn_sampler.forward_sample(size=size, show_progress=show_progress, seed=seed)
            # "tan" bayesian network needs string column names; here convert back to ints
            df.columns = [int(c) for c in df.columns]
            X_bn = df[sorted(df.columns)].values
        if X_bn is not None and add_uniform and len(self.numerical_columns) > 0:
            # turn discrete numerical columns back to uniform
            X_bn[:, self.numerical_columns] = inv_discretize(X_bn[:, self.numerical_columns], self.n_uniform_bins)

        return X_bn

    def _configure_classifier(self, X_discrete, X_trans):
        """Configure the classifier

        First select non-linear variables to model using phik/mi, then fit the classifier with those variables.

        :param X_discrete: observed data with discrete uniform numerical variables
        :param X_trans: observed data with discrete uniform numerical variables
        """
        # determine number of features to use for nonlinear modelling
        n_sample = int(X_discrete.shape[0] * (1.0 - self.test_size)) + max(250000, X_discrete.shape[0])
        X_bn = self._sample_bayesian_network(size=n_sample, add_uniform=False, seed=self.random_state)
        self._configure_nonlinear_variables(X_discrete=X_discrete, X_expect=X_bn)

        # fit the classifier with selected non-linear variables (default is all)
        if self.n_vars_ >= 2 and not self.force_uncorrelated and self.clf is not None:
            self.logger.info(f"Fitting discriminative learner: selected {len(self.nonlinear_indices_)} features.")
            # reuse X_bn as test sample below, add uniform component here (generation from bn can be slow)
            X_bn = self._sample_bayesian_network(
                X_bn=X_bn, add_uniform=True, show_progress=False, seed=self.random_state
            )
            # this function captures in matrices the residual non-linearity after the transformations above.
            # note: residual vars are not used in classifier
            self._fit_classifier(
                X1=X_trans[:, self.nonlinear_indices_],
                X0=X_bn[:, self.nonlinear_indices_],
                bins=self.n_calibration_bins,
            )

    def _fit_classifier(self, X1, bins=None, X0=None):
        """The neural network below captures the residual non-linearity after the transformations above.

        :param X1: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points to be modelled.  Each row
            corresponds to a single data point.
        """
        # fitting uniform data sample vs observed data
        # set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # make training sample and labels
        # use test sample below for probability calibration.
        X1_train, X1_test, y1_train, y1_test = train_test_split(
            X1, np.ones(X1.shape[0]), test_size=self.test_size, random_state=self.random_state
        )
        X0_train = (
            X0[: X1_train.shape[0]]
            if X0 is not None
            else self._sample_bayesian_network(size=X1_train.shape[0], show_progress=False, seed=self.random_state)[
                :, self.nonlinear_indices_
            ]
        )

        X_train = np.concatenate([X0_train, X1_train], axis=0)
        y_train = np.concatenate([np.zeros(X1_train.shape[0]), y1_train], axis=None)

        self.clf = self.clf.fit(X_train, y_train, **self.clffitkw)
        # self.train_data = (X_train, y_train)
        # self.test_data = (X1_test, y1_test)

        if self.apply_calibration:
            # Calibrate probabilities manually. (Used for weights calculation.)
            self.logger.info("Calibrating classifier")
            X0_test = (
                X0[X1_train.shape[0] :]
                if X0 is not None
                else self._sample_bayesian_network(size=max(250000, X1_test.shape[0]), show_progress=True)[
                    :, self.nonlinear_indices_
                ]
            )

            p0 = self.clf.predict_proba(X0_test)[:, 1]
            p1 = self.clf.predict_proba(X1_test)[:, 1]

            # next: evaluation of calibration and sample weights
            if not isinstance(bins, int):
                bins = min(
                    min(len(np.histogram_bin_edges(p0, bins=bins)) - 1, len(np.histogram_bin_edges(p1, bins=bins)) - 1),
                    100,
                )
                self.logger.info(f"automatically set n_calibration_bins = {bins}")
            else:
                self.logger.info(f"n_calibration_bins = {bins}")

            hist_p0, bin_edges = np.histogram(p0, bins=bins, range=(0, 1))
            hist_p1, bin_edges = np.histogram(p1, bins=bins, range=(0, 1))
            self._calibrate_classifier(hist_p0, hist_p1, bin_edges)

            # storing these for validation later on
            self.hist_p0_ = hist_p0
            self.hist_p1_ = hist_p1
            self.bin_edges_ = bin_edges
            self.bin_centers_ = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    def _calibrate_classifier(self, hist_p0, hist_p1, bin_edges, use_sample_weights=True, validation_plots=False):
        """Calibrate classifier based on probability histograms

        :param hist_p0:
        :param hist_p1:
        :param bin_edges:
        :return:
        """
        hist_p0 = hist_p0.astype(float)
        hist_p1 = hist_p1.astype(float)
        rest_p0 = np.sum(hist_p0) - hist_p0
        rest_p1 = np.sum(hist_p1) - hist_p1
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_width = bin_edges[1] - bin_edges[0]

        if validation_plots:
            plt.figure(figsize=(12, 7))
            plt.bar(bin_centers, hist_p0 / np.sum(hist_p0), width=bin_width, alpha=0.5, label="p0", log=True)
            plt.bar(bin_centers, hist_p1 / np.sum(hist_p1), width=bin_width, alpha=0.5, label="p1", log=True)
            plt.legend()
            plt.show()

        def poisson_uncertainty(n):
            # return correct poisson counts (set to one for zero counts)
            sigma_n = np.sqrt(n)
            sigma_n[sigma_n == 0] = 1.0
            return sigma_n

        sigma_bin0 = poisson_uncertainty(hist_p0)
        sigma_rest0 = poisson_uncertainty(rest_p0)
        sigma_bin1 = poisson_uncertainty(hist_p1)
        sigma_rest1 = poisson_uncertainty(rest_p1)

        def fraction_and_uncertainty(a, b, sigma_a, sigma_b):
            # return fraction a/(a+b) and uncertainty on it, given uncertainties on a and b
            sum_ab = a + b
            frac_a = np.divide(a, sum_ab, out=np.zeros_like(a), where=sum_ab != 0)
            frac_b = np.divide(b, sum_ab, out=np.zeros_like(b), where=sum_ab != 0)
            sigma_p1 = np.divide(frac_b * sigma_a, sum_ab, out=np.zeros_like(frac_b), where=sum_ab != 0)
            sigma_p2 = np.divide(frac_a * sigma_b, sum_ab, out=np.zeros_like(frac_a), where=sum_ab != 0)
            sigma_fa2 = np.power(sigma_p1, 2) + np.power(sigma_p2, 2)
            return frac_a, np.sqrt(sigma_fa2)

        frac0, sigma_frac0 = fraction_and_uncertainty(hist_p0, rest_p0, sigma_bin0, sigma_rest0)
        frac1, sigma_frac1 = fraction_and_uncertainty(hist_p1, rest_p1, sigma_bin1, sigma_rest1)
        p1cb, sigma_p1cb = fraction_and_uncertainty(frac1, frac0, sigma_frac1, sigma_frac0)

        # sample weight is set to zero in case both sigma_p1cb is zero
        sample_weight = np.divide(1.0, sigma_p1cb * sigma_p1cb, out=np.zeros_like(sigma_p1cb), where=sigma_p1cb != 0)
        sample_weight /= np.min(sample_weight[sample_weight > 0])
        sample_weight = sample_weight if use_sample_weights else None

        # make sure last entry is filled, from which max_weight is derived
        if p1cb[-1] == 0:
            filled = p1cb[(p1cb > 0) & (p1cb < 1)]
            if len(filled) > 0:
                p1cb[-1] = np.max(filled)
                if use_sample_weights:
                    sample_weight[-1] = 1e-3

        # use isotonic regression to smooth out potential fluctuations in the p1 values
        # isotonic regression assumes that p1 can only be a rising function.
        # (we're assuming that if a classifier predicts a higher probability, the calibrated probability
        # will also be higher. This will generally be a safe assumption.)
        iso_reg = IsotonicRegression(y_min=0, y_max=1, increasing=self.isotonic_increasing).fit(
            bin_centers, p1cb, sample_weight
        )
        # iso_reg = IsotonicRegression(y_min=0, y_max=1).fit(bin_centers, p1cb)
        p1pred = iso_reg.predict(bin_centers)
        self.p1f_ = interpolate.interp1d(
            bin_edges[:-1], p1pred, kind="previous", bounds_error=False, fill_value="extrapolate"
        )

        max_p1f = np.max(p1pred)
        # assert self.p1f_(bin_centers[-1]) == max_p1f
        self.max_weight_ = max_p1f / (1.0 - max_p1f) if max_p1f != 1 else self.max_scale_value
        self.logger.info(f"Maximum weight: {self.max_weight_}")

        if validation_plots:
            plt.figure(figsize=(12, 7))
            plt.plot(bin_centers, p1cb, label="p1cb")
            plt.plot(bin_centers, p1pred, label="p1pred")
            plt.plot(bin_centers, bin_centers, label="bin_centers")
            plt.legend()
            plt.show()

    def _transform_and_slice(self, X, discretize=False):
        """.

        :param X:
        :return:
        """
        X_cat = X[:, self.categorical_columns]
        X_num = X[:, self.numerical_columns]
        if X_num.shape[1] > 0:
            X_num = check_array(X_num, copy=self.copy, dtype=FLOAT_DTYPES, force_all_finite="allow-nan")

        U = self.pipe_.transform(X_num) if X_num.shape[1] > 0 else np.empty_like(X_num)
        if discretize:
            bin_width = 1.0 / self.n_uniform_bins
            U = np.floor(U / bin_width)
            U[U >= self.n_uniform_bins] = self.n_uniform_bins - 1  # correct for values at 1.

        X_trans = self._join_and_reorder(X_cat, U, self.categorical_columns, self.numerical_columns)
        return X_trans[:, self.nonlinear_indices_] if self.n_vars_ >= 2 else X_trans

    def clf_predict_proba(self, X):
        """.

        :param X:
        :return:
        """
        X_slice = self._transform_and_slice(X)
        return self.clf.predict_proba(X_slice)

    def _scale(self, X):
        """Determine density of the Copula space for input data points

        :param X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :return: array of Copula densities for all data points.
        """
        # trivial case
        n_samples = X.shape[0]
        if self.n_vars_ <= 1 or self.force_uncorrelated or self.clf is None:
            return np.ones(n_samples)

        prob = self.clf.predict_proba(X)[:, 1]
        nominator = self.p1f_(prob) if self.apply_calibration else prob
        denominator = 1.0 - nominator
        # calculate ratio. In case denominator is zero, return 1 as ratio.
        ratio = np.divide(nominator, denominator, out=np.ones_like(nominator), where=denominator != 0)
        return np.array([r if r < self.max_scale_value else self.max_scale_value for r in ratio])

    def _configure_nonlinear_variables(self, X_discrete, X_expect=None):
        """Determine the variables to be modelled non-linearly

        :param X_uniform: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """
        # determine number of variables and bins to use for classifier.
        if self.ordering == "mi":
            self.logger.info(f"Selecting non-linear features: {self.ordering}")
            self.nonlinear_indices_, self.residual_indices_ = self._configure_vars_mi(X_discrete, X_expect)
        elif self.ordering == "phik":
            self.logger.info(f"Selecting non-linear features: {self.ordering}")
            self.nonlinear_indices_, self.residual_indices_ = self._configure_vars_phik(X_discrete, X_expect)
        else:
            self.nonlinear_indices_ = list(range(X_discrete.shape[1]))
            self.residual_indices_ = []

        self.n_vars_ = len(self.nonlinear_indices_)
        self.n_resid_vars_ = len(self.residual_indices_)

    def _configure_vars_pca(self, X, X_expect=None):
        """Determine the variables to be modelled non-linearly based on pca variance

        :param X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :return: tuple of selected indices of non-linear variables and residual variables
        """
        # determine number of features and bins to use for KBinsDiscretizer below.
        # if no pca, use all features.
        _, n_features = X.shape
        n_vars = n_features

        # obvious cases
        if self.force_uncorrelated:
            nonlinear_indices = []
            residual_indices = list(range(n_features))
            return nonlinear_indices, residual_indices
        if n_features == 1:
            nonlinear_indices = []
            residual_indices = [0]
            return nonlinear_indices, residual_indices

        # if pca, pick up top-most important variables.
        if self.do_PCA and not self.force_uncorrelated and self.n_nonlinear_vars is None:
            # use pca.explained_variance_ratio_ to determine minimal number to explain the threshold set.
            n_vars = n_features
            pca = self.pipe_[1]
            if self.min_pca_variance < 1:
                pca_n_vars = n_features
                for i in range(n_features):
                    pca_explained_variance = np.sum(pca.explained_variance_ratio_[:i])
                    if pca_explained_variance >= self.min_pca_variance:
                        pca_n_vars = i
                        break
                # pick minimum of 1 and 2
                if pca_n_vars < n_vars:
                    n_vars = pca_n_vars

        # override number of vars?
        if self.do_PCA and not self.force_uncorrelated and self.n_nonlinear_vars is not None:
            n_vars = min(n_features, int(self.n_nonlinear_vars))

        nonlinear_indices = list(range(n_vars))
        residual_indices = list(range(n_vars, n_features))
        return nonlinear_indices, residual_indices

    def _configure_vars_mi(self, X_discrete, X_expect=None):
        """Determine the variables to be modelled non-linearly based on mutual-information

        :param X_discrete: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :return: tuple of selected indices of non-linear variables and residual variables
        """
        n_features = X_discrete.shape[1]

        # obvious cases
        if self.force_uncorrelated:
            nonlinear_indices = []
            residual_indices = list(range(n_features))
            return nonlinear_indices, residual_indices
        if n_features == 1:
            nonlinear_indices = []
            residual_indices = [0]
            return nonlinear_indices, residual_indices

        # use mutual information to capture residual levels of non-linearity
        mi = np.zeros((n_features, n_features))
        for i in tqdm(range(n_features), description="mutual info regression", total=n_features):
            mi[i, :] = mutual_info_regression(X_discrete, X_discrete[:, i], random_state=self.random_state + i)

        # select all off-diagonal mutual information values and index pairs
        mis = []
        indices = []
        for i, j in itertools.combinations(range(n_features), 2):
            if mi[i, j] > self.min_mutual_information or self.n_nonlinear_vars is not None:
                mis.append(mi[i, j])
                indices.append((i, j))
        mis = np.array(mis)
        indices = np.array(indices)

        # sort those from low to high MI values
        arr1inds = mis.argsort()
        indices = indices[arr1inds]
        # then sort indices (i,j) on if they also occur in next-max MI value.
        # e.g. if j occurs in next-max MI values: (i,j) -> (j,i)
        for i in range(len(indices) - 1):
            for j in indices[i]:
                if j in indices[i + 1]:
                    indices[i + 1] = (
                        (indices[i + 1][0], indices[i + 1][1])
                        if j == indices[i + 1][0]
                        else (indices[i + 1][1], indices[i + 1][0])
                    )

        # now sort from high to low MI values
        # then extract variable indices associated with highest->lowest MI values
        indices = indices[::-1]
        nonlinear_indices = []
        for ij in indices:
            for j in ij:
                if j not in nonlinear_indices:
                    nonlinear_indices.append(j)

        # truncate number of variables if so desired
        if self.n_nonlinear_vars is not None:
            nonlinear_indices = nonlinear_indices[: self.n_nonlinear_vars]

        # all remaining indices are considered residual indices;
        # needed for sampling later on to simulate complete dataset
        all_indices = set(range(n_features))
        residual_indices = sorted(all_indices.difference(nonlinear_indices))

        return nonlinear_indices, residual_indices

    def _configure_vars_phik(self, X_discrete, X_expect):
        """Determine the variables to be modelled non-linearly based on phik correlation

        :param X_discrete: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :return: tuple of selected indices of non-linear variables and residual variables
        """
        n_features = X_discrete.shape[1]

        # obvious cases
        if self.force_uncorrelated:
            nonlinear_indices = []
            residual_indices = list(range(n_features))
            return nonlinear_indices, residual_indices
        if n_features == 1:
            nonlinear_indices = []
            residual_indices = [0]
            return nonlinear_indices, residual_indices

        # use phik to capture residual levels of non-linearity
        self.phik_matrix_ = phik_matrix(X_discrete, X_expect)
        phik = self.phik_matrix_.values

        # select all off-diagonal mutual information values and index pairs
        phiks = []
        indices = []
        for i, j in itertools.combinations(range(n_features), 2):
            if phik[i, j] > self.min_phik_correlation or self.n_nonlinear_vars is not None:
                phiks.append(phik[i, j])
                indices.append((i, j))
        phiks = np.array(phiks)
        indices = np.array(indices)

        # sort the phik values from low to high
        arr1inds = phiks.argsort()
        indices = indices[arr1inds]
        # then sort indices (i,j) on if they also occur in next-max phik value.
        # e.g. if j occurs in next-max phik values: (i,j) -> (j,i)
        for i in range(len(indices) - 1):
            for j in indices[i]:
                if j in indices[i + 1]:
                    indices[i + 1] = (
                        (indices[i + 1][0], indices[i + 1][1])
                        if j == indices[i + 1][0]
                        else (indices[i + 1][1], indices[i + 1][0])
                    )

        # now sort from high to low phik values
        # then extract variable indices associated with highest->lowest phik values
        indices = indices[::-1]
        nonlinear_indices = []
        for ij in indices:
            for j in ij:
                if j not in nonlinear_indices:
                    nonlinear_indices.append(j)

        # truncate number of variables if so desired
        if self.n_nonlinear_vars is not None:
            nonlinear_indices = nonlinear_indices[: self.n_nonlinear_vars]

        # all remaining indices are considered residual indices;
        # needed for sampling later on to simulate complete dataset
        all_indices = set(range(n_features))
        residual_indices = sorted(all_indices.difference(nonlinear_indices))

        return nonlinear_indices, residual_indices

    def logpdf(self, X, add_cat=True, add_num=True, add_nonlin=True):
        """Evaluate the logarithmic probability of each point in a data set.

        :param X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :return: ndarray, shape (n_samples,)
            The array of probability density evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        X_cat = X[:, self.categorical_columns]
        X_num = X[:, self.numerical_columns]
        # sample profiles
        n_features = X_num.shape[1]
        if n_features > 0:
            X_num = check_array(X_num, copy=self.copy, dtype=FLOAT_DTYPES, force_all_finite="allow-nan")

        # perform probability calculation

        # 0. start off with scale in uniform space. This captures the non-linearity in numerical variables.
        U = self.pipe_.transform(X_num) if n_features > 0 else np.empty_like(X_num)

        # 1. categorical variables + discretized numerical variables
        #    need discretized continuous variables as input to the bayesian network
        bin_width = 1.0 / self.n_uniform_bins
        U_discrete = np.floor(U / bin_width)
        U_discrete[U_discrete >= self.n_uniform_bins] = self.n_uniform_bins - 1  # correct for values at 1.
        X_discrete = self._join_and_reorder(X_cat, U_discrete, self.categorical_columns, self.numerical_columns)
        log_p_cat = self.bn_prob.log_probability(X_discrete, ordering=self.bn_ordering)

        # 2a. add numerical variables probability density:
        #    multiply with the inverse jacobian of the first kde transformation
        log_p_num = np.zeros_like(log_p_cat)
        if n_features > 0:
            # for numerical variables, use probability density calculation.
            if add_cat:
                # first turn bn probability contribution of numerical variables into a density correction.
                # (nominal value = 1.)
                log_p_num += np.log(self.n_uniform_bins) * n_features
            jac = self.pipe_[0].jacobian(X_num)
            log_p_num -= np.log(jac)
            if self.do_PCA and n_features >= 2 and not self.force_uncorrelated:
                # 2b. multiply with the (inverse) jacobian of the second kde transformation
                #     note that pca is a rotation and does not affect the probability density,
                #     ie. jacobian(pca)==1 by construction
                log_p_num += np.log(self.pipe_[2].inverse_jacobian(U))

        # 3. non-linear correction: take the slice of selected top-n non-linear variables
        X_trans = self._join_and_reorder(X_cat, U, self.categorical_columns, self.numerical_columns)
        X_slice = X_trans[:, self.nonlinear_indices_] if self.n_vars_ >= 2 else X_trans
        log_p_nonlin = np.log(self._scale(X_slice))

        log_p = np.zeros_like(log_p_cat)
        if add_cat:
            log_p += log_p_cat
        if add_num:
            log_p += log_p_num
        if add_nonlin:
            log_p += log_p_nonlin
        # log_p = log_p_cat + log_p_num + log_p_nonlin
        return log_p

    def pdf(self, X):
        """Evaluate the probability of each point in a data set.

        :param X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :return: ndarray, shape (n_samples,)
            The array of log(density) evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        p = np.exp(self.logpdf(X))
        # require minimum pdf value, so log(p) is always defined.
        return np.array([pi if pi > 0 else self.min_pdf_value for pi in p])

    def score_samples(self, X):
        """Evaluate the logarithmic probability of each point in a data set.

        :param X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :return: ndarray, shape (n_samples,)
            The array of log(density) evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        return self.logpdf(X)

    def score(self, X, y=None):
        """Compute the total log probability density under the model.

        :param X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :param y: None
            Ignored. This parameter exists only for compatibility with
            :class:`sklearn.pipeline.Pipeline`.
        :return: float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        return np.sum(self.score_samples(X))

    def sample(self, n_samples=1, random_state=None, show_progress=False, inverse_transform=True):
        """Generate random samples from the model.

        :param n_samples: int, optional
            Number of samples to generate. Defaults to 1.
        :param random_state: int, RandomState instance, default=None
            Determines random number generation used to generate
            random samples. Pass an int for reproducible results
            across multiple function calls. Ignored for now.
            See :term: `Glossary <random_state>`.
        :return: array_like, shape (n_samples, n_features)
            List of samples, sample weights
        """
        X, sample_weights = self._sample_no_transform(n_samples, random_state, show_progress=show_progress)
        n_features = len(self.numerical_columns)
        X_cat = X[:, self.categorical_columns]
        X_num = X[:, self.numerical_columns]
        if inverse_transform:
            X_num = self.pipe_.inverse_transform(X_num) if n_features > 0 else X_num
        return self._join_and_reorder(X_cat, X_num, self.categorical_columns, self.numerical_columns), sample_weights

    def _sample_no_transform(self, n_samples=1, random_state=None, show_progress=False):
        """Generate uniform random samples from the model.

        No inverse feature transformations are applied.

        :param n_samples: int, optional
            Number of samples to generate. Defaults to 1.
        :param random_state: int, RandomState instance, default=None
            Determines random number generation used to generate
            random samples. Pass an int for reproducible results
            across multiple function calls. Ignored for now.
            See :term: `Glossary <random_state>`.
        :return: array_like, shape (n_samples, n_features)
            List of samples, sample weights
        """
        # trivial checks
        if not (self.n_vars_ > 0 or self.n_resid_vars_ > 0):
            msg = "pdf not configured for sample generation."
            raise RuntimeError(msg)

        # set random state
        if random_state is not None and isinstance(random_state, int):
            np.random.seed(random_state)

        # generate nonlinear variables with accept-reject method
        data = self._sample_bayesian_network(size=n_samples, show_progress=show_progress, seed=random_state)
        sample_weights = self._scale(data[:, self.nonlinear_indices_])

        return data, sample_weights

    def sample_no_weights(
        self, n_samples=1, random_state=None, mode="expensive", inverse_transform=True, cheap_factor=1
    ):
        """Generate random samples from the model.

        Use accept-reject to get rid of sample weights.
        The number of returned data points will be less than n_samples.

        :param n_samples: int, optional
            Number of samples to generate. Defaults to 1.
        :param random_state: int, RandomState instance, default=None
            Determines random number generation used to generate
            random samples. Pass an int for reproducible results
            across multiple function calls. Ignored for now.
            See :term: `Glossary <random_state>`.
        :param mode: str, 'expensive' samples 10x as many weighted
            samples as requested and then runs accept-reject.
            'cheap' generates minimal sample and drops or duplicates
            entries to unweight them.
        :return: array_like, shape (n_samples, n_features)
            List of samples.
        """
        if random_state is not None and isinstance(random_state, int):
            random.seed(random_state)

        if mode == "expensive":
            # all samples are unique

            # estimate the acceptance ratio to estimate n_tries required to get desired sample size
            tempdata, tempsample_weights = self._sample_no_transform(1000, random_state, show_progress=False)

            # apply accept-reject method
            tempweight_max = np.max(tempsample_weights)
            tempu = np.random.uniform(0, tempweight_max, size=1000)
            tempkeep = tempu < tempsample_weights
            tempdata = tempdata[tempkeep]

            accratio = tempdata.shape[0] / 1000
            n_tries = int(np.ceil(1 / accratio * n_samples)) + 1000

            # n_tries = 10*n_samples
            data, sample_weights = self._sample_no_transform(n_tries, random_state, show_progress=self.verbose)

            # apply accept-reject method
            weight_max = np.max(sample_weights)
            u = np.random.uniform(0, weight_max, size=n_tries)
            keep = u < sample_weights
            data = data[keep]

            if data.shape[0] > n_samples:
                data = data[:n_samples]

        elif mode == "cheap":
            # generated samples are dropped or duplicated to match weights
            data, sample_weights = self._sample_no_transform(
                n_samples * cheap_factor, random_state, show_progress=self.verbose
            )
            pop = np.asarray(range(data.shape[0]))
            probs = sample_weights / np.sum(sample_weights)
            sample = random.choices(pop, probs, k=n_samples)
            data = data[sample]
        elif mode == "ignore_weights":
            data, _ = self._sample_no_transform(n_samples, random_state, show_progress=self.verbose)
        else:
            msg = "mode should be 'expensive', 'cheap' or 'ignore_weights'"
            raise ValueError(msg)

        if not inverse_transform:
            return data

        # transform back all numerical columns
        n_features = len(self.numerical_columns)
        X_cat = data[:, self.categorical_columns]
        X_num = data[:, self.numerical_columns]
        X_num = self.pipe_.inverse_transform(X_num) if n_features > 0 else X_num
        return self._join_and_reorder(X_cat, X_num, self.categorical_columns, self.numerical_columns)
