import numpy as np
from synthsonic.models.kde_quantile_tranformer import KDEQuantileTransformer
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES # check_is_fitted, _deprecate_positional_args
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import itertools
import warnings
import inspect

from random import choices


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
    def __init__(self,
                 n_quantiles=500,
                 mirror_left=None,
                 mirror_right=None,
                 rho=0.5,
                 n_adaptive=1,
                 x_min=None,
                 x_max=None,
                 do_PCA=True,
                 ordering='pca',
                 min_pca_variance=0.99,
                 min_mutual_information=0,
                 n_nonlinear_vars=None,
                 force_uncorrelated=False,
                 clf=MLPClassifier,
                 random_state=0,
                 use_inverse_qt=False,
                 use_KDE=True,
                 copy=True,
                 **kwargs):
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
            top-n non-linear variables that are to be modelled. Used if 'ordering' is 'pca'.
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
        """
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
        self.n_nonlinear_vars = n_nonlinear_vars
        self.force_uncorrelated = force_uncorrelated
        self.copy = copy
        self.random_state = random_state
        self.use_inverse_qt = use_inverse_qt
        self.use_KDE = use_KDE
        self.min_pdf_value = 1e-20
        self.max_scale_value = 500

        # instantiate classifier - passing on random_state
        specs = inspect.getfullargspec(clf.__init__)
        if 'random_state' in specs.args or specs.varkw == 'kwargs':
            kwargs['random_state'] = self.random_state
        self.clf = clf(**kwargs)

        # basic checks on attributes
        if self.min_pca_variance < 0 or self.min_pca_variance > 1:
            raise ValueError("Invalid value for 'min_pca_variance': %f. Should be a float in (0,1]."
                             % self.min_pca_variance)
        if self.min_mutual_information < 0:
            raise ValueError("Invalid value for 'min_mutual_information': %f. Should be greater than zero."
                             % self.min_mutual_information)
        if self.n_nonlinear_vars is not None:
            if not isinstance(self.n_nonlinear_vars, (int, float, np.number)) or self.n_nonlinear_vars < 1:
                raise ValueError("Invalid value for 'n_nonlinear_vars': %d. Should be an int greater than zero."
                                 % self.n_nonlinear_vars)
        if self.ordering not in ['pca', 'mi']:
            raise ValueError("Non-linear variables should be ordered by 'pca' or mutual information 'mi'.")

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
        # here we go ...
        X = check_array(X, copy=False, dtype=FLOAT_DTYPES, force_all_finite="allow-nan")

        # sample profiles
        n_samples, n_features = X.shape

        if self.n_quantiles > n_samples:
            warnings.warn("n_quantiles (%s) is greater than the total number of samples (%s). "
                          "n_quantiles is set to num samples." % (self.n_quantiles, n_samples))
        self.n_quantiles = max(1, min(self.n_quantiles, n_samples))
        if self.n_nonlinear_vars is not None:
            if self.n_nonlinear_vars > n_features:
                warnings.warn("n_nonlinear_vars (%d) is greater than the total number of features (%d). "
                              "n_nonlinear_vars is set to num features." % (self.n_nonlinear_vars, n_features))
            self.n_nonlinear_vars = min(self.n_nonlinear_vars, n_features)
        if not self.do_PCA and self.ordering == 'pca':
            warnings.warn("Cannot order non-linear variables by pca (turned off). Switching to mutual information.")
            self.ordering = 'mi'

        if self.do_PCA and n_features > 1 and not self.force_uncorrelated:
            self.pipe_ = make_pipeline(KDEQuantileTransformer(n_quantiles=self.n_quantiles,
                                                              output_distribution='normal',
                                                              mirror_left=self.mirror_left,
                                                              mirror_right=self.mirror_right, rho=self.rho,
                                                              n_adaptive=self.n_adaptive, x_min=self.x_min,
                                                              x_max=self.x_max, copy=self.copy,
                                                              random_state=self.random_state,
                                                              use_inverse_qt=self.use_inverse_qt,
                                                              use_KDE=self.use_KDE),
                                       PCA(n_components=n_features, whiten=False, copy=self.copy),
                                       KDEQuantileTransformer(n_quantiles=self.n_quantiles,
                                                              output_distribution='uniform', rho=min(self.rho, 0.2),
                                                              n_adaptive=self.n_adaptive, use_inverse_qt=True,
                                                              copy=self.copy, random_state=self.random_state)
                                       )
        else:
            self.pipe_ = make_pipeline(KDEQuantileTransformer(n_quantiles=self.n_quantiles,
                                                              output_distribution='uniform',
                                                              mirror_left=self.mirror_left,
                                                              mirror_right=self.mirror_right, rho=self.rho,
                                                              n_adaptive=self.n_adaptive, x_min=self.x_min,
                                                              x_max=self.x_max, copy=self.copy,
                                                              random_state=self.random_state,
                                                              use_inverse_qt=self.use_inverse_qt,
                                                              use_KDE=self.use_KDE))
        print(f'Transforming variables.')
        X_uniform = self.pipe_.fit_transform(X)
        # determine number of features to use for nonlinear modelling
        self._configure_nonlinear_variables(X_uniform)

        if self.n_vars_ >= 2 and not self.force_uncorrelated and self.clf is not None:
            # note: residual vars are treated as uncorrelated
            X_slice = X_uniform[:, self.nonlinear_indices_]
            # this function captures in matrices the residual non-linearity after the transformations above.
            print(f'Fitting and calibrating classifier.')
            self._configure_classifier(X_slice)

        print(f'Model = rho: {self.rho}, number of selected non-linear variables: {self.n_vars_}')

        return self

    def _configure_classifier(self, X1):
        """ The neural network below captures the residual non-linearity after the transformations above.

        :param X1: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points to be modelled.  Each row
            corresponds to a single data point.
        """
        # fitting uniform data sample vs observed data

        # set random state
        np.random.seed(self.random_state)

        # make training sample
        X0 = np.random.uniform(size=X1.shape)
        X = np.concatenate([X0, X1], axis=0)

        # make training labels
        zeros = np.zeros(X1.shape[0])
        ones = np.ones(X1.shape[0])
        y = np.concatenate([zeros, ones], axis=None)

        self.clf = self.clf.fit(X, y)
        self.calibrated_ = CalibratedClassifierCV(self.clf, cv="prefit").fit(X, y)

    def _scale(self, X):
        """ Determine density of the Copula space for input data points

        :param X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :return: array of Copula densities for all data points.
        """
        # trivial case
        n_samples = X.shape[0]
        if self.n_vars_ <= 1 or self.force_uncorrelated or self.clf is None:
            return np.ones(n_samples)

        prob = self.calibrated_.predict_proba(X)
        nominator = prob[:, 1]
        denominator = prob[:, 0]
        # calculate ratio. In case denominator is zero, return 1 as ratio.
        ratio = np.divide(nominator, denominator, out=np.ones_like(nominator), where=denominator != 0)
        return np.array([r if r < self.max_scale_value else self.max_scale_value for r in ratio])

    def _configure_nonlinear_variables(self, X_uniform):
        """ Determine the variables to be modelled non-linearly

        :param X_uniform: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """
        # determine number of variables and bins to use for KBinsDiscretizer.
        self.nonlinear_indices_, self.residual_indices_ = self._configure_vars_pca(X_uniform) \
            if self.ordering == "pca" else self._configure_vars_mi(X_uniform)
        self.n_vars_ = len(self.nonlinear_indices_)
        self.n_resid_vars_ = len(self.residual_indices_)

    def _configure_vars_pca(self, X):
        """ Determine the variables to be modelled non-linearly based on pca variance

        :param X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :return: tuple of selected indices of non-linear variables and residual variables
        """
        # determine number of features and bins to use for KBinsDiscretizer below.
        # if no pca, use all features.
        n_samples, n_features = X.shape
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

    def _configure_vars_mi(self, X_uniform):
        """ Determine the variables to be modelled non-linearly based on mutual-information

        :param X_uniform: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :return: tuple of selected indices of non-linear variables and residual variables
        """
        n_features = X_uniform.shape[1]

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
        for i in range(n_features):
            mi[i, :] = mutual_info_regression(X_uniform, X_uniform[:, i])

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
                    indices[i + 1] = (indices[i + 1][0], indices[i + 1][1]) if j == indices[i + 1][0] \
                        else (indices[i + 1][1], indices[i + 1][0])

        # now sort from high to low MI values
        # then extract variable indices associated with highest->lowest MI values
        indices = indices[arr1inds[::-1]]
        nonlinear_indices = []
        for ij in indices:
            for j in ij:
                if j not in nonlinear_indices:
                    nonlinear_indices.append(j)

        # truncate number of variables if so desired
        if self.n_nonlinear_vars is not None:
            nonlinear_indices = nonlinear_indices[:self.n_nonlinear_vars]

        # all remaining indices are considered residual indices;
        # needed for sampling later on to simulate complete dataset
        all_indices = set(range(n_features))
        residual_indices = sorted(all_indices.difference(nonlinear_indices))

        return nonlinear_indices, residual_indices

    def pdf(self, X):
        """Evaluate the probability of each point in a data set.

        :param X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :return: ndarray, shape (n_samples,)
            The array of probability density evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, force_all_finite="allow-nan")

        n_features = X.shape[1]

        # perform probability calculation

        # 1. start off with scale in uniform space. This captures the non-linearity.
        U = self.pipe_.transform(X)
        # take the slice of selected top-n non-linear variables
        U_slice = U[:, self.nonlinear_indices_] if self.n_vars_ >= 2 else U
        p = self._scale(U_slice)

        # 2. multiply with the inverse jacobian of the first kde transformation
        p /= self.pipe_[0].jacobian(X)

        if self.do_PCA and n_features >= 2 and not self.force_uncorrelated:
            # 3. multiply with the (inverse) jacobian of the second kde transformation
            #    note that pca is a rotation and does not affect the probability, ie.
            #    jacobian(pca)==1 by construction
            p *= self.pipe_[2].inverse_jacobian(U)

        # require minimum pdf value, so log(p) is always defined.
        p = np.array([pi if pi > 0 else self.min_pdf_value for pi in p])

        return p

    def logpdf(self, X):
        """Evaluate the logarithmic probability of each point in a data set.

        :param X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        :return: ndarray, shape (n_samples,)
            The array of log(density) evaluations. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        return np.log(self.pdf(X))

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

    def sample(self, n_samples=1, random_state=None):
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
        data, sample_weights = self._sample_no_transform(n_samples, random_state)
        return self.pipe_.inverse_transform(data), sample_weights

    def _sample_no_transform(self, n_samples=1, random_state=None):
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
            raise RuntimeError('pdf not configured for sample generation.')

        data = None
        sample_weights = np.ones(n_samples)

        # set random state
        if random_state is not None and isinstance(random_state, int):
            np.random.seed(random_state)

        # generate nonlinear variables with accept-reject method
        if self.n_vars_ > 0:
            data = np.random.uniform(0, 1, size=(n_samples, self.n_vars_))
            sample_weights = self._scale(data)

        # residual variables are treated as uncorrelated
        if self.n_resid_vars_ > 0:
            resid = np.random.uniform(0, 1, size=(n_samples, self.n_resid_vars_))
            data = np.concatenate([data, resid], axis=1) if data is not None else resid

        # reorder non-linear and residual columns to original order
        if self.ordering == 'mi':
            current_order = self.nonlinear_indices_ + self.residual_indices_
            permutation = [current_order.index(i) for i in range(len(current_order))]
            reidx = np.empty_like(permutation)
            reidx[permutation] = np.arange(len(permutation))
            data[:] = data[:, reidx]  # in-place modification of data

        return data, sample_weights

    def sample_no_weights(self, n_samples=1, random_state=None, mode='expensive'):
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
        
        if mode=='expensive': #all samples are unique
            
            #estimate the acceptance ratio to estimate n_tries required to get 
            #   desired sample size
            tempdata, tempsample_weights = self._sample_no_transform(1000, random_state)
            # apply accept-reject method
            tempweight_max = np.max(tempsample_weights)
            tempu = np.random.uniform(0, tempweight_max, size=1000)
            tempkeep = tempu < tempsample_weights
            tempdata = tempdata[tempkeep]
            
            accratio = tempdata.shape[0]/1000
            n_tries = int(np.ceil(1/accratio*n_samples)) + 1000
            
            #n_tries = 10*n_samples
            data, sample_weights = self._sample_no_transform(n_tries, random_state)
            
            # apply accept-reject method
            weight_max = np.max(sample_weights)
            u = np.random.uniform(0, weight_max, size=n_tries)
            keep = u < sample_weights
            data = data[keep]
            
            if data.shape[0] > n_samples:
                data = data[:n_samples]

        elif mode=='cheap': #generated samples are dropped or duplicated to match weights
            data, sample_weights = self._sample_no_transform(n_samples, random_state)
            pop = np.asarray(range(data.shape[0]))
            probs = sample_weights/np.sum(sample_weights)
            sample = choices(pop, probs, k=n_samples)
            data = data[sample]

        return self.pipe_.inverse_transform(data)
