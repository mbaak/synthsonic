import numpy as np
from synthsonic.models.kde_utils import kde_process_data, kde_bw, kde_make_transformers, kde_smooth_peaks_1dim
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES # check_is_fitted, _deprecate_positional_args
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import norm
from scipy.special import erf
from scipy import interpolate
import warnings


class KDEQuantileTransformer(TransformerMixin, BaseEstimator):
    """ Quantile tranformer class using for each variable the CDF obtained with kernel density estimation
    """
    def __init__(self,
                 n_quantiles=1000,
                 output_distribution='uniform',
                 smooth_peaks=True,
                 mirror_left=None,
                 mirror_right=None,
                 rho=0.5,
                 n_adaptive=1,
                 x_min=None,
                 x_max=None,
                 n_integral_bins=1000,
                 use_KDE=True,
                 use_inverse_qt=False,
                 random_state=0,
                 copy=True):
        """ Parameters with the class KDEQuantileTransformer

        KDEQuantileTransformer is a quantile tranformer class using for each variable the CDF obtained with
        kernel density estimation. Besides normal transformation functions, the class also provides the jacobian
        and inverse jacobian of the transformation and inverse transformation respectively.

        The KDE quantile transformation happens in four steps, two of which are transformations:

        1. First KDE PDFs and CDFs are formed for all marginalized input variables.
        2. Using the (smooth) CDFs, all input variables are transformed to uniform distributions.
        3. Using the existing quantile transformer of sklearn, these uniform distributions are then transformed to
           normal distributions.
        4. The KDE PDFs are used to calculate the (inverse) jacobian of the transformation.

        Concerning KDE evaluation of the PDF and CDF, the adaptive bandwidths are evaluated with the eqns described in:
        Cranmer KS, Kernel Estimation in High-Energy Physics. Computer Physics Communications 136:198-207, 2001
        e-Print Archive: hep ex/0011057

        In theory both transformations could be combined into one, but there are practical advantages of using two.
        Essentially the second transformation is a backup against the first one, to smooth out residual bumps.
        For certain edge case distributions, for example those with strange discrete peaks in them at the edge
        of a distribution, it may happen that a single transformation fails, in which case doing two quantile
        transformations catches any potential imperfections in the first.
        In the inverse transformation, by default the two transformations are combined into one however, b/c else
        the impact of KDE smoothing is cancelled.

        :param int n_quantiles: number of quantiles/bins used in output histogram. If greater than number of samples,
            this is reset to number of samples. Default is 1000.
        :param str output_distribution: 'uniform' or 'normal' distribution.
        :param bool smooth_peaks: if False, do not smear peaks of non-unique values.
        :param mirror_left: array. Mirror the data on a value on the left to counter signal leakage.
            Default is None, which is no mirroring.
        :param mirror_right: array. Mirror the data on a value on the right to counter signal leakage.
            Default is None, which is no mirroring.
        :param float rho: KDE bandwidth scale parameter. default is 0.5.
        :param int n_adaptive: KDE number of adaptive iterations to be applied to improve the band width. default is 1.
        :param x_min: array. minimum value of pdf's x range. default is None (= - inf)
        :param x_max: array. maximum value of pdf's x range. default is None (= + inf)
        :param int n_integral_bins: for internal evaluation, number of integration bins beyond x-range. default is 1000.
        :param bool use_KDE: Default is True. If false, KDE smoothing is off, using default quantile transformation.
        :param bool use_inverse_qt: Default is False. If true, KDE is not used in inverse transformation.
        :param int random_state: when an integer, the seed given random generator.
        :param copy: Copy the data before transforming. Default is True.
        """
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.smooth_peaks = smooth_peaks
        self.n_adaptive = n_adaptive
        self.copy = copy
        self.use_inverse_qt = use_inverse_qt
        self.use_KDE = use_KDE
        self.n_integral_bins = max(n_integral_bins, 1000)
        self.random_state = random_state

        # integration range
        self.x_min = np.array(x_min) if isinstance(x_min, (list, tuple, np.ndarray)) else None
        self.x_max = np.array(x_max) if isinstance(x_max, (list, tuple, np.ndarray)) else None

        # left and right-hand mirror points
        self.mirror_left = np.array(mirror_left) if isinstance(mirror_left, (list, tuple, np.ndarray)) else None
        self.mirror_right = np.array(mirror_right) if isinstance(mirror_right, (list, tuple, np.ndarray)) else None

        # copy x ranges if mirror points not set
        self.mirror_left = self.x_min if self.mirror_left is None else self.mirror_left
        self.mirror_right = self.x_max if self.mirror_right is None else self.mirror_right

        # bandwidth rescaling factor
        self.rho = np.array(rho) if isinstance(rho, (list, tuple, np.ndarray)) else rho

        # basic checks on attributes
        if self.n_quantiles <= 0:
            raise ValueError("Invalid value for 'n_quantiles': %d. The number of quantiles must be at least one."
                             % self.n_quantiles)
        if self.output_distribution not in ('normal', 'uniform'):
            raise ValueError("'output_distribution' has to be either 'normal' or 'uniform'. Got '{}' instead."
                             % self.output_distribution)
        if (isinstance(self.rho, np.ndarray) and any([r <= 0 for r in self.rho])) or \
                (isinstance(self.rho, (float, np.number)) and self.rho <= 0):
            raise ValueError("Invalid value(s) for 'rho': %f. The number(s) must be greater than zero." % self.rho)
        if self.n_adaptive < 0:
            raise ValueError("Invalid value for 'n_adaptive': %d. Must be positive." % self.n_adaptive)

    def fit(self, X, y=None):
        """Compute the kde-based quantiles used for transforming.

        :param X: ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis.
        :param y: Ignored
        :return: self : object
        """
        X = check_array(X, copy=False, dtype=FLOAT_DTYPES, force_all_finite="allow-nan")

        # sample profiles
        n_samples, n_features = X.shape

        # continuation of basic checks, now that we know X
        if isinstance(self.rho, np.ndarray):
            if self.rho.shape[0] != n_features:
                raise ValueError("Invalid size of 'rho': %d. The number should match the data: %d."
                                 % (self.rho.shape[0], n_features))
        else:
            self.rho = np.array([self.rho] * n_features)
        if isinstance(self.mirror_left, np.ndarray):
            if self.mirror_left.shape[0] != n_features:
                raise ValueError("Invalid size of 'mirror_left': %d. The number should match the data: %d."
                                 % (self.mirror_left.shape[0], n_features))
        else:
            self.mirror_left = np.array([None] * n_features)
        if isinstance(self.mirror_right, np.ndarray):
            if self.mirror_right.shape[0] != n_features:
                raise ValueError("Invalid size of 'mirror_right': %d. The number should match the data: %d."
                                 % (self.mirror_right.shape[0], n_features))
        else:
            self.mirror_right = np.array([None] * n_features)
        if isinstance(self.x_min, np.ndarray):
            if self.x_min.shape[0] != n_features:
                raise ValueError("Invalid size of 'x_min': %d. The number should match the data: %d."
                                 % (self.x_min.shape[0], n_features))
        else:
            self.x_min = np.array([None] * n_features)
        if isinstance(self.x_max, np.ndarray):
            if self.x_max.shape[0] != n_features:
                raise ValueError("Invalid size of 'x_max': %d. The number should match the data: %d."
                                 % (self.x_max.shape[0], n_features))
        else:
            self.x_max = np.array([None] * n_features)

        # number of quantiles cannot be higher than number of data points. If so, reset.
        if self.n_quantiles > n_samples:
            warnings.warn("n_quantiles (%s) is greater than the total number "
                          "of samples (%s). n_quantiles is set to "
                          "n_samples."
                          % (self.n_quantiles, n_samples))
        self.n_quantiles = max(1, min(self.n_quantiles, n_samples))

        # set the (x_min, x_max) transformation range
        # if not set, by default widen the range beyond min/max to account for signal leakage
        if any([x is None for x in self.x_min]) or any([x is None for x in self.x_max]):
            gstd = np.std(X, axis=0)
            bw = np.power(4 / 3, 0.2) * gstd * np.power(n_samples, -0.2)
            min_orig = np.min(X, axis=0) - 10 * bw
            max_orig = np.max(X, axis=0) + 10 * bw
            for i in range(n_features):
                self.x_min[i] = min_orig[i] if (self.x_min[i] is None and gstd[i] > 0) else self.x_min[i]
                self.x_max[i] = max_orig[i] if (self.x_max[i] is None and gstd[i] > 0) else self.x_max[i]

        if self.use_KDE:
            # Do the actual KDE fit (to uniform distributions)
            self._kde_fit(X)
            # prepare X to do quantile transformer fit.
            # add extreme points so QT knows the true edges for inverse transformation after sampling
            X = self._kde_transform(X)
            low = np.array([[0] * X.shape[1]])
            high = np.array([[1] * X.shape[1]])
            X = np.concatenate([X, low, high], axis=0)
        elif self.smooth_peaks:
            X = self._smooth_peaks(X)
            # create pdf for quantile transformation
            self._qt_pdf(X)

        # perform quantile transformation to smooth out any residual imperfections after kde
        # standard quantile transformer helps to smooth out any residual imperfections after kde transformation,
        # and does conversion to normal.
        self.qt_ = QuantileTransformer(
            n_quantiles=self.n_quantiles, 
            output_distribution=self.output_distribution,
            copy=self.copy,
            random_state=self.random_state,
        )
        self.qt_.fit(X)

        return self

    def _qt_pdf(self, X, min_pdf_value=1e-20):
        """Internal function to make quantile transformer pdf

        Is only run when use_KDE=False

        :param X: ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis.
        """
        self.pdf_ = []

        n_samples, n_features = X.shape
        ps = np.linspace(0, 1, self.n_quantiles + 1)

        # calculate quantiles and pdf
        for i in range(n_features):
            x = X[:, i]
            qs = np.quantile(x, ps)
            bin_entries, bin_edges = np.histogram(x, bins=qs)
            bin_entries = bin_entries.astype(float) / n_samples
            bin_diffs = np.diff(bin_edges)
            pdf_norm = np.divide(bin_entries, bin_diffs, out=np.zeros_like(bin_entries), where=bin_diffs != 0)
            fast_pdf = interpolate.interp1d(bin_edges[:-1], pdf_norm, kind='previous', bounds_error=False,
                                            fill_value=(min_pdf_value, min_pdf_value))
            self.pdf_.append({'fast': fast_pdf})

    def _kde_fit(self, X):
        """Internal function to compute the kde-based quantiles used for transforming.

        :param X: ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis.
        :return: self : object
        """
        # reset
        self.pdf_ = []
        self.cdf_ = []

        n_features = X.shape[1]

        for i in range(n_features):
            # do kde fit, store each pdf
            bin_entries, bin_mean = kde_process_data(X[:, i], self.n_quantiles, self.smooth_peaks,
                                                     self.mirror_left[i], self.mirror_right[i],
                                                     random_state=self.random_state)
            band_width = kde_bw(bin_mean, bin_entries, self.rho[i], self.n_adaptive)
            # transformers to uniform distribution and back
            fast_pdf, F, Finv, kde_norm = kde_make_transformers(bin_mean, bin_entries, band_width,
                                                                x_min=self.x_min[i], x_max=self.x_max[i],
                                                                n_bins=self.n_integral_bins)
            # store cdf, inverse-cdf, and pdf.
            self.cdf_.append((F, Finv))
            pdf = {'bin_entries': bin_entries, 'bin_mean': bin_mean, 'band_width': band_width,
                   'norm': kde_norm, 'fast': fast_pdf}
            self.pdf_.append(pdf)

        return self

    def _smooth_peaks(self, X):
        """Internal function to smooth non-unique peaks

        :param X: ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis.
        :return: ndarray or sparse matrix, shape (n_samples, n_features)
            The transformed data
        """
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, force_all_finite="allow-nan")

        n_features = X.shape[1]
        for feature_idx in range(n_features):
            x = X[:, feature_idx]
            # smooth peaks - note: this adds a random component to the data
            # applying smoothing to data that's already been smoothed has no impact, b/c all peaks are already gone.
            x = kde_smooth_peaks_1dim(x, self.mirror_left[feature_idx], self.mirror_right[feature_idx],
                                      copy=False, random_state=self.random_state, smoothing_width=1e-5)
            X[:, feature_idx] = x
        return X

    def _kde_transform(self, X):
        """Internal function to transform the data

        :param X: ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis.
        :return: ndarray or sparse matrix, shape (n_samples, n_features)
            The transformed data
        """
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, force_all_finite="allow-nan")

        n_features = X.shape[1]
        for feature_idx in range(n_features):
            x = X[:, feature_idx]
            # smooth peaks - note: this adds a random component to the data
            # applying smoothing to data that's already been smoothed has no impact, b/c all peaks are already gone.
            if self.smooth_peaks:
                x = kde_smooth_peaks_1dim(x, self.mirror_left[feature_idx], self.mirror_right[feature_idx],
                                          copy=False, random_state=self.random_state)
            # transform distribution to uniform
            y = self.cdf_[feature_idx][0](x)
            # transform uniform [0,1] distribution to normal
            # X[:, feature_idx] = np.sqrt(2.) * erfinv(2. * y - 1.) if self.output_distribution == 'normal' else y
            X[:, feature_idx] = y

        return X

    def transform(self, X):
        """Transform the data

        :param X: ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis.
        :return: ndarray or sparse matrix, shape (n_samples, n_features)
            The transformed data
        """
        # 1. kde transformation to uniform.
        if self.use_KDE:
            X = self._kde_transform(X)
        elif self.smooth_peaks:
            X = self._smooth_peaks(X)

        # 2. quantile transformation to smooth out residual bumps and do conversion to normal distribution
        return self.qt_.transform(X)

    def _kde_inverse_transform(self, X):
        """Internal function to inverse transform the data

        :param X: ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to inverse scale along the features axis.
        :return: ndarray or sparse matrix, shape (n_samples, n_features)
            The inverse-transformed data
        """
        n_features = X.shape[1]
        for feature_idx in range(n_features):
            x = X[:, feature_idx]
            # transform normal back to uniform [0,1]
            if not self.use_inverse_qt:
                x = (0.5 + 0.5 * erf(x/np.sqrt(2.))) if self.output_distribution == 'normal' else x
            # transform uniform back to original distribution
            X[:, feature_idx] = self.cdf_[feature_idx][1](x)

        return X

    def inverse_transform(self, X):
        """Inverse transform the data

        :param X: ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to inverse scale along the features axis.
        :return: ndarray or sparse matrix, shape (n_samples, n_features)
            The inverse-transformed data
        """
        # 1. quantile transformation back to kde
        if self.use_inverse_qt or not self.use_KDE:
            X = self.qt_.inverse_transform(X)
        # 2. inverse kde transformation
        return self._kde_inverse_transform(X) if self.use_KDE else X

    def jacobian(self, X):
        """Provide the Jacobian of the transformation

        :param X: ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis.
        :return: ndarray or sparse matrix, shape (n_samples, )
            An array with the jacobian of each data point
        """
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, force_all_finite="allow-nan")

        # smoothing of peaks
        if self.smooth_peaks:
            X = self._smooth_peaks(X)

        jac = 1.0

        for idx in range(X.shape[1]):
            kdfi = self.pdf_[idx]['fast']
            jac /= kdfi(X[:, idx])

        if self.output_distribution == 'normal':
            X = self.transform(X)
            for idx in range(X.shape[1]):
                jac *= norm.pdf(X[:, idx])

        return jac

    def inverse_jacobian(self, X):
        """Provide the Jacobian of the inverse transformation

        :param X: ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to inverse scale along the features axis.
        :return: ndarray or sparse matrix, shape (n_samples, )
            An array with the jacobian of the inverse transformation of each input data point
        """
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, force_all_finite="allow-nan")

        inv_jac = 1.0

        if self.output_distribution == 'normal':
            for idx in range(X.shape[1]):
                inv_jac /= norm.pdf(X[:, idx])

        X = self.inverse_transform(X)

        for idx in range(X.shape[1]):
            kdfi = self.pdf_[idx]['fast']
            inv_jac *= kdfi(X[:, idx])

        return inv_jac
