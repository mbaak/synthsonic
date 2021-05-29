import numpy as np

from scipy.stats import norm
from scipy import interpolate


""" Utility functions for Kernel density estimation of 1-dimensional data distributions

To get a KDE pdf the following steps are applied:
1. kde_process_data() : Turn marginalized data distributions into quantile histograms.
   These are used as input to calculate the KDE pdfs.
       Part of the kde_process_data() call is the call kde_smooth_peaks() :
       to smooth non-unique spikes in the input data.
       This is needed such that the cdf of the input data X has no sudden jumps, and can also be inverted.
2. kde_bw() : calculates the adaptive bandwidth of each of the quantile bins.
   These are also used as input to calculate the KDE pdfs.
3. kde_make_transformers() : this creates a pdf, cdf and inverse-cdf of each KDE distribution.
   These are put into interpolation functions for fast evaluation later.
"""


def weighted_mean(a, weights=None, axis=None, dtype=None, keepdims=False):
    """Compute the weighted mean along the specified axis.

    :param a: Array containing numbers whose mean is desired. If `a` is not an array, a conversion is attempted.
    :param weights: Array containing weights for the elements of `a`. If `weights` is not an
        array, a conversion is attempted.
    :param axis: Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array. Type is None or int or tuple of ints, optional.
    :param dtype: data type to use in computing the mean.
    :param bool keepdims: If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one.
    :return: np.ndarray
    """
    if weights is None:
        return np.mean(a, axis=axis, dtype=dtype, keepdims=keepdims)
    else:
        w = np.array(weights)
        nom = np.sum(w * np.array(a), axis=axis, dtype=dtype, keepdims=keepdims)
        denom = np.sum(w, axis=axis, dtype=dtype, keepdims=keepdims)
        return nom / denom


def weighted_std(a, weights=None, axis=None, dtype=None, ddof=0, keepdims=False):
    """Compute the weighted standard deviation along the specified axis.

    :param a: Array containing numbers whose standard deviation is desired. If `a` is not an
        array, a conversion is attempted.
    :param weights: Array containing weights for the elements of `a`. If `weights` is not an
        array, a conversion is attempted.
    :param axis: Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array. Type is None or int or tuple of ints, optional.
    :param dtype: data type to use in computing the mean.
    :param int ddof: Delta Degrees of Freedom.  The divisor used in calculations
        is ``W - ddof``, where ``W`` is the sum of weights (or number of elements
        if `weights` is None). By default `ddof` is zero
    :param bool keepdims: If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one.
    :return: np.ndarray
    """
    if weights is None:
        return np.std(a, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims)
    else:
        w = np.array(weights)
        m = weighted_mean(a, weights=w, axis=axis, keepdims=True)
        return np.sqrt(np.sum(w * (np.array(a) - m) ** 2, axis=axis, dtype=dtype, keepdims=keepdims) /  # noqa: W504
                       (np.sum(w, axis=axis, dtype=dtype, keepdims=keepdims) - ddof))


def kde_smooth_peaks(X, mirror_left=None, mirror_right=None, copy=False, smoothing_width=1e-7, random_state=None):
    """Smooth non-unique values that show up as peaks in input dataset

    This is needed such that the cdf of the input data X has no sudden jumps.

    :param X: array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points.  Each row
        corresponds to a single data point.
    :param bool mirror_left: Mirror the data on a value on the left to counter signal leakage.
        Default is None, which is no mirroring.
    :param bool mirror_right: Mirror the data on a value on the right to counter signal leakage.
        Default is None, which is no mirroring.
    :param bool copy: if True, make a copy of the input data first before making changes.
    :param float smoothing_width: smear peaks of non-unique values with this width * (max-min). Default is 1e-7.
    :param int random_state: when an integer, the seed given random generator.
    :return: array_like, shape (n_samples, n_features)
        List of data points where non-unique peaks have been smoothed.
    """
    if len(X.shape) != 2:
        raise ValueError("Input 'X' should have 2 dimensions. Use reshape(-1, 1) if one dimension.")

    # modifications to the data happen below
    data = X.copy() if copy else X

    n_features = data.shape[1]

    # process left and right-hand mirror points
    mtypes = (list, tuple, np.ndarray)
    mirror_left = np.array(mirror_left) if isinstance(mirror_left, mtypes) else np.array([None] * n_features)
    mirror_right = np.array(mirror_right) if isinstance(mirror_right, mtypes) else np.array([None] * n_features)

    if len(mirror_left) != n_features:
        raise ValueError("Invalid size of 'mirror_left': %d. The number should match the data: %d."
                         % (len(mirror_left), n_features))
    if len(mirror_right) != n_features:
        raise ValueError("Invalid size of 'mirror_right': %d. The number should match the data: %d."
                         % (len(mirror_right), n_features))

    for i in range(n_features):
        data[:, i] = kde_smooth_peaks_1dim(data[:, i], mirror_left[i], mirror_right[i], copy=False,
                                           smoothing_width=smoothing_width, random_state=random_state)
    return data


def kde_smooth_peaks_1dim(X, mirror_left=None, mirror_right=None, copy=False, smoothing_width=1e-7, random_state=None):
    """Smooth non-unique values that show up as peaks in 1-dimensional input dataset

    This is needed such that the cdf of the input data X has no sudden jumps.

    If mirror_left and mirror_right parameters are set, for those values smoothing will
    be one-directional, to the right or left respectively.

    :param X: array containing numbers of which the non-unique ones will be smeared.
    :param bool mirror_left: Mirror the data on a value on the left to counter signal leakage.
        Default is None, which is no mirroring.
    :param bool mirror_right: Mirror the data on a value on the right to counter signal leakage.
        Default is None, which is no mirroring.
    :param bool copy: if True, make a copy of the input data first before making changes.
    :param float smoothing_width: smear peaks of non-unique values with this width * (max-min). Default is 1e-7.
    :param int random_state: when an integer, the seed given random generator.
    :return: array same as X but all non-unique numbers have be smeared.
    """
    # basic checks
    if X.ndim != 1:
        raise ValueError("Input 'X' should have dimension of 1.")

    mirror_left, mirror_right = _parse_mirror_points(X, mirror_left, mirror_right)
    return _kde_smooth_peaks(X, mirror_left, mirror_right, copy, smoothing_width, random_state)


def _kde_smooth_peaks(X, mirror_left=None, mirror_right=None, copy=False, smoothing_width=1e-7, random_state=None):
    """Internal function to smooth non-unique values in 1-dimensional input dataset

    This is needed such that the cdf of the input data X has no sudden jumps.

    If mirror_left and mirror_right parameters are set, for those values smoothing will
    be one-directional, to the right or left respectively.

    mirror_left and mirror_right parameters are not parsed first in this function.

    :param X: array containing numbers of which the non-unique ones will be smeared.
    :param bool mirror_left: Mirror the data on a value on the left to counter signal leakage.
        Default is None, which is no mirroring.
    :param bool mirror_right: Mirror the data on a value on the right to counter signal leakage.
        Default is None, which is no mirroring.
    :param bool copy: if True, make a copy of the input data first before making changes.
    :param float smoothing_width: smear peaks of non-unique values with this width * (max-min). Default is 1e-7.
    :param int random_state: when an integer, the seed given random generator.
    :return: array same as X but all non-unique numbers have be smeared.
    """
    # set random state
    np.random.seed(random_state)

    # sample profiles
    diff = max(X) - min(X)

    # modifications to the data happen below
    data = X.copy() if copy else X

    if mirror_left is not None:
        data[np.isclose(data, mirror_left)] = mirror_left
    if mirror_right is not None:
        data[np.isclose(data, mirror_right)] = mirror_right

    # find any peaks in the data and smooth them
    s = smoothing_width * (diff if not np.isclose(diff, 0) else 1)
    # find peaks
    u, c = np.unique(data, return_counts=True)
    peaks = u[c > 1]
    # smear them
    for v in peaks:
        idcs = np.where(np.isclose(data, v))[0]
        smeer = np.random.normal(0, s, size=len(idcs))
        if mirror_left is not None and np.isclose(v, mirror_left):
            smeer = np.abs(smeer)
        if mirror_right is not None and np.isclose(v, mirror_right):
            smeer = - np.abs(smeer)
        for i, idx in enumerate(idcs):
            data[idx] += smeer[i]

    return data


def _parse_mirror_points(X, mirror_left=None, mirror_right=None):
    """Properly parse and set left and right mirror points

    This is needed to distinguish between input options of number, bool or None.

    :param X: one-dimensional array containing numbers.
    :param bool mirror_left: Mirror the data on a value on the left to counter signal leakage.
        Default is None, which is no mirroring.
    :param bool mirror_right: Mirror the data on a value on the right to counter signal leakage.
        Default is None, which is no mirroring.
    :return: tuple with mirror_left and mirror_right mirror values.
    """
    # sample profiles
    min_orig = min(X)
    max_orig = max(X)

    # properly set left and right mirror points
    mtypes = (np.number, float, int)
    if mirror_left is not None:
        if (isinstance(mirror_left, bool) and mirror_left) or \
                (not isinstance(mirror_left, bool) and isinstance(mirror_left, mtypes)):
            mirror_left = mirror_left if isinstance(mirror_left, mtypes) else min_orig
        elif isinstance(mirror_left, bool) and not mirror_left:
            mirror_left = None
    if mirror_right is not None:
        if (isinstance(mirror_right, bool) and mirror_right) or \
                (not isinstance(mirror_right, bool) and isinstance(mirror_right, mtypes)):
            mirror_right = mirror_right if isinstance(mirror_right, mtypes) else max_orig
        elif isinstance(mirror_right, bool) and not mirror_right:
            mirror_right = None

    return mirror_left, mirror_right


def kde_process_data(X, n_quantiles=1000, smooth_peaks=True, mirror_left=None, mirror_right=None,
                     smoothing_width=1e-7, random_state=None):
    """Turn input array into a histogram useful for KDE evaluations

    From this histogram the KDE can be easily and quickly calculated.
    Then there is no need to keep the full input dataset for KDE evaluations, only the histogram summary.

    :param X: array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points.  Each row
        corresponds to a single data point.
    :param int n_quantiles: number of quantiles/bins used in output histogram. If greater than number of samples,
        this is reset to number of samples. Default is 1000.
    :param bool smooth_peaks: if False, do not smear peaks of non-unique values.
    :param bool mirror_left: Mirror the data on a value on the left to counter signal leakage.
        Default is None, which is no mirroring.
    :param bool mirror_right: Mirror the data on a value on the right to counter signal leakage.
        Default is None, which is no mirroring.
    :param float smoothing_width: smear peaks of non-unique values with this width * (max-min). Default is 1e-7.
    :param int random_state: when an integer, the seed given random generator.
    :return: tuple of bin entries and bin means of the quantiles.
        note: not bin centers in order to improve modelling accuracy.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 1:
        raise ValueError("Input data 'X' need to be a one-dimensional numpy array.")

    # sample profiles
    # note: important to do std after smooth-peaks in order to handle edge cases of unique points properly
    n_total = len(X)
    diff = max(X) - min(X)
    diff = diff if not np.isclose(diff, 0) else 1.

    # number of histogram bins to use below (= number of quantiles to evaluate)
    nbins = max(1, min(n_quantiles, n_total))

    # modifications to the data happen below
    data = X.copy()

    # parse and set left and right mirror points
    mirror_left, mirror_right = _parse_mirror_points(X, mirror_left, mirror_right)

    # find any peaks in the data and smooth them
    if smooth_peaks:
        data = _kde_smooth_peaks(data, mirror_left, mirror_right, smoothing_width=smoothing_width,
                                 random_state=random_state)

    # mirror data around left and/or right-most edge of data is so desired
    # do this to make sure that kde properly models the edges, where signal leakage
    # makes the KDE pdf drop.
    gstd = np.std(data)
    band_width_std = np.power(4 / 3, 0.2) * gstd * np.power(n_total, -0.2)
    mirror_width = max([10. * band_width_std, 10. * smoothing_width * diff])

    add_left = []
    add_right = []

    if mirror_left is not None:
        add_left = data[data <= mirror_left + mirror_width]
        add_left = -add_left + 2 * mirror_left
    if mirror_right is not None:
        add_right = data[data >= mirror_right - mirror_width]
        add_right = -add_right + 2 * mirror_right
    if mirror_left is not None or mirror_right is not None:
        data = np.concatenate([add_left, data, add_right], axis=None)

    # calculate quantiles
    ps = np.linspace(0, 1, nbins + 1)
    qs = np.quantile(data, ps)
    # this make sure that mean calculation of last bin (below) also works (b/c it contains an entry)
    qs[-1] += (1e-6 * diff) / n_quantiles

    # make summary histogram: weighted bin means & bin_entries
    bin_mean = np.array([np.mean(data[(data >= qs[i]) & (data < qs[i + 1])]) for i in range(len(qs) - 1)])
    bin_entries, bin_edges = np.histogram(data, bins=qs)

    # return a histogram that can replace X in the calculation of the KDE pdf
    return bin_entries, bin_mean


def _kde_histsum(x, bin_x, bin_entries, band_width, n_total):
    """Internal handy histogram sum function for KDE pdf calculation

    :param float x: input value for which to evaluate the kde pdf
    :param bin_x: array. bin mean values / centers of the binned data - output of kde_process_data function.
    :param bin_entries: array. bin entries of the binned data - output of kde_process_data function.
    :param band_width: array. bandwidth of to be used for each bin - output of kde_bw function.
    :param int n_total: total number of bin entries
    :return: float, KDE pdf value for x.
    """
    if not isinstance(x, (float, int, np.number)):
        raise RuntimeError('x has wrong type')
    return np.sum(bin_entries * norm.pdf(x, loc=bin_x, scale=band_width)) / n_total


def _kde_pdf(x, bin_x, bin_entries=None, band_width=None):
    """Internal handy function for KDE pdf calculation

    :param x: array or number. input value(s) for which to evaluate the kde pdf
    :param bin_x: array. bin mean values / centers of the binned data - output of kde_process_data function.
    :param bin_entries: array. bin entries of the binned data - output of kde_process_data function. Default is None.
    :param band_width: array. bandwidth to be used for each bin - output of kde_bw function. Default is None.
    :return: array or number. KDE pdf value(s) for x.
    """
    # basic input checks and set up
    if not isinstance(x, (float, int, np.number, np.ndarray, list, tuple)):
        raise RuntimeError('x has wrong type')
    if bin_entries is not None:
        if bin_x.shape != bin_entries.shape:
            raise RuntimeError('bin_entries has wrong type')
    if band_width is None:
        # pick up zero-order band-width
        band_width = kde_bw(bin_x, bin_entries, n_adaptive=0)
    n_total = len(bin_x) if bin_entries is None else np.sum(bin_entries)
    if bin_entries is None:
        bin_entries = 1.0

    # evaluate kdf pdf at x
    if isinstance(x, (float, int, np.number)):
        p = _kde_histsum(x, bin_x, bin_entries, band_width, n_total)
    elif isinstance(x, (np.ndarray, list, tuple)):
        x = np.array(x)
        p = np.array([_kde_histsum(xi, bin_x, bin_entries, band_width, n_total) for xi in x.ravel()]).reshape(x.shape)
    return p


def kde_bw(bin_x, bin_entries, rho=1, n_adaptive=0):
    """Evaluate the adaptive KDE bandwidth

    Adaptive bandwidths are evaluated with the formulas described in:
    Cranmer KS, Kernel Estimation in High-Energy Physics. Computer Physics Communications 136:198-207, 2001
    e-Print Archive: hep ex/0011057

    :param bin_x: array. bin mean values / centers of the binned data - output of kde_process_data function.
    :param bin_entries: array. bin entries of the binned data - output of kde_process_data function.
    :param float rho: bandwidth scale parameter. default is 1.0.
    :param int n_adaptive: number of adaptive iterations to be applied to improve the band width. default is 0.
    :return: array. evaluated band width corresponding to each bin.
    """
    assert isinstance(n_adaptive, (int, np.integer)) and n_adaptive >= 0

    # pass 0: constant bandwidth
    gstd = weighted_std(bin_x, weights=bin_entries)
    n_total = len(bin_x) if bin_entries is None else np.sum(bin_entries)
    band_width = rho * np.power(4 / 3, 0.2) * gstd * np.power(n_total, -0.2)

    # band widths are improved iteratively by calling _kde to evaluate local density.
    for _ in range(n_adaptive):
        # first use the existing bandwidth to evaluate pdf
        ps = _kde_pdf(bin_x, bin_x, bin_entries, band_width)
        # then calculate the updated "adaptive" bandwidth
        band_width = rho * np.power(4 / 3, 0.2) * np.sqrt(gstd) * np.power(n_total, -0.2) / np.sqrt(ps)

    return band_width


def kde_pdf(x, bin_x, bin_entries=None, band_width=None, rho=1, n_adaptive=0, ret_bw=False, normalization=1.0):
    """Function for KDE pdf calculation

    :param x: array or number. input value(s) for which to evaluate the kde pdf
    :param bin_x: array. bin mean values / centers of the binned data - output of kde_process_data function.
    :param bin_entries: array. bin entries of the binned data - output of kde_process_data function. Default is None.
    :param band_width: array. bandwidth to be used for each bin - output of kde_bw function. Default is None.
    :param float rho: bandwidth scale parameter. default is 1.0.
    :param int n_adaptive: number of adaptive iterations to be applied to improve the band width. default is 0.
    :param bool ret_bw: if true, return bandwidth array next to calculated probability value(s).
    :param float normalization: pdf normalization value. If different from 1.0, all calculated probability value(s)
        are divided by this number.
    :return: array or number. KDE pdf value(s) for x.
    """
    if band_width is None:
        band_width = kde_bw(bin_x, bin_entries, rho=rho, n_adaptive=n_adaptive)
    p = _kde_pdf(x, bin_x, bin_entries, band_width)
    if normalization != 1.0:
        p /= normalization
    return (p, band_width) if ret_bw else p


def kde_make_transformers(bin_mean, bin_entries, band_width=None, x_min=None, x_max=None, rho=1.0, n_bins=1000,
                          min_pdf_value=1e-20):
    """Create a pdf, cdf and inverse-cdf of a KDE distribution

    These are put into interpolation functions for fast evaluation later.

    :param bin_mean: array. bin mean values / centers of the binned data - output of kde_process_data function.
    :param bin_entries: array. bin entries of the binned data - output of kde_process_data function.
    :param band_width: array. bandwidth to be used for each bin - output of kde_bw function. Default is None.
    :param float x_min: minimum value of pdf's x range. default is None (= - inf)
    :param float x_max: maximum value of pdf's x range. default is None (= + inf)
    :param float rho: bandwidth scale parameter. default is 1.0.
    :param int n_bins: for internal evaluation, number of integration bins beyond x-range. default is 1000.
    :param float min_pdf_value: minimum kde pdf value. default is 1e-20.
    :return: tuple of pdf, cdf, inverse cdf, pdf normalization value
    """
    # set the integration range
    if x_min is None or x_max is None:
        gstd = weighted_std(bin_mean, weights=bin_entries)
        n_total = len(bin_mean) if bin_entries is None else np.sum(bin_entries)
        bw = np.power(4 / 3, 0.2) * gstd * np.power(n_total, -0.2)
        x_min = bin_mean[0] - 10 * bw if x_min is None else x_min
        x_max = bin_mean[-1] + 10 * bw if x_max is None else x_max

    # The x grid we'll use for integration
    # We merge linear and quantile points along x-axis.
    # center is quantiles. left and right we pad with linear spacing

    # the center uses quantile binning
    x_center = bin_mean[(bin_mean > x_min) & (bin_mean < x_max)]

    # on the left and right we use linear binning
    x_padding = np.linspace(x_min, x_max, n_bins + 1)
    x_linear_left = x_padding[x_padding < x_center[0]]
    x_linear_right = x_padding[x_padding > x_center[-1]]

    # in the transitions from linear to quantile we use same binning as quantile
    step_x_left = x_center[1] - x_center[0]
    step_x_right = x_center[-1] - x_center[-2]

    x_transition_left = np.arange(x_linear_left[-1], x_center[0], step_x_left) if step_x_left > 0 else []
    x_transition_right = np.arange(x_center[-1], x_linear_right[0], step_x_right) if step_x_right > 0 else []

    if len(x_transition_left) > 100:
        x_transition_left = x_transition_left[-100:]
    if len(x_transition_right) > 100:
        x_transition_right = x_transition_right[:100]

    # merge all pieces together
    x_grid = np.concatenate([x_linear_left, x_transition_left, x_center, x_transition_right, x_linear_right], axis=None)
    x_grid = np.unique(x_grid)

    # mid points for Simpson's integration rule
    x_mid = x_grid[:-1] + np.diff(x_grid)

    p_grid, bw = kde_pdf(x_grid, bin_x=bin_mean, bin_entries=bin_entries, band_width=band_width, rho=rho, ret_bw=True)
    p_mid = kde_pdf(x_mid, bin_x=bin_mean, bin_entries=bin_entries, band_width=bw, rho=rho)

    # use simpsons rule for integration
    integral_sections = np.array(
        [((x_grid[i + 1] - x_grid[i]) / 6.) * (p_grid[i] + 4. * pm + p_grid[i + 1]) for i, pm in enumerate(p_mid)])

    # at first x-point integral is zero by construction; add artificial point
    # adding epsilon at point 0 to ensure that cumsum never reaches zero and can always be inverted.
    epsilon = 10 * np.finfo(float).eps
    integral_sections = np.concatenate([[epsilon], integral_sections], axis=None)

    # adding epsilon at point -1 to ensure that cumsum never reaches 1 and can always be inverted.
    cumsum = np.cumsum(integral_sections)
    kde_norm = cumsum[-1]
    cumsum_to = cumsum / (kde_norm * (1. + epsilon))
    cumsum_from = cumsum / kde_norm

    F = interpolate.interp1d(x_grid, cumsum_to, bounds_error=False, fill_value=(cumsum[0], cumsum[-1]))
    Finv = interpolate.interp1d(cumsum_from, x_grid, bounds_error=False, fill_value="extrapolate")

    pdf_norm = p_grid / kde_norm
    fast_pdf = interpolate.interp1d(x_grid, pdf_norm, bounds_error=False, fill_value=(min_pdf_value, min_pdf_value))

    return fast_pdf, F, Finv, kde_norm
