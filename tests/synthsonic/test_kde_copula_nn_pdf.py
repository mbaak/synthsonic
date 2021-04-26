import numpy as np
import pandas as pd
from synthsonic.models.kde_copula_nn_pdf import KDECopulaNNPdf
from scipy.stats import multivariate_normal


def get_data():
    # example dataset with correlations
    df = pd.read_csv('notebooks/001-mb-kdecopulannpdf-examples/correlated_data.sv', sep=' ')
    df.drop(labels=['Unnamed: 5'], inplace=True, axis=1)
    data = df.values
    return data


def get_normal_data():
    # generate bivariate gaussian with correlation
    mux = 0
    muy = 0
    sigmax = 1
    sigmay = 1
    rho = 0.7
    N = 100000

    np.random.seed(42)
    X2 = np.random.multivariate_normal([mux, muy], [[sigmax * sigmax, rho * sigmax * sigmay],
                                                    [rho * sigmax * sigmay, sigmay * sigmay]], size=N)

    # theoretical pdf values
    rv = multivariate_normal([mux, muy], [[sigmax * sigmax, rho * sigmax * sigmay],
                                          [rho * sigmax * sigmay, sigmay * sigmay]])
    p2 = rv.pdf(X2)

    return X2, p2


def test_fit_transformations():
    data = get_data()

    # ranges for pdf normalization
    # none means set autimatically.
    x_min = [None] * 5
    x_max = [None] * 5
    x_min[0] = -1.
    x_max[0] = 1.
    x_min = np.array(x_min)
    x_max = np.array(x_max)

    kde = KDECopulaNNPdf(x_min=x_min, x_max=x_max, rho=0.5).fit(data)

    np.testing.assert_array_equal(kde.x_min, [-1.0, None, None, None, None])
    np.testing.assert_array_equal(kde.x_max, [1.0, None, None, None, None])

    # test 1st transformation: kde-quantile to normal distribution
    X_g = kde.pipe_[0].transform(data)
    hist, bin_edges = np.histogram(X_g[:, 0], bins=50)

    entries = np.array([1, 1, 0, 1, 2, 2, 2, 3, 4, 5, 5, 7, 8, 10, 11, 13, 14,
                        16, 17, 19, 20, 21, 22, 23, 23, 23, 23, 22, 21, 20, 19, 17, 16, 14,
                        13, 11, 10, 8, 7, 5, 5, 4, 3, 2, 2, 2, 1, 0, 1, 1])
    np.testing.assert_array_equal(hist, entries)

    # test 2nd transformation: pca
    X_p = kde.pipe_[0:2].transform(data)
    hist, bin_edges = np.histogram(X_p[:, 1], bins=40)

    entries = np.array([2, 2, 2, 2, 7, 10, 6, 8, 7, 18, 15, 11, 16, 28, 21, 15, 22,
                        24, 26, 25, 27, 29, 26, 17, 21, 20, 18, 15, 8, 10, 11, 9, 9, 5,
                        3, 0, 0, 3, 1, 1])
    np.testing.assert_array_equal(hist, entries)

    # test 3rd transformation: pca to uniform distribution
    X_u = kde.pipe_.transform(data)
    hist, bin_edges = np.histogram(X_u[:, 2], bins=40)

    entries = np.array([13, 12, 13, 13, 12, 13, 12, 13, 12, 12, 12, 13, 13, 12, 13, 12, 13,
                        12, 13, 12, 13, 12, 13, 12, 12, 13, 12, 13, 12, 12, 14, 12, 12, 12,
                        13, 13, 11, 14, 12, 13])
    np.testing.assert_array_equal(hist, entries)

    # test correlation coefficients
    corr = np.corrcoef(X_p[:, 0], X_p[:, 1])[0][1]
    np.testing.assert_almost_equal(corr, -2.0263726648922494e-17)

    corr = np.corrcoef(X_u[:, 0], X_u[:, 1])[0][1]
    np.testing.assert_almost_equal(corr, -0.00238107108964288)

    # test nll sum
    score = kde.score(data)
    np.testing.assert_almost_equal(score, -2538.2435292716964)

    # test sample generation
    X_gen, sample_weight = kde.sample(200000, random_state=42)
    np.testing.assert_almost_equal(np.sum(sample_weight), 19056.719368079168)

    hist, bin_edges = np.histogram(X_gen[:, 1], bins=40)
    entries = np.array([113, 101, 91, 70, 61, 52, 59, 83, 136,
                        262, 579, 1101, 1584, 3289, 5883, 8537, 13258, 16205,
                        19779, 23657, 24862, 23703, 18458, 12934, 9594, 6113, 3750,
                        2312, 1103, 641, 408, 340, 263, 200, 121, 89,
                        71, 57, 46, 35])
    np.testing.assert_array_equal(hist, entries)

    hist, bin_edges = np.histogram(X_gen[:, 3], bins=40)
    entries = np.array([65, 67, 69, 83, 66, 113, 147, 212, 305,
                        526, 974, 1550, 2497, 4224, 7067, 11100, 15886, 20214,
                        23491, 24409, 22738, 19906, 15547, 11247, 7285, 4273, 2397,
                        1329, 734, 422, 258, 171, 132, 81, 79, 78,
                        59, 53, 76, 70])
    np.testing.assert_array_equal(hist, entries)

    # sampling without weights (using accept-reject method)
    # number of returned data points will be equal or less than n_sample
    X_gen = kde.sample_no_weights(200000, random_state=42)
    assert len(X_gen) <= 200_000


def test_pdf_values():
    X2, p2 = get_normal_data()
    pdf = KDECopulaNNPdf(rho=0.4).fit(X2)

    # test nll sum
    score = pdf.score(X2)
    np.testing.assert_almost_equal(score, -583162.2048191493)

    # test probability density values
    p = pdf.pdf(X2)

    values = np.array([0.00704714, 0.00208712, 0.0079493 , 0.0017365 , 0.00634369,
                       0.00638151, 0.00120891, 0.00151661, 0.00464017, 0.0020796])
    np.testing.assert_array_almost_equal(p[:10], values)

    delta_p = p - p2
    np.testing.assert_almost_equal(np.mean(delta_p), -0.10733660695446766)
    np.testing.assert_almost_equal(np.std(delta_p), 0.06201152010732619)
