import pandas as pd
import numpy as np
from synthsonic.models.kde_quantile_tranformer import KDEQuantileTransformer


def get_data():
    # example dataset with correlations and 5 variables
    df = pd.read_csv('notebooks/001-mb-kdecopulannpdf-examples/correlated_data.sv', sep=' ')
    df.drop(labels=['Unnamed: 5'], inplace=True, axis=1)
    data = df.values
    return data


def test_fit_init():
    data = get_data()

    # fitting and initialization
    x_min = [None] * 5
    x_max = [None] * 5
    x_min[0] = -1.
    x_max[0] = 1.
    x_min = np.array(x_min)
    x_max = np.array(x_max)

    kt = KDEQuantileTransformer(mirror_left=x_min, mirror_right=x_max, x_min=x_min, x_max=x_max, n_quantiles=500)
    kt = kt.fit(data)

    np.testing.assert_array_almost_equal(kt.x_max, [1.0, 5.9849285644844, 6.882161249497145,
                                                    6.240404095253973, 5.94480442281632])
    np.testing.assert_array_almost_equal(kt.x_min, [-1.0, -6.085097064484399, -6.201666549497145,
                                                    -5.9606727952539735, -6.13638422281632])
    np.testing.assert_array_equal(kt.mirror_left, [-1.0, None, None, None, None])
    np.testing.assert_array_equal(kt.mirror_right, [1.0, None, None, None, None])
    np.testing.assert_array_equal(kt.rho, [0.5, 0.5, 0.5, 0.5, 0.5])


def test_transform():
    data = get_data()

    # fitting and initialization
    x_min = [None] * 5
    x_max = [None] * 5
    x_min[0] = -1.
    x_max[0] = 1.
    x_min = np.array(x_min)
    x_max = np.array(x_max)

    kt = KDEQuantileTransformer(mirror_left=x_min, mirror_right=x_max, x_min=x_min, x_max=x_max, n_quantiles=500)
    kt = kt.fit(data)

    # transformation to uniform
    u = kt.transform(data)

    np.testing.assert_array_almost_equal(np.min(u, axis=0), [0.00200185, 0.00200075, 0.00199739, 0.00200288,
                                                             0.00199725])
    np.testing.assert_array_almost_equal(np.max(u, axis=0), [0.99800344, 0.99799953, 0.99801163, 0.99800202,
                                                             0.99799813])

    hist, bin_edges = np.histogram(u[:, 0], bins=40)
    entries = np.array([13, 12, 13, 13, 12, 12, 13, 13, 12, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 12, 13, 13, 12,
                        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13])
    np.testing.assert_array_equal(hist, entries)

    hist, bin_edges = np.histogram(u[:, 4], bins=40)
    entries = np.array([13, 12, 13, 12, 13, 13, 12, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 12,
                        13, 12, 13, 12, 13, 12, 13, 12, 12, 13, 13, 12, 12, 13, 13, 12, 13])
    np.testing.assert_array_equal(hist, entries)


def test_inverse_transform():
    data = get_data()

    # fitting and initialization
    # none means set automatically.
    x_min = [None] * 5
    x_max = [None] * 5
    x_min[0] = -1.
    x_max[0] = 1.
    x_min = np.array(x_min)
    x_max = np.array(x_max)

    # use_inverse_qt=True makes the inversion exact (= same as input distribution).
    # use_inverse_qt=False transforms to the smoothed kde distribution.
    kt = KDEQuantileTransformer(mirror_left=x_min, mirror_right=x_max, x_min=x_min, x_max=x_max, n_quantiles=500,
                                use_inverse_qt=True)
    kt = kt.fit(data)

    # inverse transformation
    u = kt.transform(data)
    y = kt.inverse_transform(u)

    hist, bin_edges = np.histogram(y[:, 1], bins=40)
    entries = np.array([1, 1, 1, 1, 2, 2, 4, 4, 6, 9, 10, 13, 14, 18, 21, 23, 24, 28, 28, 30, 29, 30, 28,
                        26, 26, 22, 19, 17, 14, 12, 9, 7, 6, 5, 3, 2, 2, 1, 1, 1])
    np.testing.assert_array_equal(hist, entries)

    hist, bin_edges = np.histogram(y[:, 2], bins=40)
    entries = np.array([1, 0, 1, 2, 2, 3, 5, 6, 9, 12, 14, 19, 22, 25, 28, 32, 33, 35, 34, 34, 31, 29, 26, 21, 20,
                        14, 12, 9, 6, 6, 3, 2, 2, 1, 0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(hist, entries)

    hist2, bin_edges = np.histogram(data[:, 2], bins=bin_edges)
    np.testing.assert_array_equal(hist2, entries)


def test_inverse_jacobian():
    data = get_data()

    # fitting and initialization
    x_min = [None] * 5
    x_max = [None] * 5
    x_min[0] = -1.
    x_max[0] = 1.
    x_min = np.array(x_min)
    x_max = np.array(x_max)

    kt = KDEQuantileTransformer(mirror_left=x_min, mirror_right=x_max, x_min=x_min, x_max=x_max, n_quantiles=500)
    kt = kt.fit(data)

    u = kt.transform(data)
    inv_jac = kt.inverse_jacobian(u)
    ij = [1.32938513e-03, 1.09859511e-02, 1.16347084e-02, 6.50289831e-04, 6.92759272e-03, 7.89634168e-05,
          1.54534709e-03, 7.34501741e-03, 6.14355849e-03, 8.81810294e-03]

    np.testing.assert_array_almost_equal(inv_jac[:10], ij)


def test_jacobian():
    data = get_data()

    # fitting and initialization
    x_min = [None] * 5
    x_max = [None] * 5
    x_min[0] = -1.
    x_max[0] = 1.
    x_min = np.array(x_min)
    x_max = np.array(x_max)

    kt = KDEQuantileTransformer(mirror_left=x_min, mirror_right=x_max, x_min=x_min, x_max=x_max, n_quantiles=500,
                                output_distribution='normal')
    kt = kt.fit(data)

    jac = kt.jacobian(data)
    j = [0.40192749, 0.50745591, 0.71506617, 0.88604897, 0.73708417, 0.36197164, 0.84148185, 0.56407231,
         0.50541513, 0.55398609]

    np.testing.assert_array_almost_equal(jac[:10], j)
