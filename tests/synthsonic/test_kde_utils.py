import numpy as np
from numpy import random
from synthsonic.models.kde_utils import kde_process_data, kde_pdf, kde_bw, kde_make_transformers, kde_smooth_peaks_1dim


def get_data():
    # generate random normal distribution
    np.random.seed(42)
    nevs = 10000
    g = random.normal(size=nevs)
    return g


def test_kde_process_data():
    g = get_data()

    # test conversion of data points to quantile histogram (input to kde pdf)
    bin_entries, bin_mean = kde_process_data(g, n_quantiles=100)

    entries = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                        100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                        100, 100, 100, 100, 100])

    mean = np.array([-2.69683947, -2.16589276, -1.97339563, -1.826522, -1.71423698, -1.59620703, -1.51893032,
                     -1.44467834, -1.38110361, -1.32440524, -1.26058369, -1.20251124, -1.14988351, -1.10303419,
                     -1.05924954, -1.01452836, -0.98069737, -0.93961851, -0.90055278, -0.86104473, -0.82794739,
                     -0.79667152, -0.76321923, -0.72820436, -0.69218269, -0.65542617, -0.62611587, -0.59988624,
                     -0.56995318, -0.54165648, -0.51298208, -0.48507374, -0.45807071, -0.42984014, -0.39956605,
                     -0.37469989, -0.34682477, -0.31997796, -0.29569161, -0.26759096, -0.24060449, -0.21830682,
                     -0.19327703, -0.16859764, -0.14055805, -0.11343289, -0.08595467, -0.06111565, -0.03887246,
                     -0.01617315, 0.01020131, 0.03313447, 0.06009455, 0.0848424, 0.11077313, 0.13876702, 0.16585931,
                     0.19047522, 0.21216093, 0.23720022, 0.26418654, 0.29128651, 0.31649046, 0.33935467, 0.3697868,
                     0.395873, 0.42131099, 0.44859928, 0.47628065, 0.50266571, 0.53078718, 0.56273067, 0.59396888,
                     0.62356542, 0.65505875, 0.68510918, 0.71255828, 0.74573284, 0.78297137, 0.82184578, 0.85712487,
                     0.89342921, 0.93965807, 0.97655332, 1.02154771, 1.06272492, 1.10761346, 1.15861989, 1.20115379,
                     1.25561263, 1.31579855, 1.37798418, 1.44627022, 1.52084352, 1.60484395, 1.69024116, 1.81291632,
                     1.97468856, 2.17466409, 2.69283089])

    np.testing.assert_array_almost_equal(bin_entries, entries)
    np.testing.assert_array_almost_equal(bin_mean, mean)


def test_kde_bw():
    g = get_data()

    # test conversion of data points to quantile histogram (input to kde pdf)
    bin_entries, bin_mean = kde_process_data(g, n_quantiles=100)

    # calculate adaptive band width per histogram bin
    band_width = kde_bw(bin_mean, bin_entries, n_adaptive=1)

    bw = np.array([1.08764512, 0.83991465, 0.67359994, 0.59927195, 0.5489647, 0.49854079, 0.46878879, 0.44405272,
                   0.42575916, 0.41107038, 0.39560558, 0.38210904, 0.37033498, 0.36036844, 0.35163946, 0.34341234,
                   0.33767791, 0.33126823, 0.32568749, 0.32048028, 0.31639566, 0.31272361, 0.30896097, 0.30517737,
                   0.30143739, 0.29778557, 0.29500531, 0.29262796, 0.29005484, 0.28776889, 0.28560348, 0.28364336,
                   0.28188315, 0.28018133, 0.27850578, 0.27723882, 0.27592911, 0.27477254, 0.27381094, 0.27279495,
                   0.27191404, 0.27125483, 0.27058792, 0.2700055, 0.26943407, 0.26897267, 0.26859689, 0.26833633,
                   0.26816628, 0.26805362, 0.2679986, 0.26801562, 0.26811105, 0.2682692, 0.26850699, 0.26884771,
                   0.26926371, 0.26971968, 0.2701872, 0.27080949, 0.27158691, 0.27248776, 0.27344101, 0.27440697,
                   0.27584803, 0.27722839, 0.27870562, 0.28043542, 0.28234421, 0.28430857, 0.28655904, 0.28931417,
                   0.29221669, 0.29516014, 0.29850099, 0.3018878, 0.3051455, 0.30927723, 0.31413798, 0.31941538,
                   0.32433906, 0.32950349, 0.33619965, 0.3416572, 0.34852236, 0.35511356, 0.3627786, 0.37227289,
                   0.3809316, 0.39305637, 0.40775561, 0.42422351, 0.44372209, 0.46707564, 0.49728521, 0.53404882,
                   0.59809044, 0.69191338, 0.85724562, 1.08664638])

    np.testing.assert_array_almost_equal(band_width, bw)


def test_kde_pdf():
    g = get_data()

    # test conversion of data points to quantile histogram (input to kde pdf)
    bin_entries, bin_mean = kde_process_data(g, n_quantiles=100)

    # calculate adaptive band width per histogram bin
    band_width = kde_bw(bin_mean, bin_entries, n_adaptive=1, rho=0.5)

    # test pdf values
    x = np.array([3.5, 2.5, 1.5, 0.5])
    p = kde_pdf(x, bin_mean, bin_entries, band_width)
    np.testing.assert_array_almost_equal(p, [0.00118108, 0.02138821, 0.12732973, 0.35486775])


def test_kde_transformers():
    g = get_data()

    # test conversion of data points to quantile histogram (input to kde pdf)
    bin_entries, bin_mean = kde_process_data(g, n_quantiles=100)

    # calculate adaptive band width per histogram bin
    band_width = kde_bw(bin_mean, bin_entries, n_adaptive=1, rho=0.5)

    # get fast pdf, cdf, invcdf, and pdf normalization
    pdf, F, Finv, kdenorm = kde_make_transformers(bin_mean, bin_entries, band_width)

    # test pdf normalization
    np.testing.assert_almost_equal(kdenorm, 1.002632731018065)

    # test fast pdf values
    x = np.array([3.5, 2.5, 1.5, 0.5])
    p = pdf(x)
    np.testing.assert_array_almost_equal(p, [0.001178, 0.023228, 0.12709176, 0.35392584])

    # test fast pdf and cdf values
    xnew = np.arange(-4, 4, 0.2)
    p = pdf(xnew)
    y = F(xnew)

    probs = np.array([3.44959686e-05, 1.73983233e-04, 6.73043274e-04, 2.00604518e-03, 4.65036398e-03, 8.56904467e-03,
                      1.31313945e-02, 1.95744411e-02, 2.77637833e-02, 3.59531256e-02, 5.10102240e-02, 7.38428471e-02,
                      1.06950025e-01, 1.46937498e-01, 1.91381870e-01, 2.43938851e-01, 2.88358444e-01, 3.32126326e-01,
                      3.67886104e-01, 3.88906306e-01, 3.97247174e-01, 3.91939064e-01, 3.72559530e-01, 3.31953535e-01,
                      2.80741025e-01, 2.35674397e-01, 1.94211691e-01, 1.48340443e-01, 1.07459700e-01, 7.36938349e-02,
                      4.94996058e-02, 3.44533975e-02, 2.69697990e-02, 1.94862004e-02, 1.33438997e-02, 8.61408617e-03,
                      4.62443953e-03, 1.97718950e-03, 6.58550743e-04, 1.69103123e-04])

    np.testing.assert_array_almost_equal(p, probs)

    cdfv = np.array([3.69523053e-06, 2.17190884e-05, 9.85541466e-05, 3.51821658e-04, 1.00139885e-03, 2.31793155e-03,
                     4.49589175e-03, 9.25026554e-03, 1.59955225e-02, 2.27407795e-02, 3.22784778e-02, 4.57053858e-02,
                     6.47813350e-02, 9.11575340e-02, 1.25667938e-01, 1.70000926e-01, 2.23956807e-01, 2.86377882e-01,
                     3.57028483e-01, 4.33092527e-01, 5.12044239e-01, 5.91019459e-01, 6.67717753e-01, 7.37994480e-01,
                     7.98782088e-01, 8.49588400e-01, 8.92107045e-01, 9.25310645e-01, 9.49814959e-01, 9.66568944e-01,
                     9.77380380e-01, 9.84521186e-01, 9.88370009e-01, 9.92218832e-01, 9.95569707e-01, 9.97745041e-01,
                     9.99038697e-01, 9.99666639e-01, 9.99907796e-01, 9.99979944e-01])

    np.testing.assert_array_almost_equal(y, cdfv)

    # test pdf normalization from restricted x-range
    pdf, F, Finv, kdenorm = kde_make_transformers(bin_mean, bin_entries, band_width, x_min=-1, x_max=1)
    np.testing.assert_almost_equal(kdenorm, 0.6814761490561737)


def test_kde_smooth_peaks_1dim():
    x = np.ones(100)

    # test smoothing of non-unique peaks
    x_out = kde_smooth_peaks_1dim(x)
    u, c = np.unique(x_out, return_counts=True)
    np.testing.assert_equal(np.all(c == 1), True)

    # test mirror around right-edge
    x = np.ones(100)
    x_out = kde_smooth_peaks_1dim(x, mirror_right=1.0)
    np.testing.assert_equal(np.all(x_out < 1), True)

    # test mirror around left-edge
    x = np.ones(100)
    x_out = kde_smooth_peaks_1dim(x, mirror_left=1.0)
    np.testing.assert_equal(np.all(x_out > 1), True)
