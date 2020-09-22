from sdgym.synthesizers.base import BaseSynthesizer

from synthsonic.models.kde_copula_nn_pdf import KDECopulaNNPdf


class KDECopulaNNPDFSynthesizer(BaseSynthesizer):
    def __init__(self, *args, **kwargs):
        self.kde = KDECopulaNNPdf(*args, **kwargs)

    def fit(self, train_data):
        self.kde = self.kde.fit(train_data)

    def sample(self, samples):
        X_gen, sample_weight = self.kde.sample(samples)
        return X_gen
