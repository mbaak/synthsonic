import copy

# import pandas as pd

from sdgym.synthesizers.base import BaseSynthesizer


class IdentitySynthesizer(BaseSynthesizer):
    """Trivial synthesizer.

    Returns the same exact data that is used to fit it.
    """

    def fit(self, train_data, *args):
        self.data = copy.deepcopy(train_data)

    def sample(self, samples):
        return self.data
        # return self.data.sample(samples, replace=True).to_numpy().copy()
