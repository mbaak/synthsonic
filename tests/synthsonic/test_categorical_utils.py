import numpy as np
import pytest

from synthsonic.models.categorical_utils import encode_one_hot, decode_one_hot, \
    categorical_frequency_mapping, categorical_frequency_inverse_mapping


@pytest.fixture(scope='function')
def data():
    n_features = 10
    n_rows = 25

    data = np.arange(n_features * n_rows).reshape((n_rows, n_features))
    data[5:8, 5:8] = 4
    return data


def test_categorical_one_hot(data):
    columns = list(range(data.shape[1]))

    unique_values, encoded = encode_one_hot(data, columns)
    # (model goes here)
    recreated = np.empty_like(data)

    # Start idx is used when other variable types are present
    start_idx = 0
    recreated[:, columns] = decode_one_hot(encoded[:, start_idx:], columns, unique_values)
    assert (recreated == data).all()


def test_categorical_mapping(data):
    columns = list(range(data.shape[1]))

    old_data = data.copy()
    data, inv_mappings = categorical_frequency_mapping(data, columns)

    assert (categorical_frequency_inverse_mapping(data, columns, inv_mappings) == old_data[:, columns]).all()
