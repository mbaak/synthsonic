import numpy as np
import pytest

from synthsonic.models.categorical_utils import encode_one_hot, decode_one_hot, \
    categorical_frequency_mapping, categorical_frequency_inverse_mapping, encode_integer, \
    decode_integer


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


def test_encode_integer():
    data = np.array([[1, 2, 3, 'text with space', 'AL', 'male'],
                    [4, 5, 6, 'random text', 'TX', 'female'],
                    [4, 5, 6, 'random different text', 'IL', 'na']])
    categorical_cols = [3, 4, 5]
    original_data = data[:, categorical_cols]
    encoded_data, enc = encode_integer(data, categorical_cols)
    encoded_test = np.array([[2., 0., 1.],
                             [1., 2., 0.],
                             [0., 1., 2.]])
    decoded_data = decode_integer(encoded_data, enc)
    assert np.allclose(encoded_data, encoded_test)
    assert (original_data == decoded_data).all()

