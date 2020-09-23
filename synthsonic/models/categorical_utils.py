import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def categorical_round(data, cols):
    data[:, cols] = np.round(data[:, cols])
    return data


def vec_translate(a, my_dict):
    return np.vectorize(my_dict.__getitem__)(a)


def categorical_frequency_mapping(data, columns):
    """Sort the """
    inv_mappings = {}
    for col in columns:
        unique_values, counts = np.unique(data[:, col], return_counts=True)
        sorted_values = unique_values[np.argsort(-counts)]

        mapping = {k: v for k, v in zip(unique_values, sorted_values)}
        data[:, col] = vec_translate(data[:, col], mapping)

        inv_mappings[col] = {k: v for k, v in zip(sorted_values, unique_values)}
    return data, inv_mappings


def categorical_frequency_inverse_mapping(data, columns, inv_mappings):
    for col in columns:
        data[:, col] = vec_translate(data[:, col], inv_mappings[col])

    data = categorical_round(data, columns)
    return data


def encode_one_hot(df, cols):
    categorical_data = pd.DataFrame(df[:, cols], columns=cols)
    one_hot_encoded = pd.get_dummies(data=categorical_data, columns=cols).values
    unique_values = [np.unique(df[:, col]) for col in cols]
    return unique_values, one_hot_encoded


def decode_one_hot(samples, columns, unique_values):
    recreated = np.empty((samples.shape[0], len(columns)))
    end_idx = 0
    for col in columns:
        start_idx = end_idx
        end_idx = start_idx + len(unique_values[col])

        indices = samples[:, start_idx : end_idx].argmax(axis=1).astype(int)
        assert np.max(indices) <= end_idx - start_idx
        recreated[:, col] = unique_values[col][indices]
    return recreated


def encode_integer(data, categorical_columns):
    enc = OrdinalEncoder()
    encoded_data = enc.fit_transform(data[:, categorical_columns])
    return encoded_data, enc


def decode_integer(encoded_data, enc):
    return enc.inverse_transform(encoded_data)
