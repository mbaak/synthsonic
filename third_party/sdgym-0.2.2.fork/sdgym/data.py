import json
import logging
import os
import urllib

import pandas as pd
import numpy as np

from sdgym.constants import CATEGORICAL, ORDINAL

LOGGER = logging.getLogger(__name__)

BASE_URL = 'http://sdgym.s3.amazonaws.com/datasets/'
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(filename, loader):
    local_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(local_path):
        os.makedirs(DATA_PATH, exist_ok=True)
        url = BASE_URL + filename

        LOGGER.info('Downloading file %s to %s', url, local_path)
        urllib.request.urlretrieve(url, local_path)

    return loader(local_path)


def _get_columns(metadata, data, distinct_threshold):
    categorical_columns = []
    ordinal_columns = []
    numeric_columns = []

    df = pd.DataFrame(data)
    for column_idx, column in enumerate(metadata['columns']):
        counts = df[column_idx].value_counts()
        distinct = len(counts)

        if column['type'] == CATEGORICAL or (distinct_threshold is not None and distinct < distinct_threshold):
            if column['type'] != CATEGORICAL:
                LOGGER.info(
                    f"'{column['name']}' was converted to CATEGORICAL because number of distinct values "
                    f"{distinct} was lower than the threshold {distinct_threshold}"
                )
                LOGGER.info(
                    f"Most frequent 10 values: {counts.head(20).index.values.tolist()}"
                )
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)
        else:
            numeric_columns.append(column_idx)

    return categorical_columns, ordinal_columns, numeric_columns


def numeric_subset(train, test, meta, categorical_columns, ordinal_columns):
    for idx, v in enumerate(meta['columns']):
        if v['name'] == 'label':
            break

    cat_col = categorical_columns + ordinal_columns
    cat_col.remove(idx)
    train = np.delete(train, cat_col, axis=1)
    test = np.delete(test, cat_col, axis=1)

    categorical_columns = []
    ordinal_columns = []

    for idx in reversed(sorted(cat_col)):
        del meta['columns'][idx]

    assert len(meta['columns']) == train.shape[1]
    return train, test, meta, categorical_columns, ordinal_columns


def categorical_subset(train, test, meta, categorical_columns, ordinal_columns):
    for idx, v in enumerate(meta['columns']):
        if v['name'] == 'label':
            break

    num_col = set(range(train.shape[1]))
    num_col.remove(idx)
    num_col -= set(categorical_columns + ordinal_columns)
    num_col = list(num_col)
    train = np.delete(train, num_col, axis=1)
    test = np.delete(test, num_col, axis=1)
    categorical_columns = [v - sum([1 for x in num_col if x < v]) for v in categorical_columns]
    ordinal_columns = [v - sum([1 for x in num_col if x < v]) for v in ordinal_columns]

    for idx in reversed(sorted(num_col)):
        del meta['columns'][idx]

    assert len(meta['columns']) == train.shape[1]
    return train, test, meta, categorical_columns, ordinal_columns


def parse_name(name):
    distinct_threshold = -1
    type_subset = "all"
    zero_code = 1.

    parts = name.split("_")
    if parts[-1].startswith("zc"):
        zero_code = float(parts.pop()[2:])

    if parts[-1].startswith("u"):
        distinct_threshold = int(parts.pop()[1:])

    if parts[-1] in ["categorical", "numeric"]:
        type_subset = parts.pop()

    name = "_".join(parts)
    return name, type_subset, distinct_threshold, zero_code


def get_zero_code_columns(data, threshold, num_cols):
    df = pd.DataFrame(data)
    cols = []
    for col in df.columns:
        if col not in num_cols:
            continue
        counts = df[col].value_counts()

        total = counts.sum()
        x = (counts.head(1).index.values.tolist()[0] == 0.0) and (counts.head(1).sum() / total) > threshold
        if x:
            cols.append(col)
    return cols


def load_dataset(name: str, benchmark=False):
    name, type_subset, distinct_threshold, zero_code = parse_name(name)

    LOGGER.info('Loading dataset %s (%s variables, %d distinct values threshold)', name, type_subset, distinct_threshold)
    data = _load_file(name + '.npz', np.load)
    meta = _load_file(name + '.json', _load_json)

    train = data['train']
    test = data['test']
    full = np.concatenate([train, test])

    categorical_columns, ordinal_columns, numeric_columns = _get_columns(meta, full, distinct_threshold)
    if zero_code:
        vars_to_code = get_zero_code_columns(full, zero_code, numeric_columns)
        LOGGER.info(f"Columns to zero-code: {vars_to_code}")

    if type_subset != "all":
        if type_subset == 'numeric':
            train, test, meta, categorical_columns, ordinal_columns = numeric_subset(train, test, meta, categorical_columns, ordinal_columns)
        elif type_subset == 'categorical':
            train, test, meta, categorical_columns, ordinal_columns = categorical_subset(train, test, meta, categorical_columns, ordinal_columns)
        else:
            raise ValueError("type_subset should be 'numeric' or 'categorical'")

    if benchmark:
        return train, test, meta, categorical_columns, ordinal_columns

    return train, categorical_columns, ordinal_columns


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dataset("census_numeric_u50_zc0.4")
    load_dataset("census_u50_zc0.4")
    load_dataset("census_zc0.4")
    load_dataset("adult_numeric_u50_zc0.4")
    load_dataset("covtype_numeric_u50_zc0.4")
    load_dataset("covtype_u50")
