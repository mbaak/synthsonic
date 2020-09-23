from io import StringIO
from typing import List

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from synthsonic.models.kde_copula_nn_pdf import KDECopulaNNPdf


def set_min_max(data, nf):
    x_min = [0] * nf
    x_max = [1.] * nf
    for i in range(0, nf):
        x_min[i] = data[:, i].min()
        x_max[i] = data[:, i].max()
    return x_min, x_max


def _bytes_to_df(data_bytes: bytes) -> pd.DataFrame:
    data = StringIO(str(data_bytes, 'utf-8'))

    return pd.read_csv(data)


def _df_to_bytes(df: pd.DataFrame) -> bytes:
    _result = StringIO()

    df.to_csv(_result, mode='w', encoding='UTF_8', index=False)

    result = _result.getvalue().encode('utf-8')

    return result


def _find_col_postitions(df: pd.DataFrame, columns: List[str]) -> List[int]:
    positions = []

    for c in columns:
        try:
            position = df.columns.get_loc(c)
            positions.append(position)
        except:
            pass

    return positions


def get_model(**kwargs):
    _classifier = kwargs.pop('classifier')

    return KDECopulaNNPdf(**kwargs, rho=0.5, clf=get_classifier(_classifier))


def get_classifier(classifier):
    if classifier == 'MLPClassifier':
        return MLPClassifier(random_state=0, max_iter=500, early_stopping=True)
    # elif classifier == 'XGBoost':
    #     return XGBoost()
    else:
        return MLPClassifier(random_state=0, max_iter=500, early_stopping=True)


def get_data(**kwargs) -> bytes:
    input_df = _bytes_to_df(kwargs.get('bytes_data'))

    input_array = input_df.to_numpy()

    categorical_column_positions = _find_col_postitions(input_df, kwargs.get('categorical_columns', []))
    ordinal_column_positions = _find_col_postitions(input_df, kwargs.get('ordinal_columns', []))

    data = input_array

    n_features = data.shape[1]

    # @todo use that if possible
    # x_min, x_max = set_min_max(data, n_features)

    # @todo remove x_*
    kde_smoothing = kwargs.get('kde_smoothing', False)
    model_args = dict(x_max=None, x_min=None, use_KDE=kde_smoothing,
                      classifier=kwargs.get('classifier', 'MLPClassifier'))

    _model = get_model(**model_args)
    model = _model.fit(data)

    sample_data = model.sample_no_weights(kwargs.get('rows', 5), mode='expensive')

    df = pd.DataFrame(sample_data, columns=input_df.columns)

    return _df_to_bytes(df)
