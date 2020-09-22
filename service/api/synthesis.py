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


def get_model(x_min, x_max):
    return KDECopulaNNPdf(x_min=x_min, x_max=x_max, rho=0.5,
                          clf=MLPClassifier(random_state=0, max_iter=500, early_stopping=True))


def get_data(bytes_data: bytes, categorical_columns: List[str] = [], ordinal_columns: List[str] = [],
             rows: int = 5) -> bytes:
    input_df = _bytes_to_df(bytes_data)

    input_array = input_df.to_numpy()

    categorical_column_positions = _find_col_postitions(input_df, categorical_columns)
    ordinal_column_positions = _find_col_postitions(input_df, ordinal_columns)

    data = np.float64(input_array)

    n_features = data.shape[1]

    x_min, x_max = set_min_max(data, n_features)

    _model = get_model(None, None)
    model = _model.fit(data)

    sample_data = model.sample_no_weights(rows, mode='expensive')

    df = pd.DataFrame(sample_data, columns=input_df.columns)

    return _df_to_bytes(df)
