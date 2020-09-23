from io import StringIO
from typing import List
import numpy as np

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

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

    classifier, _ht_false, _ht_true = get_classifier(_classifier)

    return KDECopulaNNPdf(**kwargs, rho=0.5, clf=classifier(**_ht_false)), classifier, _ht_false, _ht_true


def get_classifier(classifier):
    # _hyper_tune_true params will be passed to grid search (if selected) or used as-is if not using grid search
    # _hyper_tune_false params are not subject of hypertuning
    if classifier == 'MLP':
        _hyper_tune_false = dict(random_state=0, max_iter=500, early_stopping=True)
        _hyper_tune_true = dict(clf__alpha=10.0 ** -np.arange(1, 3))
        return MLPClassifier, _hyper_tune_false, _hyper_tune_true
    elif classifier == 'XGBoost':
        _hyper_tune_false = dict()
        _hyper_tune_true = dict(clf__max_depth=[3, 6])
        return XGBClassifier, _hyper_tune_false, _hyper_tune_true


def get_data(**kwargs) -> bytes:
    input_df = _bytes_to_df(kwargs.get('bytes_data'))

    input_array = input_df.to_numpy()

    categorical_column_positions = _find_col_postitions(input_df, kwargs.get('categorical_columns', []))
    ordinal_column_positions = _find_col_postitions(input_df, kwargs.get('ordinal_columns', []))

    data = input_array

    n_features = data.shape[1]

    # @todo use that if possible
    # x_min, x_max = set_min_max(data, n_features)

    use_grid_search = kwargs.get('use_grid_search', False)
    model_args = dict(x_max=None, x_min=None, use_KDE=False)

    _clf_params = {}

    # clf_params_base = dict(random_state=0, max_iter=500, early_stopping=True)
    # clf_params_before_tuning = dict(clf__alpha=10.0 ** -np.arange(1, 3))
    #
    # clf_params.update(clf_params_base)
    #
    # # MLP Classifier  dict(clf__alpha=10.0 ** -np.arange(1, 3))
    # # XGBoost

    _model, _classifier, _clf_ht_false, _clf_ht_true = get_model(**dict(classifier=kwargs.get('classifier')))

    if use_grid_search:
        grid = GridSearchCV(_model, _clf_ht_true, cv=5)
        grid.fit(data)

        clf_params_tuned = {k.replace('clf__', ''): v for k, v in grid.best_params_.items()}

        _clf_params.update(clf_params_tuned)

    _clf_params.update(_clf_ht_false)
    clf_params = {k.replace('clf__', ''): v for k, v in _clf_params.items()}

    print(clf_params)

    _model = KDECopulaNNPdf(**model_args, rho=0.5, clf=_classifier(**clf_params))

    model = _model.fit(data)

    sample_data = model.sample_no_weights(kwargs.get('rows', 5), mode='expensive')

    df = pd.DataFrame(sample_data, columns=input_df.columns)

    return _df_to_bytes(df)
