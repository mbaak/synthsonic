"""
Script to compare variations of Synthsonic. Used for the paper's ablation study / comparing extensions across datasets.

"""
import multiprocessing
import time
from copy import deepcopy
from collections import defaultdict

from pgmpy.estimators import BayesianEstimator
from sklearn.metrics import normalized_mutual_info_score
from xgboost import XGBClassifier
import logging
from sdgym import run
from sdgym.data import load_dataset
from sdgym.synthesizers.base import BaseSynthesize
import numpy as np
from phik.phik import phik_from_binned_array

from synthsonic.models.kde_copula_nn_pdf import KDECopulaNNPdf


logging.basicConfig(level=logging.INFO)

settings = {
    'default': {
        'pdf_args': {
            'test_size': 0.35,
            'edge_weights_fn': "mutual_info",
            'n_calibration_bins': 100,
            'n_uniform_bins': 30,
            'distinct_threshold': -1,
            'do_PCA': True,
            'estimator_type': 'chow-liu',
            'apply_calibration': True,
            "isotonic_increasing": "auto",
            'verbose': True,
        },
        'sample_args': {
            'mode': 'cheap',
        },
        'target_args': {
            'tan_target': -1,
            'conditional_target': -1,
        },
        'random_state': 0,
        # FIXME: auto detect... visions!
        'round': True,
    },
    # ============================================
    # Gaussian Sim.
    # ============================================
    'ring': {
        'pdf_args': {
            'test_size': 0.25,
            'edge_weights_fn': phik_from_binned_array,
            'n_uniform_bins': 50,
        },
        'round': False,
    },
    'grid': {
        'pdf_args': {
            'n_uniform_bins': 50,
        },
        'round': False,
    },
    'gridr': {
        'pdf_args': {
            'n_uniform_bins': 50,
        },
        'round': False,
    },
    # ============================================
    # Bayesian Network Sim.
    # ============================================
    'asia': {
        'pdf_args': {
            'bm_fit_args': dict(estimator=BayesianEstimator, prior_type='dirichlet', pseudo_counts=0.1),
        }
    },
    'child': {
        'pdf_args': {
            'bm_fit_args': dict(estimator=BayesianEstimator, prior_type='dirichlet', pseudo_counts=0.1),
        }
    },
    'insurance': {
        'pdf_args': {
            'bm_fit_args': dict(estimator=BayesianEstimator, prior_type='dirichlet', pseudo_counts=0.1),
        }
    },
    'alarm': {
        'pdf_args': {
            'bm_fit_args': dict(estimator=BayesianEstimator, prior_type='dirichlet', pseudo_counts=0.1),
        }
    },
    # ============================================
    # Real
    # ============================================
    'census': {
        
    },
    'credit': {

    },
    'adult': {
        'pdf_args': {
            'test_size': 0.25,
            'edge_weights_fn': normalized_mutual_info_score,
        },
    },
    'intrusion': {
        
    },
    'covtype': {
        'pdf_args':{
            'n_uniform_bins': 40,
            'test_size': 0.25,
        },
    },
    'news': {

    }
}

choices = [
    'no_pca',
    'no_pca_mi',
    'no_clf',
    'no_calibration',
    'no_tuning',
    'all',
    'auto-tree-cl',
    'auto-tree-tan',
    'auto-bins-uniform',
    'auto-bins-calibration',
    'oracle-tree-class',
]
dsets_bn = ['child', 'asia', 'insurance', 'alarm']
dsets_gm = ['ring', 'gridr', 'grid']
dsets_rl = ['covtype', 'intrusion', 'adult', 'credit', 'news', 'census']
dsets_all = dsets_gm + dsets_bn + dsets_rl

dsets = dsets_all


def get_label_col(dataset_name):
    _, _, meta, _, _ = load_dataset(dataset_name, benchmark=True)
    for idx, c in enumerate(meta['columns']):
        if c['name'] == 'label':
            return idx
    return -1


def factory(my_sets, my_ablation):
    class BaseClass(BaseSynthesizer):
        def __init__(self, iterations):
            self.random_state = my_sets['random_state'] + iterations * 1000

        def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
            self.categorical_columns = categorical_columns
            self.ordinal_columns = ordinal_columns
            xdata = np.float64(data)

            kde = KDECopulaNNPdf(
                use_KDE=False,
                categorical_columns=categorical_columns + ordinal_columns,
                random_state=self.random_state,
                **my_sets['pdf_args']
            )
            self.kde = kde.fit(
                xdata,
            )

        def sample(self, samples):
            X_gen = self.kde.sample_no_weights(
                samples,
                random_state=self.random_state + 10,
                **my_sets['sample_args']
            )

            if len(self.categorical_columns) + len(self.ordinal_columns) > 0:
                X_gen[:, self.categorical_columns + self.ordinal_columns] = np.round(
                    X_gen[:, self.categorical_columns + self.ordinal_columns]
                )

            # FIXME: use visions/any logic to automate this... should be the user (sdgym 3 provides this...)
            X_gen = np.float32(X_gen)
            if settings[dataset_name].get('round') and dataset_name not in ['grid', 'gridr', 'ring']:
                X_gen = np.round(X_gen)
            return X_gen

    class NewClass(BaseClass): pass
    NewClass.__name__ = f"Synthsonic[{my_ablation}]"
    return deepcopy(NewClass)


for dataset_name in dsets:
    all_synthesizers = []
    for start_seed, ablation in enumerate(choices, start=1337):
        current_settings = defaultdict(dict)
        current_settings.update(deepcopy(settings['default']))
        if ablation != 'no_tuning':
            current_settings.update(deepcopy(settings[dataset_name]))
            for key, nested_dict in deepcopy(settings[dataset_name]).items():
                if isinstance(nested_dict, dict):
                    current_settings[key].update(deepcopy(nested_dict))
        current_settings = dict(current_settings)
        current_settings['pdf_args']['verbose'] = True

        current_settings['random_state'] = start_seed

        if ablation == 'no_pca':
            current_settings['pdf_args']['do_PCA'] = False
            current_settings['pdf_args']['ordering'] = ''
        if ablation == 'no_pca_mi':
            current_settings['pdf_args']['do_PCA'] = False
            current_settings['pdf_args']['ordering'] = 'phik'
        if ablation == 'no_calibration':
            current_settings['pdf_args']['apply_calibration'] = False
        if ablation == 'no_clf':
            current_settings['pdf_args']['clf'] = None
        else:
            current_settings['pdf_args']['clf'] = XGBClassifier(
                n_estimators=250,
                reg_lambda=1,
                gamma=0,
                max_depth=9,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=start_seed,
            )
        if ablation == 'auto-bins-uniform':
            current_settings['pdf_args']['n_uniform_bins'] = "auto"
        if ablation == 'auto-bins-calibration':
            current_settings['pdf_args']['n_calibration_bins'] = "auto"
        if ablation == 'auto-tree-cl':
            current_settings['pdf_args']['estimator_type'] = "auto-cl"
        if ablation == 'auto-tree-tan':
            current_settings['pdf_args']['estimator_type'] = "auto-tan"
        if ablation == 'oracle-tree-class':
            col = get_label_col(dataset_name)
            if col != -1:
                current_settings['pdf_args']['estimator_type'] = 'tan'
                current_settings['pdf_args']['class_node'] = col
            else:
                continue

        print(f'ablation={ablation}, dataset_name={dataset_name}, current settings={current_settings}, seed={start_seed}')

        all_synthesizers.append(factory(deepcopy(current_settings), ablation))

    try:
        datasets = [dataset_name]
        scores = run(
            synthesizers=all_synthesizers,
            datasets=datasets,
            iterations=3,
            add_leaderboard=False,
            workers=int(multiprocessing.cpu_count() / 2)
        )
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        scores.to_csv(f"ablation/scores_{dataset_name}_{time_str}.csv")

        df = scores.copy(deep=True)
        total = df.loc['Synthsonic[all]'].copy()
        df = df - total.values.squeeze()
        df.loc['Synthsonic[all]'] = total
        df.to_csv(f"ablation/diff_scores_{dataset_name}_{time_str}.csv")
    except ValueError:
        print(f"Failed to compute {dataset_name} scores")
