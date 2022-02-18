import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sdgym.data import load_dataset
from sdgym.results import make_leaderboard
from sdgym.synthesizers.base import BaseSynthesizer
from synthsonic.models.kde_copula_nn_pdf import KDECopulaNNPdf
from xgboost import XGBClassifier

dataset_to_metric = {
    "intrusion": "macro_f1",
    "ring": "test_likelihood",
    "grid": "test_likelihood",
    "gridr": "test_likelihood",
    "child": "test_likelihood",
    "alarm": "test_likelihood",
    "asia": "test_likelihood",
    "insurance": "test_likelihood",
    "covtype": "macro_f1",
    "adult": "f1",
    "credit": "f1",
    "census": "f1",
    "news": "r2",
}

dataset_groups = [
    ("grid", "gridr", "ring"),
    ("asia", "alarm", "child", "insurance"),
    ("adult", "census", "credit", "covtype", "intrusion", "news"),
]


def fmt(v):
    if v < 0:
        pref = "-"
        v = -1 * v
    else:
        pref = "\D"

    return f"${pref}{v:.2f}$"


def make_diff_leaderboard(scores, output, dataset_name, mad):
    df = pd.read_csv(scores, index_col=0)
    total = df.loc[f"Synthsonic[all][{dataset_name}]"].copy()
    df = df - total.values.squeeze()
    df.loc[f"Synthsonic[all][{dataset_name}]"] = total

    df.to_csv(output)

    results = {
        key: df.at[
            f"Synthsonic[{key}][{dataset_name}]",
            f"{dataset_name}/{dataset_to_metric[dataset_name]}",
        ]
        for key in ["all", "no_pca", "no_tuning", "no_clf", "no_calibration"]
    }

    res = fmt(results["all"]) + " \pm " + fmt(mad)
    dsn = f"{{\\texttt{{{dataset_name.capitalize()}}}}}"
    row = (
        f"\multicolumn{{1}}{{l|}}{dsn: <25}&"
        f"{fmt(results['no_pca'])}\t\t&"
        f"{fmt(results['no_clf'])}\t\t&"
        f"{fmt(results['no_calibration'])}\t\t&"
        f"{fmt(results['no_tuning'])}\t\t&"
        f"{res: <15}\\\\"
    )
    return row


def compute_mad(dataset):
    files = Path(f"ablation/{dataset}/raw/").glob(r"Synthsonic*all*_*")
    scores = [
        pd.read_csv(file_name)[dataset_to_metric[dataset]].mean() for file_name in files
    ]
    my_mean = np.mean(scores)
    mad = np.mean(np.abs(scores - np.mean(scores)))
    return my_mean, mad


def redo_scores():
    dirs = Path(__file__).parent.glob("ablation/*")
    rows = {}
    for dir_name in dirs:

        if dir_name.is_dir() and dir_name.stem in dataset_to_metric:
            make_leaderboard(
                str(dir_name / "raw"),
                add_leaderboard=False,
                output_path=str(dir_name / "scores.csv"),
            )
            dsname = dir_name.stem
            my_mean, my_mad = compute_mad(dsname)
            rows[dsname] = make_diff_leaderboard(
                str(dir_name / "scores.csv"),
                str(dir_name / "scores_diff.csv"),
                dsname,
                my_mad,
            )

    for group in dataset_groups:
        for ds in group:
            print(rows[ds])
        print("\hline")


logging.basicConfig(level=logging.INFO)


def get_settings():
    settings = {
        "default": {
            "pdf_args": {
                "test_size": 0.35,
                "edge_weights_fn": "normalized_mutual_info",
                "n_calibration_bins": 100,
                "n_uniform_bins": 30,
                "distinct_threshold": -1,
                "estimator_type": "tan",
                "class_node": None,
                "isotonic_increasing": "auto",
                "do_PCA": True,
                "apply_calibration": True,
                "verbose": True,
            },
            "sample_args": {
                "mode": "cheap",
            },
            "random_state": 0,
        },
        # ============================================
        # Gaussian Sim.
        # ============================================
        "ring": {
            "pdf_args": {
                "test_size": 0.25,
                "n_uniform_bins": 50,
            },
        },
        "grid": {
            "pdf_args": {
                "n_uniform_bins": 50,
            },
        },
        "gridr": {
            "pdf_args": {
                "n_uniform_bins": 50,
            },
        },
        # ============================================
        # Bayesian Network Sim.
        # ============================================
        "asia": {
            "pdf_args": {}
        },
        "child": {
            "pdf_args": {}
        },
        "insurance": {
            "pdf_args": {}
        },
        "alarm": {
            "pdf_args": {}
        },
        # ============================================
        # Real
        # ============================================
        "census": {
            "pdf_args": {},
        },
        "credit": {
            "pdf_args": {
                "n_uniform_bins": 25,
            },
        },
        "adult": {
            "pdf_args": {
                "test_size": 0.25,
            },
        },
        "intrusion": {
            "pdf_args": {},
        },
        "covtype": {
            "pdf_args": {
                "n_uniform_bins": 40,
                "test_size": 0.25,
            },
        },
        "news": {
            "pdf_args": {
                "clf": None,
            }
        },
    }
    return settings


def get_label_col(dataset_name):
    _, _, meta, _, _ = load_dataset(dataset_name, benchmark=True)
    for idx, c in enumerate(meta["columns"]):
        if c["name"] == "label":
            return idx
    return -1


def integer_columns(a):
    res = a == np.round(a)
    res = np.sum(res, axis=0)
    cols = np.argwhere(res == a.shape[0]).flatten().tolist()
    return cols


def factory(my_sets, my_ablation, dataset):
    class BaseClass(BaseSynthesizer):
        def __init__(self, iterations=None):
            self.random_state = my_sets["random_state"] + iterations * 1000

        def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
            self.categorical_columns = categorical_columns
            self.ordinal_columns = ordinal_columns
            self.numeric_columns = list(
                set(range(data.shape[1])) - set(categorical_columns + ordinal_columns)
            )
            self.integer_columns = list(
                set(integer_columns(data)) - set(categorical_columns + ordinal_columns)
            )
            xdata = np.float64(data)

            kde = KDECopulaNNPdf(
                use_KDE=False,
                categorical_columns=categorical_columns + ordinal_columns,
                random_state=self.random_state,
                **my_sets["pdf_args"],
            )
            self.kde = kde.fit(
                xdata,
            )

        def sample(self, samples):
            X_gen = self.kde.sample_no_weights(
                samples, random_state=self.random_state + 10, **my_sets["sample_args"]
            )

            if len(self.categorical_columns) > 0:
                X_gen[:, self.categorical_columns] = np.round(
                    X_gen[:, self.categorical_columns]
                )
            if len(self.ordinal_columns) > 0:
                X_gen[:, self.ordinal_columns] = np.round(
                    X_gen[:, self.ordinal_columns]
                )
            if len(self.integer_columns) > 0:
                X_gen[:, self.integer_columns] = np.round(
                    X_gen[:, self.integer_columns]
                )

            X_gen = np.float32(X_gen)

            return X_gen

    class NewClass(BaseClass):
        pass

    NewClass.__name__ = f"Synthsonic[{my_ablation}]"
    return deepcopy(NewClass)


def get_synthsonic(dataset, start_seed=0, synth_settings=None, ablation="all"):
    if synth_settings is None:
        synth_settings = get_synthsonic_options(ablation, dataset, start_seed)
    return factory(deepcopy(synth_settings), ablation, dataset)


def get_synthsonic_options(ablation, dataset_name, start_seed):
    settings = get_settings()
    current_settings = defaultdict(dict)
    current_settings.update(deepcopy(settings["default"]))
    if ablation != "no_tuning":
        for key, nested_dict in deepcopy(settings[dataset_name]).items():
            if isinstance(nested_dict, dict):
                current_settings[key].update(deepcopy(nested_dict))
    current_settings = dict(current_settings)
    current_settings["pdf_args"]["verbose"] = True

    current_settings["random_state"] = start_seed

    if ablation == "no_pca":
        current_settings["pdf_args"]["do_PCA"] = False
        current_settings["pdf_args"]["ordering"] = ""
    if ablation == "no_pca_mi":
        current_settings["pdf_args"]["do_PCA"] = False
        current_settings["pdf_args"]["ordering"] = "phik"
    if ablation == "no_calibration":
        current_settings["pdf_args"]["apply_calibration"] = False
    if ablation == "no_clf":
        current_settings["pdf_args"]["clf"] = None

    if "clf" not in current_settings["pdf_args"]:
        current_settings["pdf_args"]["clf"] = XGBClassifier(
            n_estimators=250,
            reg_lambda=1,
            gamma=0,
            max_depth=9,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=start_seed,
        )

    if ablation == "auto-bins-uniform":
        current_settings["pdf_args"]["n_uniform_bins"] = "auto"
    if ablation == "knuth-bins-uniform":
        current_settings["pdf_args"]["n_uniform_bins"] = "knuth"
    if ablation == "blocks-bins-uniform":
        current_settings["pdf_args"]["n_uniform_bins"] = "blocks"
    if ablation == "auto-bins-calibration":
        current_settings["pdf_args"]["n_calibration_bins"] = "auto"
    if ablation == "auto-tree-cl":
        current_settings["pdf_args"]["estimator_type"] = "cl"
        current_settings["pdf_args"]["root_node"] = None
    if ablation == "auto-tree-tan":
        current_settings["pdf_args"]["estimator_type"] = "tan"
        current_settings["pdf_args"]["class_node"] = None
    if ablation == "oracle-tree-class":
        col = get_label_col(dataset_name)
        if col != -1:
            current_settings["pdf_args"]["estimator_type"] = "tan"
            current_settings["pdf_args"]["class_node"] = col
        else:
            raise ValueError("has no label column")

    return current_settings
