from pathlib import Path

import numpy as np
import pandas as pd
from sdgym.results import make_leaderboard

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
    total = df.loc[f"Synthsonic[all]"].copy()
    df = df - total.values.squeeze()
    df.loc[f"Synthsonic[all]"] = total

    df.to_csv(output)

    results = {
        key: df.at[
            f"Synthsonic[{key}]", f"{dataset_name}/{dataset_to_metric[dataset_name]}"
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
    files = Path(f"results/2_ablation_analysis/{dataset}/raw/").glob(
        r"Synthsonic*all*_*"
    )
    scores = [
        pd.read_csv(file_name)[dataset_to_metric[dataset]].mean() for file_name in files
    ]
    my_mean = np.mean(scores)
    mad = np.mean(np.abs(scores - np.mean(scores)))
    return my_mean, mad


def redo_scores():
    dirs = Path(__file__).parent.glob("results/2_ablation_analysis/*")
    rows = {}
    for dir_name in dirs:

        if dir_name.is_dir() and dir_name.stem in dataset_to_metric:
            make_leaderboard(
                str(dir_name / "raw/"),
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

    table_head = r"""\begin{table}[H]
\vspace{-3em}
\caption{Ablation study results ($\mathcal{L}_{test}$ for simulations, $F_1$ for real datasets - higher is better). The right column shows the absolute performance of our model, where the other columns contain the performance change relative to that. Where the entry differences exceed two times the MAD are marked in bold.}
\begin{tabular}{llllll}
\hline
\diagbox{Dataset}{Model}        & \texttt{w/o PCA}               &  \texttt{w/o clf}        & \texttt{w/o calibration}       & \texttt{w/o tuning}         & \texttt{Synthsonic}                      \\ 
\hline"""

    table_foot = r"""\end{tabular}   
    \label{table:ablation}
\end{table}"""

    table = table_head
    for group in dataset_groups:
        for ds in group:
            table += rows[ds] + "\n"
        table += "\hline\n"
    table += table_foot
    return table


tex = redo_scores()
Path("output/table3.tex").write_text(tex)
