from functools import partial
from pathlib import Path

import pandas as pd
from sdgym.results import make_leaderboard


def get_columns(metric, datasets, columns):
    return [
        col
        for col in columns
        if (
            (isinstance(metric, str) and col.endswith(f"/{metric}"))
            or any(col.endswith(f"/{m}") for m in metric)
        )
        and any(col.startswith(f"{d}/") for d in datasets)
    ]


def table_agg(leaderboard, scores):
    leaderboard = pd.concat([leaderboard, scores])
    fn = partial(get_columns, columns=leaderboard.columns)
    leaderboard["gmm_syn"] = leaderboard[
        fn("syn_likelihood", ["grid", "gridr", "ring"])
    ].mean(axis=1)
    leaderboard["gmm_test"] = leaderboard[
        fn("test_likelihood", ["grid", "gridr", "ring"])
    ].mean(axis=1)
    leaderboard["bn_syn"] = leaderboard[
        fn("syn_likelihood", ["asia", "alarm", "child", "insurance"])
    ].mean(axis=1)
    leaderboard["bn_test"] = leaderboard[
        fn("test_likelihood", ["asia", "alarm", "child", "insurance"])
    ].mean(axis=1)
    leaderboard["real_f1"] = leaderboard[
        fn(["f1", "macro_f1"], ["credit", "census", "covtype", "adult", "intrusion"])
    ].mean(axis=1)
    leaderboard["real_r2"] = leaderboard[fn("r2", ["news"])].mean(axis=1)

    table_head = r"""\begin{table}[tbh!]
      \caption{Performance of Synthsonic on artificial datasets, compared against top-performers on SDGym leaderboard v0.2.2. The top scores are printed in bold.}
      \label{tbl:agg-scores}
      \centering
      \begin{tabular}{lllllll}
        \toprule
                                            & \multicolumn{2}{c}{GM. Sim}   &   \multicolumn{2}{c}{BN. Sim} & \multicolumn{2}{c}{Real}      \\
        \cmidrule(r){2-3} \cmidrule(r){4-5}\cmidrule(r){6-7}
        Method                              & $\mathcal{L}_{syn}$     & $\mathcal{L}_{test}$    & $\mathcal{L}_{syn}$     & $\mathcal{L}_{test}$    & clf $F_1$           & reg $R^2$           \\
        \midrule
"""

    table_rows = ""
    idx = "IdentitySynthesizer"
    name = "Identity"
    s1 = leaderboard.at[idx, "gmm_syn"]
    t1 = leaderboard.at[idx, "gmm_test"]
    s2 = leaderboard.at[idx, "bn_syn"]
    t2 = leaderboard.at[idx, "bn_test"]
    s3 = leaderboard.at[idx, "real_f1"]
    t3 = leaderboard.at[idx, "real_r2"]
    table_rows += f"{name} & ${s1:.02f}$ & ${t1:.02f}$ & ${s2:.02f}$ & ${t2:.02f}$ & ${s3:.02f}$ & ${t3:.02f}$ \\\\ \n"
    table_rows += "\midrule\n"
    for name in ["CLBN", "PrivBN", "TVAE", "CTGAN"]:
        idx = name
        if name != "CTGAN":
            idx += "Synthesizer"
        s1 = leaderboard.at[idx, "gmm_syn"]
        t1 = leaderboard.at[idx, "gmm_test"]
        s2 = leaderboard.at[idx, "bn_syn"]
        t2 = leaderboard.at[idx, "bn_test"]
        s3 = leaderboard.at[idx, "real_f1"]
        t3 = leaderboard.at[idx, "real_r2"]
        table_rows += f"{name} & ${s1:.02f}$ & ${t1:.02f}$ & ${s2:.02f}$ & ${t2:.02f}$ & ${s3:.02f}$ & ${t3:.02f}$ \\\\ \n"
    table_rows += "\\bottomrule\n"
    idx = "Synthsonic[all]"
    name = "Synthsonic"
    s1 = leaderboard.at[idx, "gmm_syn"]
    t1 = leaderboard.at[idx, "gmm_test"]
    s2 = leaderboard.at[idx, "bn_syn"]
    t2 = leaderboard.at[idx, "bn_test"]
    s3 = leaderboard.at[idx, "real_f1"]
    t3 = leaderboard.at[idx, "real_r2"]
    table_rows += f"{name} & ${s1:.02f}$ & ${t1:.02f}$ & ${s2:.02f}$ & ${t2:.02f}$ & ${s3:.02f}$ & ${t3:.02f}$ \\\\ \n"

    table_foot = (
        r"""\end{tabular}
      \vspace{-5em}
    \end{table}
    """
        ""
    )

    return table_head + table_rows + table_foot


leaderboard = pd.read_csv("leaderboard.csv", index_col=0)
synth_scores = make_leaderboard(
    "results/1_leaderboard/",
    add_leaderboard=False,
)
tex = table_agg(leaderboard, synth_scores)
Path("output/table1.tex").write_text(tex)
