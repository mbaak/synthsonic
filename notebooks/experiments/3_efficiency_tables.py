from pathlib import Path

import numpy as np
import pandas as pd
from sdgym.results import make_leaderboard


def fmt(v):
    if np.isnan(v):
        return f"\sout{{0.0}}"
    rounded = round(v)
    if rounded == 0:
        return "$< 1$"

    diff = 3 - len(str(rounded))
    pref = ""
    if diff > 0:
        pref = "\Z" * diff

    return f"${pref}{rounded}$"


def row_gmm(row):
    if row["synthesizer"] == "PrivBN":
        row = (
            f"\t{row['synthesizer']: <10}&"
            f"\multicolumn{{2}}{{l}}{{{fmt(row['grid/fit_time']+row['grid/sample_time'])}}}&"
            f"\multicolumn{{2}}{{l}}{{{fmt(row['gridr/fit_time']+row['gridr/sample_time'])}}}&"
            f"\multicolumn{{2}}{{l}}{{{fmt(row['ring/fit_time']+row['ring/sample_time'])}}}"
            f"\\\\\n"
        )
    else:
        row = (
            f"\t{row['synthesizer']: <10}&"
            f"{fmt(row['grid/fit_time'])}&"
            f"{fmt(row['grid/sample_time'])}&"
            f"{fmt(row['gridr/fit_time'])}&"
            f"{fmt(row['gridr/sample_time'])}&"
            f"{fmt(row['ring/fit_time'])}&"
            f"{fmt(row['ring/sample_time'])}"
            f"\\\\\n"
        )
    return row


def table_gmm():
    table_head = r"""\begin{table}[H]
    \caption{Computational efficiency of algorithms on GMM datasets (seconds)}
    \label{tbl:efficiency-full-gmm}
    \centering
    \begin{tabular}{lllllll}
    \toprule
                & \MC{grid}                         & \MC{gridr}                            & \MC{ring}                     \\
                \cmidrule(r){2-3}                   \cmidrule(r){4-5}                       \cmidrule(r){6-7}
    Method       & fit              & sample        & fit               & sample            & fit               & sample    \\
    \midrule    
"""
    times = make_leaderboard(
        "results/3_efficiency_cpu_gmm/",
        add_leaderboard=False,
    )
    times = times[
        [
            "grid/fit_time",
            "grid/sample_time",
            "gridr/fit_time",
            "gridr/sample_time",
            "ring/fit_time",
            "ring/sample_time",
        ]
    ]
    times.reset_index(level=0, inplace=True)
    times["synthesizer"] = times["synthesizer"].str.replace("Synthesizer", "")
    times["synthesizer"] = times["synthesizer"].str.replace("\[all\]", "")
    table_rows = ""
    rows = {row["synthesizer"]: row for _, row in times.iterrows()}
    table_rows += row_gmm(rows["CLBN"])
    table_rows += row_gmm(rows["PrivBN"])
    table_rows += row_gmm(rows["TVAE"])
    table_rows += row_gmm(rows["CTGAN"])
    table_rows += "\\bottomrule\n"
    table_rows += row_gmm(rows["Synthsonic"])
    table_foot = r"""\end{tabular}
\end{table}"""
    return times, table_head + table_rows + table_foot


def row_bn(row):
    if row["synthesizer"] == "PrivBN":
        row = (
            f"\t{row['synthesizer']: <10}&"
            f"\multicolumn{{2}}{{l}}{{{fmt(row['asia/fit_time']+row['asia/sample_time'])}}}&"
            f"\multicolumn{{2}}{{l}}{{{fmt(row['alarm/fit_time']+row['alarm/sample_time'])}}}&"
            f"\multicolumn{{2}}{{l}}{{{fmt(row['child/fit_time']+row['child/sample_time'])}}}&"
            f"\multicolumn{{2}}{{l}}{{{fmt(row['insurance/fit_time']+row['insurance/sample_time'])}}}"
            f"\\\\\n"
        )
    else:
        row = (
            f"\t{row['synthesizer']: <10}&"
            f"{fmt(row['asia/fit_time'])}&"
            f"{fmt(row['asia/sample_time'])}&"
            f"{fmt(row['alarm/fit_time'])}&"
            f"{fmt(row['alarm/sample_time'])}&"
            f"{fmt(row['child/fit_time'])}&"
            f"{fmt(row['child/sample_time'])}&"
            f"{fmt(row['insurance/fit_time'])}&"
            f"{fmt(row['insurance/sample_time'])}"
            f"\\\\\n"
        )
    return row


def table_bn():
    table_head = r"""\begin{table}[H]
    \caption{Computational efficiency of algorithms on Bayesian Network datasets (seconds)}
    \label{tbl:efficiency-full-bn}
    \centering
    \begin{tabular}{lllllllll}
    \toprule
                & \MC{asia}                         & \MC{alarm}                & \MC{child}                & \MC{insurance}            \\
                \cmidrule(r){2-3}                   \cmidrule(r){4-5}           \cmidrule(r){6-7}           \cmidrule(r){8-9}
    Method      & fit               & sample        & fit           & sample    & fit           & sample    & fit           & sample    \\
    \midrule    
"""
    times = make_leaderboard(
        "results/3_efficiency_cpu_bn/",
        add_leaderboard=False,
    )
    times = times[
        [
            "asia/fit_time",
            "asia/sample_time",
            "alarm/fit_time",
            "alarm/sample_time",
            "child/fit_time",
            "child/sample_time",
            "insurance/fit_time",
            "insurance/sample_time",
        ]
    ]
    times.reset_index(level=0, inplace=True)
    times["synthesizer"] = times["synthesizer"].str.replace("Synthesizer", "")
    times["synthesizer"] = times["synthesizer"].str.replace("\[all\]", "")
    table_rows = ""
    rows = {row["synthesizer"]: row for _, row in times.iterrows()}
    table_rows += row_bn(rows["CLBN"])
    table_rows += row_bn(rows["PrivBN"])
    table_rows += row_bn(rows["TVAE"])
    table_rows += row_bn(rows["CTGAN"])
    table_rows += "\\bottomrule\n"
    table_rows += row_bn(rows["Synthsonic"])
    table_foot = r"""\end{tabular}
\end{table}"""
    return times, table_head + table_rows + table_foot


def row_real(row):
    synthesizer = (
        row["synthesizer"][:-4] if row["synthesizer"][-4:] == "_cpu" else "with gpu"
    )
    if row["synthesizer"] == "PrivBN_cpu":
        row = (
            f"\t{synthesizer: <10}&"
            f"\multicolumn{{2}}{{l}}{{{fmt(row['adult/fit_time']+row['adult/sample_time'])}}}&"
            f"\multicolumn{{2}}{{l}}{{{fmt(row['census/fit_time']+row['census/sample_time'])}}}&"
            f"\multicolumn{{2}}{{l}}{{{fmt(row['credit/fit_time']+row['credit/sample_time'])}}}&"
            f"\multicolumn{{2}}{{l}}{{{fmt(row['covtype/fit_time']+row['covtype/sample_time'])}}}&"
            f"\multicolumn{{2}}{{l}}{{{fmt(row['intrusion/fit_time']+row['intrusion/sample_time'])}}}&"
            f"\multicolumn{{2}}{{l}}{{{fmt(row['news/fit_time']+row['news/sample_time'])}}}"
            f"\\\\\n"
        )
    else:
        row = (
            f"\t{synthesizer: <10}&"
            f"{fmt(row['adult/fit_time'])}&"
            f"{fmt(row['adult/sample_time'])}&"
            f"{fmt(row['census/fit_time'])}&"
            f"{fmt(row['census/sample_time'])}&"
            f"{fmt(row['credit/fit_time'])}&"
            f"{fmt(row['credit/sample_time'])}&"
            f"{fmt(row['covtype/fit_time'])}&"
            f"{fmt(row['covtype/sample_time'])}&"
            f"{fmt(row['intrusion/fit_time'])}&"
            f"{fmt(row['intrusion/sample_time'])}&"
            f"{fmt(row['news/fit_time'])}&"
            f"{fmt(row['news/sample_time'])}"
            f"\\\\\n"
        )
    return row


def table_real():
    table_head = r"""\begin{table}[H]
    \setlength{\tabcolsep}{1}
    \caption{Computational efficiency of algorithms on real datasets (in seconds)}
    \label{tbl:efficiency-full-real}
    \centering
    \begin{tabular}{lllllllllllll}
    \toprule
                & \MC{adult}                & \MC{census}                   & \MC{credit}                   &  \MC{covtype}             & \MC{intrusion}        & \MC{news}         \\
                \cmidrule(r){2-3}           \cmidrule(r){4-5}               \cmidrule(r){6-7}               \cmidrule(r){8-9}           \cmidrule(r){10-11}     \cmidrule(r){12-13}
    Method      & fit           & sample    & fit           & sample        & fit           & sample        & fit           & sample    & fit       & sample   & fit   & sample     \\
    \midrule
"""
    times_cpu = make_leaderboard(
        "results/3_efficiency_cpu_real/",
        add_leaderboard=False,
    )
    times_cpu.index += "_cpu"
    times_gpu = make_leaderboard(
        "results/3_efficiency_gpu_real/",
        add_leaderboard=False,
    )
    times_gpu.index += "_gpu"

    times = pd.concat(([times_cpu, times_gpu]))
    times = times[
        [
            "adult/fit_time",
            "adult/sample_time",
            "census/fit_time",
            "census/sample_time",
            "credit/fit_time",
            "credit/sample_time",
            "covtype/fit_time",
            "covtype/sample_time",
            "intrusion/fit_time",
            "intrusion/sample_time",
            "news/fit_time",
            "news/sample_time",
        ]
    ]
    times.reset_index(level=0, inplace=True)
    times["synthesizer"] = times["synthesizer"].str.replace("Synthesizer", "")
    times["synthesizer"] = times["synthesizer"].str.replace("\[all\]", "")
    table_rows = ""
    rows = {row["synthesizer"]: row for _, row in times.iterrows()}
    table_rows += row_real(rows["CLBN_cpu"])
    table_rows += row_real(rows["PrivBN_cpu"])
    table_rows += row_real(rows["TVAE_cpu"])
    table_rows += row_real(rows["TVAE_gpu"])
    table_rows += row_real(rows["CTGAN_cpu"])
    table_rows += row_real(rows["CTGAN_gpu"])
    table_rows += "\\bottomrule\n"
    table_rows += row_real(rows["Synthsonic_cpu"])
    table_foot = r"""\end{tabular}
\end{table}"""
    return times, table_head + table_rows + table_foot


if __name__ == "__main__":
    times, tex = table_gmm()
    Path("output/table7.tex").write_text(tex)
    times.to_csv("output/table7.csv", index=False)

    times, tex = table_bn()
    Path("output/table8.tex").write_text(tex)
    times.to_csv("output/table8.csv", index=False)

    times, tex = table_real()
    Path("output/table9.tex").write_text(tex)
    times.to_csv("output/table9.csv", index=False)
