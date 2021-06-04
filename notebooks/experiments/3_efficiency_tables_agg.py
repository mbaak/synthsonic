from pathlib import Path

import pandas as pd


def fmt(v):
    if isinstance(v, float):
        rounded = round(v)
        if v == 0:
            return "$< 1$"
        return f"${rounded}$"
    return v


def row_agg(row_cpu, row_gpu=None):
    if row_gpu is None:
        row_gpu = {"fit_mean": "-", "sample_mean": "-"}
    synthesizer = row_cpu["synthesizer"][:-4]
    if row_cpu["synthesizer"] == "PrivBN_cpu":
        row = (
            f"\t\\texttt{{{synthesizer: <10}}}&"
            f"\multicolumn{{2}}{{l}}{{{fmt(row_cpu['fit_mean']+row_cpu['sample_mean'])}}}&"
            f"\multicolumn{{2}}{{l}}{{-}}"
            f"\\\\\n"
        )
    else:
        row = (
            f"\t\\texttt{{{synthesizer: <10}}}&"
            f"{fmt(row_cpu['fit_mean'])}&"
            f"{fmt(row_cpu['sample_mean'])}&"
            f"{fmt(row_gpu['fit_mean'])}&"
            f"{fmt(row_gpu['sample_mean'])}"
            f"\\\\\n"
        )
    return row


def table_agg():
    table_head = r"""\begin{table}[!tbh]
    \vspace{-4em}
    \caption{Efficiency of Synthsonic on real datasets, compared against top-performers on SDGym leaderboard v0.2.2. Reported times (sec) are averages over the six datasets, unless otherwise mentioned.}
    \label{sample-table}
    \centering
    \begin{tabular}{lllll}
    \toprule
                & \multicolumn{4}{c}{Time (s)}                                  \\
                & \multicolumn{2}{c}{CPU}       & \multicolumn{2}{c}{GPU}       \\
                \cmidrule(r){2-3}               \cmidrule(r){4-5}
    Method      & fit           & sample        & fit   & sample                \\
    \midrule
"""
    real = pd.read_csv("output/table9.csv")
    real["fit_mean"] = real[
        [col for col in real.columns if col.endswith("/fit_time")]
    ].mean(axis=1)
    real["sample_mean"] = real[
        [col for col in real.columns if col.endswith("/sample_time")]
    ].mean(axis=1)

    table_rows = ""
    rows = {row["synthesizer"]: row for _, row in real.iterrows()}
    table_rows += row_agg(rows["CLBN_cpu"], None)
    table_rows += row_agg(rows["PrivBN_cpu"], None)
    table_rows += row_agg(rows["TVAE_cpu"], rows["TVAE_gpu"])
    table_rows += row_agg(rows["CTGAN_cpu"], rows["CTGAN_gpu"])
    table_rows += "\\bottomrule\n"
    table_rows += row_agg(rows["Synthsonic_cpu"], None)
    table_foot = r"""\end{tabular}
    \vspace{-3em}
\end{table}"""
    return table_head + table_rows + table_foot


if __name__ == "__main__":
    tex = table_agg()
    Path("output/table2.tex").write_text(tex)
