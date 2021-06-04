from pathlib import Path

import pandas as pd
from sdgym.results import make_leaderboard


def table4(leaderboard, scores):
    table_head = r"""
    \begin{table}[H]
      \caption{Performance of synthsonic on synthetic data generation. Values closest (\answerTODO{}) to identity are marked bold. }
      \label{tbl:perf1}
      \centering
      \begin{tabular}{lllllll}
        \toprule
        &           \multicolumn{2}{c}{\texttt{grid}}     &   \multicolumn{2}{c}{\texttt{gridr}}&   \multicolumn{2}{c}{\texttt{ring}}\\
        \cmidrule(r){2-3} \cmidrule(r){4-5}\cmidrule(r){6-7}
        Method      & $\mathcal{L}_{syn}$     & $\mathcal{L}_{test}$    & $\mathcal{L}_{syn}$     & $\mathcal{L}_{test}$  & $\mathcal{L}_{syn}$     & $\mathcal{L}_{test}$    \\
        \midrule
        Identity
                                            & $-3.47$           & $-3.49$           & $-3.59$       & $-3.64$   & $-1.71$           & $-1.70$  \\
        \midrule
    """

    table_rows = ""
    name = "Identity"
    idx = "IdentitySynthesizer"
    s1 = leaderboard.at[idx, "grid/syn_likelihood"]
    t1 = leaderboard.at[idx, "grid/test_likelihood"]
    s2 = leaderboard.at[idx, "gridr/syn_likelihood"]
    t2 = leaderboard.at[idx, "gridr/test_likelihood"]
    s3 = leaderboard.at[idx, "ring/syn_likelihood"]
    t3 = leaderboard.at[idx, "ring/test_likelihood"]
    table_rows += f"\\texttt{{{name}}}   & ${s1:.02f}$           & ${t1:.02f}$           & ${s2:.02f}$       & ${t2:.02f}$  & ${s3:.02f}$       & ${t3:.02f}$\\\\ \n"
    table_rows += "\\midrule\n"
    for name in ["CLBN", "PrivBN", "TVAE", "CTGAN"]:
        idx = name
        if name != "CTGAN":
            idx += "Synthesizer"
        s1 = leaderboard.at[idx, "grid/syn_likelihood"]
        t1 = leaderboard.at[idx, "grid/test_likelihood"]
        s2 = leaderboard.at[idx, "gridr/syn_likelihood"]
        t2 = leaderboard.at[idx, "gridr/test_likelihood"]
        s3 = leaderboard.at[idx, "ring/syn_likelihood"]
        t3 = leaderboard.at[idx, "ring/test_likelihood"]
        table_rows += f"\\texttt{{{name}}}   & ${s1:.02f}$           & ${t1:.02f}$           & ${s2:.02f}$       & ${t2:.02f}$  & ${s3:.02f}$       & ${t3:.02f}$\\\\ \n"
    table_rows += "\\bottomrule\n"
    name = "Synthsonic"
    idx = "Synthsonic[all]"
    s1 = scores.at[idx, "grid/syn_likelihood"]
    t1 = scores.at[idx, "grid/test_likelihood"]
    s2 = scores.at[idx, "gridr/syn_likelihood"]
    t2 = scores.at[idx, "gridr/test_likelihood"]
    s3 = scores.at[idx, "ring/syn_likelihood"]
    t3 = scores.at[idx, "ring/test_likelihood"]
    table_rows += f"\\texttt{{{name}}}   & ${s1:.02f}$           & ${t1:.02f}$           & ${s2:.02f}$       & ${t2:.02f}$  & ${s3:.02f}$       & ${t3:.02f}$\\\\ \n"

    table_foot = r"""\end{tabular}
    \end{table}"""

    table = table_head + table_rows + table_foot

    Path("output/table4.tex").write_text(table)


def table5(leaderboard, scores):
    table_head = r"""\begin{table}[H]
      \caption{Performance of synthsonic on synthetic data generation}
      \label{tbl:perf2}
      \centering
      \begin{tabular}{lllllllll}
        \toprule
        &           \multicolumn{2}{c}{\texttt{asia}}     &   \multicolumn{2}{c}{\texttt{alarm}}&   \multicolumn{2}{c}{\texttt{child}}&   \multicolumn{2}{c}{\texttt{insurance}}\\
        \cmidrule(r){2-3} \cmidrule(r){4-5}\cmidrule(r){6-7}\cmidrule(r){8-9}
        Method      & $\mathcal{L}_{syn}$         & $\mathcal{L}_{test}$        & $\mathcal{L}_{syn}$         & $\mathcal{L}_{test}$        & $\mathcal{L}_{syn}$         & $\mathcal{L}_{test}$        & $\mathcal{L}_{syn}$     & $\mathcal{L}_{test}$   \\
        \midrule
    """

    table_rows = ""
    name = "Identity"
    idx = "IdentitySynthesizer"
    s1 = leaderboard.at[idx, "asia/syn_likelihood"]
    t1 = leaderboard.at[idx, "asia/test_likelihood"]
    s2 = leaderboard.at[idx, "alarm/syn_likelihood"]
    t2 = leaderboard.at[idx, "alarm/test_likelihood"]
    s3 = leaderboard.at[idx, "child/syn_likelihood"]
    t3 = leaderboard.at[idx, "child/test_likelihood"]
    s4 = leaderboard.at[idx, "insurance/syn_likelihood"]
    t4 = leaderboard.at[idx, "insurance/test_likelihood"]
    table_rows += f"\\texttt{{{name}}}   & ${s1:.02f}$           & ${t1:.02f}$           & ${s2:.02f}$       & ${t2:.02f}$  & ${s3:.02f}$       & ${t3:.02f}$& ${s4:.02f}$       & ${t4:.02f}$\\\\ \n"
    table_rows += "\\midrule\n"
    for name in ["CLBN", "PrivBN", "TVAE", "CTGAN"]:
        idx = name
        if name != "CTGAN":
            idx += "Synthesizer"
        s1 = leaderboard.at[idx, "asia/syn_likelihood"]
        t1 = leaderboard.at[idx, "asia/test_likelihood"]
        s2 = leaderboard.at[idx, "alarm/syn_likelihood"]
        t2 = leaderboard.at[idx, "alarm/test_likelihood"]
        s3 = leaderboard.at[idx, "child/syn_likelihood"]
        t3 = leaderboard.at[idx, "child/test_likelihood"]
        s4 = leaderboard.at[idx, "insurance/syn_likelihood"]
        t4 = leaderboard.at[idx, "insurance/test_likelihood"]
        table_rows += f"\\texttt{{{name}}}   & ${s1:.02f}$           & ${t1:.02f}$           & ${s2:.02f}$       & ${t2:.02f}$  & ${s3:.02f}$       & ${t3:.02f}$& ${s4:.02f}$       & ${t4:.02f}$\\\\ \n"
    table_rows += "\\bottomrule\n"
    name = "Synthsonic"
    idx = "Synthsonic[all]"
    s1 = scores.at[idx, "asia/syn_likelihood"]
    t1 = scores.at[idx, "asia/test_likelihood"]
    s2 = scores.at[idx, "alarm/syn_likelihood"]
    t2 = scores.at[idx, "alarm/test_likelihood"]
    s3 = scores.at[idx, "child/syn_likelihood"]
    t3 = scores.at[idx, "child/test_likelihood"]
    s4 = scores.at[idx, "insurance/syn_likelihood"]
    t4 = scores.at[idx, "insurance/test_likelihood"]
    table_rows += f"\\texttt{{{name}}}   & ${s1:.02f}$           & ${t1:.02f}$           & ${s2:.02f}$       & ${t2:.02f}$  & ${s3:.02f}$       & ${t3:.02f}$& ${s4:.02f}$       & ${t4:.02f}$\\\\ \n"

    table_foot = r"""\end{tabular}
    \end{table}"""

    table = table_head + table_rows + table_foot

    Path("output/table5.tex").write_text(table)


def table6(leaderboard, scores):
    table_head = r"""
    \begin{table}[H]
      \caption{Performance of synthsonic on synthetic data generation}
      \label{tbl:perf3}
      \centering
      \begin{tabular}{lllllll}
        \toprule
                    & \texttt{adult}& \texttt{census}   & \texttt{credit}   & \texttt{covtype}     & \texttt{intrusion}   & \texttt{news}\\
        Method      & $F_1$           & $F_1$             & $F_1$             & Macro $F_1$               & Macro $F_1$               & $R^2$    \\
        \midrule
    """

    table_rows = ""
    name = "Identity"
    idx = "IdentitySynthesizer"
    s1 = leaderboard.at[idx, "adult/f1"]
    t1 = leaderboard.at[idx, "adult/f1"]
    s2 = leaderboard.at[idx, "census/f1"]
    t2 = leaderboard.at[idx, "census/f1"]
    s3 = leaderboard.at[idx, "credit/f1"]
    t3 = leaderboard.at[idx, "credit/f1"]
    s4 = leaderboard.at[idx, "covtype/macro_f1"]
    t4 = leaderboard.at[idx, "covtype/macro_f1"]
    s5 = leaderboard.at[idx, "intrusion/macro_f1"]
    t5 = leaderboard.at[idx, "intrusion/macro_f1"]
    s6 = leaderboard.at[idx, "news/r2"]
    t6 = leaderboard.at[idx, "news/r2"]
    table_rows += f"\\texttt{{{name}}} & ${s1:.02f}$ & ${t1:.02f}$ & ${s2:.02f}$ & ${t2:.02f}$ & ${s3:.02f}$ & ${t3:.02f}$ & ${s4:.02f}$ & ${t4:.02f}$ & ${s5:.02f}$ & ${t5:.02f}$ & ${s6:.02f}$ & ${t6:.02f}$\\\\ \n"
    table_rows += "\\midrule\n"

    for name in ["CLBN", "PrivBN", "TVAE", "CTGAN"]:
        idx = name
        if name != "CTGAN":
            idx += "Synthesizer"
        s1 = leaderboard.at[idx, "adult/f1"]
        t1 = leaderboard.at[idx, "adult/f1"]
        s2 = leaderboard.at[idx, "census/f1"]
        t2 = leaderboard.at[idx, "census/f1"]
        s3 = leaderboard.at[idx, "credit/f1"]
        t3 = leaderboard.at[idx, "credit/f1"]
        s4 = leaderboard.at[idx, "covtype/macro_f1"]
        t4 = leaderboard.at[idx, "covtype/macro_f1"]
        s5 = leaderboard.at[idx, "intrusion/macro_f1"]
        t5 = leaderboard.at[idx, "intrusion/macro_f1"]
        s6 = leaderboard.at[idx, "news/r2"]
        t6 = leaderboard.at[idx, "news/r2"]
        table_rows += f"\\texttt{{{name}}} & ${s1:.02f}$ & ${t1:.02f}$ & ${s2:.02f}$ & ${t2:.02f}$ & ${s3:.02f}$ & ${t3:.02f}$ & ${s4:.02f}$ & ${t4:.02f}$ & ${s5:.02f}$ & ${t5:.02f}$ & ${s6:.02f}$ & ${t6:.02f}$\\\\ \n"
    table_rows += "\\bottomrule\n"
    name = "Synthsonic"
    idx = "Synthsonic[all]"
    s1 = scores.at[idx, "adult/f1"]
    t1 = scores.at[idx, "adult/f1"]
    s2 = scores.at[idx, "census/f1"]
    t2 = scores.at[idx, "census/f1"]
    s3 = scores.at[idx, "credit/f1"]
    t3 = scores.at[idx, "credit/f1"]
    s4 = scores.at[idx, "covtype/macro_f1"]
    t4 = scores.at[idx, "covtype/macro_f1"]
    s5 = scores.at[idx, "intrusion/macro_f1"]
    t5 = scores.at[idx, "intrusion/macro_f1"]
    s6 = scores.at[idx, "news/r2"]
    t6 = scores.at[idx, "news/r2"]
    table_rows += f"\\texttt{{{name}}} & ${s1:.02f}$ & ${t1:.02f}$ & ${s2:.02f}$ & ${t2:.02f}$ & ${s3:.02f}$ & ${t3:.02f}$ & ${s4:.02f}$ & ${t4:.02f}$ & ${s5:.02f}$ & ${t5:.02f}$ & ${s6:.02f}$ & ${t6:.02f}$\\\\ \n"

    table_foot = r"""\end{tabular}
    \end{table}"""

    table = table_head + table_rows + table_foot

    Path("output/table6.tex").write_text(table)


leaderboard = pd.read_csv("leaderboard.csv", index_col=0)
synth_scores = make_leaderboard(
    "results/1_leaderboard/",
    add_leaderboard=False,
)
table4(leaderboard, synth_scores)
table5(leaderboard, synth_scores)
table6(leaderboard, synth_scores)
