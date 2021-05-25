from pathlib import Path

import pandas as pd


for file_name in Path('./').glob('scores_*.csv'):
    print(file_name.stem)
    df = pd.read_csv(file_name, index_col=0)

    c1 = df.columns[0]
    c2 = df.columns[0].replace("_mean", "_std")

    # https://stats.stackexchange.com/a/160481
    my_col = df[c2] * 1.96 # 95% confidence interval
    scores = pd.Series([f"{m:.3f} ± {s:.3f}" for m, s in zip(df[c1], my_col)], index=df.index)

    # automatic to latex
    # scores.to_latex(f"abc.tex")

    print(scores)
