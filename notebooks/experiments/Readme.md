# Experiments

The experimental results in the paper can be reproduced using the instructions below.
Results will be stored in the ``results`` directory, and the paper's tables will be automatically generated and placed in the ``output`` directory.

To be able to run the experiments our fork of SDGym (v0.2.2) is necessary to be installed.
This extends the package to measure fit and sample time and improves reproducibility by providing different random seed values for each synthesizer run.

## 1. Effectivity 

Obtain results: ```bash experiments/1_leaderboard.sh ```

Reproduce Table 4, 5, 6: ```python experiments/1_leaderboard_tables.py```

Reproduce Table 1: ```python experiments/1_leaderboard_tables_agg.py```

Requires leaderboard.csv (originally obtained from [here](https://github.com/sdv-dev/SDGym/blob/master/sdgym/leaderboard.csv))

## 2. Ablation  analysis

Obtain results: ```bash experiments/2_ablation_analysis.sh```

Reproduce Table 3: ```python experiments/2_ablation_analysis_table.py```

## 3. Efficiency 

Obtain results (warning: takes multiple days):  ```bash experiments/3_efficiency_cpu.sh``` and  ```bash experiments/3_efficiency_gpu.sh```

Reproduce Table 7, 8 and 9 supplementary material:
- ```python experiments/3_efficiency_tables.py```

Reproduce Table 2
- ```python experiments/3_efficiency_tables_agg.py```

