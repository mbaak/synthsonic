# run ablation analysis for N runs per dataset iteratively
for ITERATION in {0..2}
do
    for DATASET in grid gridr ring adult alarm asia census child covtype credit insurance intrusion news
    do
        echo $DATASET $ITERATION
        python ./1_leaderboard.py $DATASET --start_idx $ITERATION
    done
done