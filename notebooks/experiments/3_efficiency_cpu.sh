DEVICE="cpu"
python experiments/3_efficiency.py $DEVICE gmm
python experiments/3_efficiency.py $DEVICE bn
python experiments/3_efficiency.py $DEVICE real