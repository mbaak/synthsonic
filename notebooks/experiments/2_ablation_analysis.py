import argparse
import logging

from sdgym import __version__, run

if __version__ != "0.2.2.fork":
    raise ImportError(
        f"Fork of SDGym required (0.0.2.fork), your version is: {__version__}"
    )

from synthsonic_synth import get_synthsonic, get_synthsonic_options

logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=[
            "grid",
            "gridr",
            "ring",
            "adult",
            "alarm",
            "asia",
            "census",
            "child",
            "covtype",
            "credit",
            "insurance",
            "intrusion",
            "news",
        ],
    )
    parser.add_argument("--start_idx", type=int, default=0)

    args = parser.parse_args()
    return args.dataset, args.start_idx


if __name__ == "__main__":
    dataset, start_idx = parse_arguments()

    choices = [
        "no_pca",
        "no_clf",
        "no_calibration",
        "no_tuning",
        "all",
        # Other options:
        # 'no_pca_mi',
        # 'auto-tree-cl',
        # 'auto-tree-tan',
        # 'auto-bins-uniform',
        # 'knuth-bins-uniform',
        # 'blocks-bins-uniform',
        # 'auto-bins-calibration',
        # 'oracle-tree-class',
    ]

    all_synthesizers = []
    start_seed = 1337 + start_idx
    for ablation in choices:
        current_settings = get_synthsonic_options(ablation, dataset, start_seed)
        print(
            f"ablation={ablation}, dataset_name={dataset}, current settings={current_settings}, seed={start_seed}"
        )

        all_synthesizers.append(
            get_synthsonic(dataset, synth_settings=current_settings, ablation=ablation)
        )

    try:
        scores = run(
            synthesizers=all_synthesizers,
            datasets=[dataset],
            iterations=1,
            add_leaderboard=False,
            cache_dir=f"results/2_ablation_analysis/{dataset}/raw/",
            iteration_start_idx=start_idx,
        )
    except ValueError:
        print(f"Failed to compute {dataset} scores")
