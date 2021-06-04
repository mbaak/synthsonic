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
    start_seed = 1337 + start_idx
    current_settings = get_synthsonic_options("all", dataset, start_seed)
    print(
        f"dataset_name={dataset}, current settings={current_settings}, seed={start_seed}"
    )

    try:
        scores = run(
            synthesizers=[get_synthsonic(dataset, synth_settings=current_settings)],
            datasets=[dataset],
            iterations=1,
            add_leaderboard=False,
            cache_dir=f"results/1_leaderboard/",
            iteration_start_idx=start_idx,
        )
    except ValueError:
        print(f"Failed to compute {dataset} scores")
