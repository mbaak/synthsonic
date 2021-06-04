import argparse

from sdgym import __version__, run

if __version__ != "0.2.2.fork":
    raise ImportError(
        f"Fork of SDGym required (0.0.2.fork), your version is: {__version__}"
    )

from sdgym.synthesizers import (
    CTGAN,
    CLBNSynthesizer,
    PrivBNSynthesizer,
    TVAESynthesizer,
)

from synthsonic_synth import get_synthsonic


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("dataset_group", choices=["gmm", "bn", "real"], default="gmm")
    parser.add_argument("--iterations", type=int, default=1)

    args = parser.parse_args()
    return args.device, args.dataset_group, args.iterations


def get_synthesizers(device, dataset):
    if device == "cpu":
        all_synthesizers = [
            get_synthsonic(
                dataset,
            ),
            TVAESynthesizer,
            CLBNSynthesizer,
            CTGAN,
            PrivBNSynthesizer,
        ]
    else:
        all_synthesizers = [
            TVAESynthesizer,
            CTGAN,
        ]
    return all_synthesizers


if __name__ == "__main__":
    device, dataset_group, repeats = parse_arguments()

    if dataset_group == "gmm":
        datasets = ["grid", "gridr", "ring"]
    elif dataset_group == "bn":
        datasets = ["child", "asia", "intrusion", "alarm"]
    else:
        datasets = ["adult", "news", "credit", "covtype", "insurance", "census"]

    for dataset in datasets:
        run(
            synthesizers=get_synthesizers(device, dataset),
            datasets=[dataset],
            iterations=repeats,
            add_leaderboard=False,
            cache_dir=f"results/3_efficiency_{device}_{dataset_group}/",
        )
