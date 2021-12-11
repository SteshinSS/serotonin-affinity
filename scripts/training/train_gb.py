import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Train-GB")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train gradient boosting tree classifier for activity prediction."
    )
    parser.add_argument(
        "input", type=str, help="Path to folder with train/val/test.npz features"
    )
    parser.add_argument("output", type=str, help="Path to save result model")
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.1,
        help="Fraction of samples used for fitting a single tree",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="The number of boosting stages to perform",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=3,
        help="The maximum depth of the individual regression estimators",
    )
    parser.add_argument(
        "--use_weights",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Use weighted loss. Use it if you have unbalanced dataset",
    )
    return parser


def get_weight(labels: np.ndarray):
    total_elements = labels.size
    total_ones = (labels == 1).sum()
    coefficient = (total_elements - total_ones) / total_ones
    weights = np.ones(labels.shape)
    weights[labels == 1] *= coefficient
    return weights


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    gb = GradientBoostingClassifier(
        subsample=args.subsample,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
    )
    dataset = np.load(args.input)

    if args.use_weights:
        weights = get_weight(dataset["y"])
    else:
        weights = None

    gb.fit(dataset["X"], dataset["y"], weights)

    # Create folder if there is none
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(gb, f)
