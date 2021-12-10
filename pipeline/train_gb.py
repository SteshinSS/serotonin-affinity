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
    parser.add_argument("--subsample", type=float, default=0.1, help="Fraction of samples used for fitting a single tree")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    gb = GradientBoostingClassifier(subsample=args.subsample)
    dataset = np.load(args.input)

    gb.fit(dataset["X"], dataset["y"])

    # Create folder if there is none
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(gb, f)

