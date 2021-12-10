import argparse
import os
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Splitter")


def get_parser():
    parser = argparse.ArgumentParser(description="Split data into train/val/test.")
    parser.add_argument(
        "filtered_dataset", type=str, help="Path to the filtered.csv file"
    )
    parser.add_argument("output", type=str, help="Path to save result file")
    parser.add_argument(
        "--test_ratio",
        default=0.2,
        type=float,
        help="Ratio of the test split (of the all filtered dataset)",
    )
    parser.add_argument(
        "--val_ratio",
        default=0.1,
        type=float,
        help="Ratio of the validation split (of the filtered dataset without test data)",
    )
    parser.add_argument(
        "--homology_based",
        default="true",
        type=str,
        choices=["true", "false"],
        help="Split clusters of similar molecules instead of single molecules, if true",
    )
    parser.add_argument("--random_seed", default=228, type=int)
    return parser


def get_random_split(dataset: pd.DataFrame, test_ratio: float, val_ratio: float, rng):
    train_and_val, test = train_test_split(data, test_size=test_ratio, random_state=rng)
    train, val = train_test_split(train_and_val, test_size=val_ratio, random_state=rng)
    return train, val, test


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    data = pd.read_csv(args.filtered_dataset, index_col=0)

    rng = np.random.RandomState(args.random_seed)
    if args.homology_based == 'true':
        raise NotImplementedError()
    else:
        train, val, test = get_random_split(data, args.test_ratio, args.val_ratio, rng)
    
    # Create folder if there is none
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv")
    val.to_csv(output_dir / "val.csv")
    test.to_csv(output_dir / "test.csv")
    log.info(f"Done. Result is saved to {args.output}")
    

