import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
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


def get_fingerprints(dataset: pd.DataFrame):
    fingerprints = []
    for row in dataset.itertuples():
        smiles = row[1]
        molecule = Chem.MolFromSmiles(smiles)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, 1024)
        fingerprints.append(fingerprint)
    return fingerprints


def clusterize_fingerprints(fps, cutoff=0.2):
    # See https://rdkit.readthedocs.io/en/latest/Cookbook.html#clustering-molecules
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return cs


def get_flattened_list(list_of_tuples):
    result = []
    for tuple in list_of_tuples:
        result.extend(tuple)
    return result


def get_homology_based_split(
    dataset: pd.DataFrame, test_ratio: float, val_ratio: float, rng
):
    fingerprints = get_fingerprints(dataset)
    clusters = clusterize_fingerprints(fingerprints)
    id_train_and_val, id_test = train_test_split(clusters, test_size=test_ratio)
    id_train, id_val = train_test_split(id_train_and_val, test_size=val_ratio)

    id_train = get_flattened_list(id_train)
    train = dataset.iloc[id_train]

    id_val = get_flattened_list(id_val)
    val = dataset.iloc[id_val]

    id_test = get_flattened_list(id_test)
    test = dataset.iloc[id_test]

    return train, val, test


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    data = pd.read_csv(args.filtered_dataset, index_col=0)

    rng = np.random.RandomState(args.random_seed)
    if args.homology_based == "true":
        train, val, test = get_homology_based_split(data, args.test_ratio, args.val_ratio, rng)
    else:
        train, val, test = get_random_split(data, args.test_ratio, args.val_ratio, rng)

    # Create folder if there is none
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv")
    val.to_csv(output_dir / "val.csv")
    test.to_csv(output_dir / "test.csv")
    log.info(f"Done. Result is saved to {args.output}")
