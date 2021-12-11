import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Featurizer-MorganFP")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Calculates Morgan fingerprints for train/val/test dataset."
    )
    parser.add_argument(
        "input", type=str, help="Path to folder with train/val/test.csv files"
    )
    parser.add_argument("output", type=str, help="Path to save result features")
    parser.add_argument(
        "--radius", type=int, default=2, help="Radius of Morgan fingerprints"
    )
    parser.add_argument(
        "--n_bits", type=int, default=1024, help="Number of bit of a fingerprint"
    )
    return parser


def get_morgan_fingerprint(molecule: Chem.rdchem.Mol, radius: int, n_bits: int):
    return AllChem.GetMorganFingerprintAsBitVect(molecule, radius, n_bits).ToList()


def convert_dataset(dataset: pd.DataFrame, radius: int, n_bits: int):
    features = []
    is_active = []
    bad_smiles = []

    for row in dataset.itertuples():
        try:
            molecule = Chem.MolFromSmiles(row[1])
            fingerprint = get_morgan_fingerprint(molecule, radius, n_bits)
            features.append(fingerprint)
            is_active.append(row[2])
        except IndexError:
            bad_smiles.append(row[1])

    if bad_smiles:
        log.warning("Couldn't calculate fingerprint for molecules:")
        for smile in bad_smiles:
            log.warning(smile)

    return np.row_stack(features), np.array(is_active)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    for dataset in ["train", "val", "test"]:
        log.info("Featurizing %s dataset...", dataset)
        path = input_path / (dataset + ".csv")
        data = pd.read_csv(path, index_col=0)
        features, is_active = convert_dataset(data, args.radius, args.n_bits)
        np.savez_compressed(output_path / dataset, X=features, y=is_active)
        log.info("Done.\n")
