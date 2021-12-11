import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path
import argparse
import logging
import pickle

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Conformer-Generator")


def get_parser():
    parser = argparse.ArgumentParser(description="Generates conformers for train/val/test datasets by rdkit's UFF.")
    parser.add_argument(
        "input", type=str, help="Path to folder with train/val/test.csv files"
    )
    parser.add_argument("output", type=str, help="Path to save result conformers")
    return parser


def get_conformer(molecule: Chem.rdchem.Mol):
    molecule = Chem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule, randomSeed=25391793)
    AllChem.MMFFOptimizeMolecule(molecule)
    return molecule


def convert_dataset(dataset: pd.DataFrame):
    conformers = []
    bad_smiles = []

    for i, row in enumerate(dataset.itertuples()):
        if i % 50 == 0:
            log.info(f"Processed {i} of {len(dataset)} molecules...")
        try:
            molecule = Chem.MolFromSmiles(row[1])
            molecule = get_conformer(molecule)
            conformers.append((molecule, row[2]))
        except (IndexError, ValueError):
            bad_smiles.append(row[1])
    
    if bad_smiles:
        log.warning("Couldn't calculate fingerprint for molecules:")
        for smile in bad_smiles:
            log.warning(smile)

    return conformers


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    for dataset in ["train", "val", "test"]:
        path = input_path / (dataset + ".csv")
        data = pd.read_csv(path, index_col=0)
        log.info(f"Generating conformers for the {dataset} dataset...")
        conformers = convert_dataset(data)
        with open(output_path / (dataset + "_conformers"), "wb") as f:
            pickle.dump(conformers, f)
        log.info("Done.\n")

