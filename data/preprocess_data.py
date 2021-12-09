import numpy as np
import pandas as pd
from rdkit import Chem


def get_topological_fingerprint(molecule: Chem.rdchem.Mol):
    return Chem.RDKFingerprint(molecule).ToList()


def convert_dataset(dataset: pd.DataFrame):
    features = []
    affinity = []
    bad_smiles = []

    for row in dataset.itertuples():
        try:
            molecule = Chem.MolFromSmiles(row[1])
            fingerprint = get_topological_fingerprint(molecule)
            features.append(fingerprint)
            affinity.append(row[2])
        except IndexError:
            bad_smiles.append(row[1])

    features = np.row_stack(features)  # type: ignore
    return features, np.array(affinity), bad_smiles


if __name__ == "__main__":
    TRAIN_PATH = "data/preprocessed/train.csv"
    train = pd.read_csv(TRAIN_PATH, index_col=0)
    train_features, train_affinity, _ = convert_dataset(train)
    np.savez_compressed("data/preprocessed/train", X=train_features, y=train_affinity)

    VAL_PATH = "data/preprocessed/val.csv"
    val = pd.read_csv(VAL_PATH, index_col=0)
    val_features, val_affinity, _ = convert_dataset(val)
    np.savez_compressed("data/preprocessed/val", X=val_features, y=val_affinity)

    TEST_PATH = "data/preprocessed/test.csv"
    test = pd.read_csv(TEST_PATH, index_col=0)
    test_features, test_affinity, _ = convert_dataset(test)
    np.savez_compressed("data/preprocessed/test", X=test_features, y=test_affinity)
