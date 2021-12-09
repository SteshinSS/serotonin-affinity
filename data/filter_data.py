import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Simple DataProcessor")

MIN_AFFINITY = 8.0
SMILES_PATH = "data/raw/smiles.tsv.gz"
AFFINITY_PATH = "data/raw/activities.tsv.gz"
TARGET_PATH = "data/preprocessed/filtered_data.csv"

smiles = pd.read_csv(SMILES_PATH, compression="gzip", sep="\t")
activity = pd.read_csv(AFFINITY_PATH, compression="gzip", sep="\t")

id_with_affinity = set(activity["molregno"].to_list())
all_id = set(smiles["molregno"].to_list())
smiles_with_affinity = all_id.intersection(id_with_affinity)

log.info(f"Total pairs smiles-known affinity: {len(smiles_with_affinity)}")
log.info(f"Creating filtered data table...")

filtered_smiles = []
filtered_affinity = []

for molregno in smiles_with_affinity:
    new_smiles = smiles[smiles["molregno"] == molregno]["canonical_smiles"].iloc[0]
    filtered_smiles.append(new_smiles)

    all_affinities = activity[activity["molregno"] == molregno]["pchembl_value"]
    if all_affinities.shape[0] > 1:
        new_affinity = all_affinities.mean(skipna=True) > MIN_AFFINITY
    else:
        new_affinity = all_affinities.iloc[0]
        if np.isnan(new_affinity):
            new_affinity = False
        else:
            new_affinity = new_affinity > 8.0
    filtered_affinity.append(int(new_affinity))


filtered_data = pd.DataFrame(
    {"smiles": filtered_smiles, "filtered_affinity": filtered_affinity}
)


filtered_data.to_csv(TARGET_PATH)
log.info(f"Done. Result is saved to {TARGET_PATH}")
