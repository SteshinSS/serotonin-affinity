import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Filter")


def filter_dataset(
    smiles: pd.DataFrame,
    activity: pd.DataFrame,
    selected_molregno: set,
    affinity_threshold: float,
    keep_null: bool,
) -> pd.DataFrame:
    # TODO: Write docstring.
    filtered_smiles = []
    filtered_affinity = []

    for molregno in selected_molregno:
        all_affinities = activity[activity["molregno"] == molregno]["pchembl_value"]
        is_multiple_known_affinities = all_affinities.shape[0] > 1
        if is_multiple_known_affinities:
            is_active = all_affinities.mean(skipna=True) > affinity_threshold
        else:
            affinity = all_affinities.iloc[0]
            if np.isnan(affinity):
                if keep_null:
                    is_active = False
                else:
                    continue
            else:
                is_active = affinity > affinity_threshold
        filtered_affinity.append(int(is_active))

        new_smiles = smiles[smiles["molregno"] == molregno]["canonical_smiles"].iloc[0]
        filtered_smiles.append(new_smiles)

    filtered_data = pd.DataFrame(
        {"smiles": filtered_smiles, "filtered_affinity": filtered_affinity}
    )

    return filtered_data


def get_parser():
    parser = argparse.ArgumentParser(description="Filter trash out of raw data")
    parser.add_argument(
        "smiles_dataset", type=str, help="Path to the smiles.tsv.gz file"
    )
    parser.add_argument(
        "activity_dataset", type=str, help="Path to the activities.tsv.gz file"
    )
    parser.add_argument("output", type=str, help="Path to save result file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=8.0,
        help="Minimal value to consider a ligand to be active",
    )
    parser.add_argument(
        "--keep_null",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Keep molecules with null affinity value. Consider null molecules as not active, if true.",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    smiles = pd.read_csv(args.smiles_dataset, compression="gzip", sep="\t")
    activity = pd.read_csv(args.activity_dataset, compression="gzip", sep="\t")

    id_with_affinity = set(activity["molregno"].to_list())
    all_id = set(smiles["molregno"].to_list())
    smiles_with_affinity = all_id.intersection(id_with_affinity)

    log.info(f"Total pairs smiles-known affinity: {len(smiles_with_affinity)}")
    log.info(f"Creating filtered data table...")

    filtered_data = filter_dataset(
        smiles, activity, smiles_with_affinity, args.threshold, args.keep_null == "true"
    )

    # Create folder if there is none
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_data.to_csv(args.output)
    log.info(f"Done. Result is saved to {args.output}")
