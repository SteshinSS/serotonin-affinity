import numpy as np
from moleculekit.smallmol.smallmol import SmallMol
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors
from pathlib import Path
import argparse
import logging
import pickle

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Voxels-Generator")


def get_parser():
    parser = argparse.ArgumentParser(description="Generates voxels for train/val/test conformers via MoleculeKit.")
    parser.add_argument(
        "input", type=str, help="Path to folder with train/val/test conformers"
    )
    parser.add_argument("output", type=str, help="Path to save result voxels")
    parser.add_argument("--voxel_size", type=float, default=1.0, help="Resolution of voxel step")
    parser.add_argument("--box_size_x", type=int, default=20, help="Shape of X coordinate of the voxel box")
    parser.add_argument("--box_size_y", type=int, default=20, help="Shape of Y coordinate of the voxel box")
    parser.add_argument("--box_size_z", type=int, default=20, help="Shape of Z coordinate of the voxel box")
    return parser

def generate_voxels(conformers: list, voxel_size: float, box_size: list):
    voxels = []
    is_active = []
    for i, (conformer, is_active_current) in enumerate(conformers):
        if i % 50 == 0:
            log.info(f"Generated {i} voxels out of {len(conformers)}...")
        molecule = SmallMol(conformer)
        voxel, _, _ = getVoxelDescriptors(molecule, boxsize=box_size, voxelsize=voxel_size, center=[0.0, 0.0, 0.0])
        voxel = voxel.reshape((box_size[0], box_size[1], box_size[2], -1))
        voxels.append(voxel)
        is_active.append(is_active_current)
    return np.stack(voxels), np.stack(is_active)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    box_size = [args.box_size_x, args.box_size_y, args.box_size_z]

    for dataset in ["train", "val", "test"]:
        path = input_path / (dataset + "_conformers")
        with open(path, "rb") as f:
            conformers = pickle.load(f)
        log.info(f"Generating voxels for the {dataset} dataset...")
        voxels, is_active = generate_voxels(conformers, args.voxel_size, box_size)
        np.savez_compressed(output_path / dataset, X=voxels, y=is_active)
        log.info("Done.\n")

