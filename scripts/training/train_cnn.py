import numpy as np
from pathlib import Path
import argparse
import logging
import pickle

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Train-CNN")


def get_parser():
    parser = argparse.ArgumentParser(description="Trains 3D CNN for molecules classification by their voxel grid.")
    parser.add_argument(
        "input", type=str, help="Path to folder with train/val voxels"
    )
    parser.add_argument("output", type=str, help="Path to save result model")
    return parser

if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()
