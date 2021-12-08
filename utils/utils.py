import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl


def change_directory_to_repo():
    """Changes working directory to the repository root folder."""
    current_dir = Path.cwd()
    for parent in current_dir.parents:
        # Repository is the first folder with the .git folder
        files = list(parent.glob(".git"))
        if files:
            os.chdir(str(parent))


def set_deafult_seed(seed=228):
    pl.seed_everything(seed, workers=True)  # set seed for pytorch lightning
    return np.random.RandomState(seed)  # set and return seed for sklearn
