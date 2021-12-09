import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def change_directory_to_repo():
    """Changes working directory to the repository root folder."""
    current_dir = Path.cwd()
    for parent in current_dir.parents:
        # Repository is the first folder with the .git folder
        files = list(parent.glob(".git"))
        if files:
            os.chdir(str(parent))


change_directory_to_repo()

from utils import utils

rng = utils.set_deafult_seed()

data = pd.read_csv("data/preprocessed/filtered_data.csv", index_col=0)
from sklearn.model_selection import train_test_split

train_and_val, test = train_test_split(data, test_size=0.2, random_state=rng)
train, val = train_test_split(train_and_val, test_size=0.1, random_state=rng)

