# Simple script for data downloading
import logging
import shutil
import tempfile

import requests  # type: ignore
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Simple Downloader")

SMILES_URL = (
    "https://drive.google.com/uc?export=download&id=1UcLbBOBNZ2EAcgv7W3b2jRIrfWXzCbJb"
)
ACTIVITY_URL = (
    "https://drive.google.com/uc?export=download&id=1QnaTezmHxgBYt_r4UyOG6B6R4yb-TmB6"
)


def download_to_tempfile(url: str):
    """Download file into tempfile
    Args:
        url (str): url of a file
    Returns:
        file: temporary file.
    Example:
        temp_file = download_to_tempfile('https://site.com/dataset')
        data = ad.read_h5ad(temp_file.name)
    """
    temp_file = tempfile.NamedTemporaryFile("wb")

    # In such way there is no need to keep file in memory.
    # The file will be saved by blocks.
    # The progress bar is took from here https://github.com/shaypal5/tqdl/blob/master/tqdl/core.py
    response = requests.get(url, stream=True, timeout=None)
    response.raise_for_status()

    file_size = int(response.headers.get("Content-Length", None))
    block_size = 1024
    progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True)
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        temp_file.write(data)
    progress_bar.close()
    temp_file.flush()
    if file_size != 0 and progress_bar.n != file_size:
        log.error("Something went wrong in downloading.")
    return temp_file


def download_all():
    log.info("Downloading smiles archive from %s...", SMILES_URL)
    smiles_tempfile = download_to_tempfile(SMILES_URL)
    shutil.copy(smiles_tempfile.name, "data/raw/smiles.tsv.gz")
    log.info("Done.")

    log.info("Downloading activity data from %s...", ACTIVITY_URL)
    activity_tempfile = download_to_tempfile(ACTIVITY_URL)
    shutil.copy(activity_tempfile.name, "data/raw/activities.tsv.gz")
    log.info("Done.")


if __name__ == "__main__":
    download_all()
