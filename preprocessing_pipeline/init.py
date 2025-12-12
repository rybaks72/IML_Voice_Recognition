import os
from glob import glob
from pathlib import Path

from torch.utils.data import dataset

VERSION_FILE = "./kaggle_version.txt"
DATASET = 'quochoangvuvan/ml-voice-recognition'

def get_local_version():
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r") as f:
            return int(f.read().strip())
    return None

def get_kaggle_dataset_version():
    import kaggle
    owner, dataset = DATASET.split("/")
    meta = kaggle.api.dataset_metadata(owner, dataset)
    return meta['versionNumber']

def download():
    import kaggle
    kaggle.api.authenticate()
    version = get_kaggle_dataset_version()
    with open(VERSION_FILE, "w") as f:
        f.write(str(version))

def download_data():
    if not os.path.exists("./data") or get_local_version() is None:
        download()
        # print(kaggle.api.dataset_list_files(DATASET).files)
        print(f"Data successfully downloaded.")
    elif get_local_version() != get_kaggle_dataset_version():
        download()
        print("Data successfully updated.")
    else:
        print(f"Directory ./data already exists. Skipping download.")

def create_directories(target_dir):
    for p in glob("./data/*/*"):
        parent = os.path.basename(os.path.dirname(p))
        child = os.path.basename(p)

        directory = Path(f"{target_dir}/{parent}/{child}")
        directory.mkdir(parents=True, exist_ok=True)
        # print("Directory:", directory)