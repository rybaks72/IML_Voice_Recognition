import kaggle
import os
from glob import glob
from pathlib import Path

# Setup functions for downloading the Kaggle dataset and creating the required folder structure

def dowload_data():
    if not os.path.exists(".\\data"):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('quochoangvuvan/ml-voice-recognition', path=".", unzip=True)

        print(kaggle.api.dataset_list_files('quochoangvuvan/ml-voice-recognition').files)
        kaggle.api.dataset_metadata('quochoangvuvan/ml-voice-recognition', path=".")
        print(f"Data successfully downloaded.")
    else:
        print(f"Directory ./data already exists. Skipping download.")

def create_directories(target_dir):
    if not os.path.exists(f".\\{target_dir}"):
        for p in glob('.\\data\\*\\*'):
            path,_ = os.path.splitext(p)
            path = "\\".join([f'{target_dir}', *path.split("\\")[2:]])
            directory = Path(path)
            directory.mkdir(parents=True, exist_ok=True)
            # print("Directory:", directory)
        print("Directories created.")
    else:
        print(f"{target_dir} already exists.")