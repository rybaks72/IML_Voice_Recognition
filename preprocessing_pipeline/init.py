import os
from glob import glob
from pathlib import Path

def dowload_data():
    if not os.path.exists("./data"):
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('quochoangvuvan/ml-voice-recognition', path=".", unzip=True)

        print(kaggle.api.dataset_list_files('quochoangvuvan/ml-voice-recognition').files)
        kaggle.api.dataset_metadata('quochoangvuvan/ml-voice-recognition', path=".")
        print(f"Data successfully downloaded.")
    else:
        print(f"Directory ./data already exists. Skipping download.")

def create_directories(target_dir):
    for p in glob('./data/*/*'):
        path,_ = os.path.splitext(p)
        path = "/".join([f'{target_dir}', *path.split("\\")[1:]])
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        # print("Directory:", directory)