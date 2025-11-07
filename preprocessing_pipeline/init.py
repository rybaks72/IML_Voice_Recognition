import kaggle
import os

def dowload_data():
    if not os.path.exists("./data"):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('quochoangvuvan/ml-voice-recognition', path=".", unzip=True)

        print(kaggle.api.dataset_list_files('quochoangvuvan/ml-voice-recognition').files)
        kaggle.api.dataset_metadata('quochoangvuvan/ml-voice-recognition', path=".")
        print(f"Data successfully downloaded.")
    else:
        print(f"Directory ./data already exists. Skipping download.")
