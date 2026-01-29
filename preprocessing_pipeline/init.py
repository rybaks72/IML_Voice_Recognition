import os
from glob import glob
from pathlib import Path
"This function was for downloading data from a shared kaggle dataset. Since we deliver the whole dataset along with the code, this function is redundant"
def download_data():
    pass
    # if not os.path.exists("./data"):
    #     import kaggle
    #     kaggle.api.authenticate()
    #     kaggle.api.dataset_download_files('quochoangvuvan/ml-voice-recognition', path=".", unzip=True)
    #
    #     print(kaggle.api.dataset_list_files('quochoangvuvan/ml-voice-recognition').files)
    #     kaggle.api.dataset_metadata('quochoangvuvan/ml-voice-recognition', path=".")
    #     print(f"Data successfully downloaded.")
    # else:
    #     print(f"Directory ./data already exists. Skipping download.")
