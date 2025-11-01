import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob
from itertools import cycle
from init import dowload_data

import librosa
import librosa.display
import IPython.display as ipd
import os
from pathlib import Path

#WHAT WAS DONE
###
# 1. Normalization
# 2. Trimming silence
# 3. Cropping clips to 1 min each
# 4. Transforming voice memos into mel-spectograms

# QUESTIONS
# 1. Do we denoise? MOST IMPORTANT
# 2. Do we augment data?
###

def create_directories():
    for p in glob('./data/*/*'):
        path,_ = os.path.splitext(p)
        path = "/".join(["./spectogram_data", *path.split("\\")[1:]])
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        # print("Directory:", directory)

#preprocessing - normalize, trim, crop into 1 min audios, split into 3s clips
def preprocess_data(path, clip_length):
    y, sr = librosa.load(path) #NOTE: librosa.load by default standardizes the sr to 22050 HZ
    y_norm = librosa.util.normalize(y)
    y_trimmed, _ = librosa.effects.trim(y_norm, top_db=20)
    clip_length_samples = clip_length * sr
    max_len = sr * 60

    y_cropped = y_trimmed[:max_len]
    y_clips = librosa.util.frame(y_cropped, frame_length=clip_length_samples, hop_length=clip_length_samples).T
    return y_clips, sr

#convert voice memo to spectograms
def convert_to_spectograms(path, clip_length):
    y_clips, sr = preprocess_data(path, clip_length)
    spectograms = []

    for sample in y_clips:
        S = librosa.feature.melspectrogram(y=sample, sr=sr, n_mels=128, )
        S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
        spectograms.append(S_db_mel)

    return spectograms

#save spectograms in .npz format
def save_spectograms(path, spectograms, label):
    path, _ = os.path.splitext(path)
    path = path.split("/")[2:][0].split("\\")
    name = path[-1] + ".npz"
    path.remove(path[-1])
    path.append(name)
    path = "/".join(["./spectogram_data", *path]).lower()
    X = np.array(spectograms)
    Y = np.array([label] * len(spectograms))
    # print(f"X shape: {X.shape}")
    # print(f"Y shape: {Y.shape}")

    np.savez_compressed(path, X=X, Y=Y)

#function used to process gathered data
def create_spectograms_from_data(clip_length):
    dowload_data()
    create_directories()

    class0 = glob("./data/Class0/*/*.mp3")
    for path in class0:
        spectograms = convert_to_spectograms(path, clip_length)
        save_spectograms(path, spectograms, 0)

    class1 = glob("./data/Class1/*/*.mp3")
    for path in class1:
        spectograms = convert_to_spectograms(path, clip_length)
        save_spectograms(path, spectograms, 1)


# create_spectograms_from_data(3)
