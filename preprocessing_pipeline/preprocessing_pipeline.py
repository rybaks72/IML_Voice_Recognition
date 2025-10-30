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

sns.set_theme(style="white", palette = None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(color_pal)

def load(path):
    return glob(path)

#preprocessing - normalize, trim, crop into 1 min audios, split into 3s clips
def process_data(path, clip_length):
    y, sr = librosa.load(path)
    y_norm = librosa.util.normalize(y)
    y_trimmed, _ = librosa.effects.trim(y_norm, top_db=20)
    clip_length_samples = clip_length * sr
    max_len = sr * 60

    y_cropped = y_trimmed[:max_len]
    y_clips = librosa.util.frame(y_cropped, frame_length=clip_length_samples, hop_length=clip_length_samples).T
    spectograms = []

    for clip in y_clips:
        S = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=128, )
        S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
        spectograms.append(S_db_mel)

    return spectograms

#convert data from data directory to spectograms
def convert_to_spectograms(clip_length):
    dowload_data()

    all_spectograms = []
    all_labels = []


    class0 = glob("./data/Class0/*/*.mp3")
    class1 = glob("./data/Class1/*/*.mp3")

    #class0
    for clip in class0:
        specs = process_data(clip, clip_length)
        all_spectograms.extend(specs)
        all_labels.extend([0] * len(specs))

    # class1
    for clip in class1:
        specs = process_data(clip, clip_length)
        all_spectograms.extend(specs)
        all_labels.extend([1] * len(specs))

    X = np.array(all_spectograms)
    Y = np.array(all_labels)
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    #save to a file
    np.savez_compressed("./audio_data.npz", X=X, Y=Y)
    print("data saved")

convert_to_spectograms(3)
