import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob
from itertools import cycle

import librosa
import librosa.display
import IPython.display as ipd
import os
from pathlib import Path

import torch
from init import dowload_data, create_directories
from util import preprocess_data, convert_to_spectograms, convert_with_pitch_shift

import gc
#GUIDE
###
##process_data(path, clip_length):
#takes the voice memo path and the desired clip length as inputs
#=> normalizes, trims, and divides the voice memo into <clip_length> second clips
#output: list of clips, sample rate

##convert_to_spectograms(path, clip_length):
#takes the voice memo path and the desired clip length as inputs
#=> processes data via process_data
#=> converts each clip from the process_data output into a melspectogram
#output: list of melspectograms derived from the processed voice memo

##create_spectograms_from_data(clip_length):
#takes the desired clip length as input
#creates directories & downloads the dataset from kaggle
#=> uses all the functions above to create a spectogram for each voice memo for each person in each class
#output: no output

##random_data_split():
#input: none
#randomly assigns spectograms into train-test-validate sets: class1 by clip for each person. 10-3-2
                                                            #class0 by person 14-4-2
#output: object with the following fields
        #train - train dataset
        #test - test dataset
        #validate - validate dataset

##create_raw_spectograms():
###

#WHAT WAS DONE
###
# 1. Normalization
# 2. Trimming silence
# 3. Cropping clips to 1 min each
# 4. Transforming voice memos into mel-spectograms

# QUESTIONS
# 1. Do we denoise? MOST IMPORTANT
# 2. Do we augment data?
# 3. Do we save the spectograms in files or do we just pass them as a function output
###

#save spectograms in .npz format
def save_spectrogram(path,target_dir, spectrogram, label):
    path, _ = os.path.splitext(path)
    path = path.split("/")[2:][0].split("\\")
    name = path[-1] + ".npz"
    path.remove(path[-1])
    path.append(name)
    path = "/".join([f"{target_dir}", *path]).lower()
    X = np.array(spectrogram)
    Y = np.array([label] * len(spectrogram))
    # print(f"X shape: {X.shape}")
    # print(f"Y shape: {Y.shape}")

    np.savez_compressed(path, X=X, Y=Y)

def create_raw_spectrogram():
    dowload_data()
    create_directories("./raw_spectrogram")
    class0 = glob("./data/Class0/*/*.mp3")
    for path in class0:
        y, sr = librosa.load(path)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, )
        spectrogram = librosa.amplitude_to_db(S, ref=np.max)
        save_spectrogram(path, "./raw_spectrogram", spectrogram, 0)

    class1 = glob("./data/Class1/*/*.mp3")
    for path in class1:
        y, sr = librosa.load(path)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, )
        spectrogram = librosa.amplitude_to_db(S, ref=np.max)
        save_spectrogram(path, "./raw_spectrogram", spectrogram, 1)

def spectrogram_conversion_loop(path_list, label, clip_length, pitch=False):
    for path in path_list:
        audio, sr = librosa.load(path)
        if pitch:
            spectrogram = convert_with_pitch_shift(audio, sr, clip_length)
        else:
            spectrogram = convert_to_spectograms(audio, sr, clip_length)

        save_spectrogram(path, "./spectogram_data", spectrogram, label)
        del audio, spectrogram
        gc.collect()

#function used to process gathered data
def create_spectrogram_from_data(clip_length):
    dowload_data()
    create_directories("./spectogram_data")

    class0 = glob("./data/Class0/*/*.mp3")
    spectrogram_conversion_loop(class0, 0, clip_length)

    augmented_class0 = glob("./data/Class0/*/*.mp3")[1::4]
    spectrogram_conversion_loop(augmented_class0, 0, clip_length, True)

    noise = glob("./data/Random_Noise/*/*.mp3")
    spectrogram_conversion_loop(noise, 0, clip_length, True)

    class1 = glob("./data/Class1/*/*.mp3")
    spectrogram_conversion_loop(class1, 1, clip_length)
create_spectrogram_from_data(3)
#create_raw_spectrogram()