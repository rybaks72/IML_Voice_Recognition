import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob
from itertools import cycle
from init import dowload_data, create_directories

import librosa
import librosa.display
import IPython.display as ipd
import os
from pathlib import Path

import torch
import random as rand

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

##def create_spectograms_from_data(clip_length):
#takes the desired clip length as input
#creates directories & downloads the dataset from kaggle
#=> uses all the functions above to create a spectogram for each voice memo for each person in each class
#output: no output

##def random_data_split():
#input: none
#randomly assigns spectograms into train-test-validate sets: class1 by clip for each person. 10-3-2
                                                            #class0 by person 14-4-2
#output: object with the following fields
        #train - train dataset
        #test - test dataset
        #validate - validate dataset


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
###

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

def random_data_split(): #random test-train-validate datasets
    class0 = glob("./spectogram_data/Class0/*")
    class1 = glob("./spectogram_data/Class1/*")
    train = {
        'X': [],
        'Y': [],
        'length': 14
    }
    test = {
        'X': [],
        'Y': [],
        'length': 4
    }
    validate = {
        'X': [],
        'Y': [],
        'length': 2
    }
    sets = [test, train, validate]
    #print(len(class0))

    for person in class0:
        person_path = glob(f'{person}/*.npz')

        available_sets = [s for s in sets if s['length'] != 0]
        dataset = rand.choice(available_sets)
        dataset['length']-=1
        for path in person_path:
            specs = np.load(path, allow_pickle=True)
            dataset['X'].extend(specs['X'])
            dataset['Y'].extend(specs['Y'])

    for person in class1:
        person_path = glob(f'{person}/*.npz')

        train['length'] = 10
        test['length'] = 3
        validate['length'] = 2

        for path in person_path:
            available_sets = [s for s in sets if s['length'] != 0]
            dataset = rand.choice(available_sets)
            dataset['length']-=1

            specs = np.load(path, allow_pickle=True)
            dataset['X'].extend(specs['X'])
            dataset['Y'].extend(specs['Y'])

    return {
        'train': (torch.tensor(np.array(train['X'])), torch.tensor(np.array(train['Y']))),
        'test': (torch.tensor(np.array(test['X'])), torch.tensor(np.array(test['Y']))),
        'validate': (torch.tensor(np.array(validate['X'])), torch.tensor(np.array(validate['Y'])))
    }






# create_spectograms_from_data(3)

print(random_ttv_datasets())