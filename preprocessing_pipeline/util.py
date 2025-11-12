from glob import glob
import torch
import numpy as np
import random as rand
import librosa

#INPUT: path to spectrogram data
def random_data_split(path="./"): #random test-train-validate datasets
    class0 = glob(f"{path}/spectogram_data/Class0/*")
    class1 = glob(f"{path}/spectogram_data/Class1/*")
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

    print(len(train['X']),len(train['Y']))
    print(len(test['X']),len(test['Y']))
    print(len(validate['X']),len(validate['Y']))
    # print(train)
    # print(test)
    # print(validate)
    return {
        'train': (torch.tensor(np.array(train['X'])).unsqueeze(1), torch.tensor(np.array(train['Y']))),
        'test': (torch.tensor(np.array(test['X'])).unsqueeze(1), torch.tensor(np.array(test['Y']))),
        'validate': (torch.tensor(np.array(validate['X'])).unsqueeze(1), torch.tensor(np.array(validate['Y'])))
    }

#preprocessing - normalize, trim, crop into 1 min audios, split into 3s clips
def preprocess_data(audio, sr, clip_length):
    #y, sr = librosa.load(audio) #NOTE: librosa.load by default standardizes the sr to 22050 HZ
    y_norm = librosa.util.normalize(audio)

    y_trimmed, _ = librosa.effects.trim(y_norm, top_db=20)
    intervals = librosa.effects.split(y_trimmed, top_db=20)
    y_no_silence = np.concatenate([y_trimmed[interval[0]:interval[1]] for interval in intervals])
    clip_length_samples = clip_length * sr
    # max_len = sr * 60
    #
    # y_cropped = y_trimmed[:max_len]
    y_clips = librosa.util.frame(y_no_silence, frame_length=clip_length_samples, hop_length=clip_length_samples).T
    return y_clips

#convert voice memo to spectograms
def convert_to_spectograms(audio, sr, clip_length):
    y_clips = preprocess_data(audio, sr, clip_length)
    spectograms = []

    for sample in y_clips:
        S = librosa.feature.melspectrogram(y=sample, sr=sr, n_mels=128, )
        S_db_mel = librosa.pcen(S, sr=sr)
        spectograms.append(S_db_mel)

    return spectograms

# print(random_data_split())