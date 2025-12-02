from glob import glob
import torch
import numpy as np
import random as rand
import librosa

def helper(path_lists, train, test, validate):
    sets = [test, train, validate]
    count = 0
    for path in path_lists:
        available_sets = [s for s in sets if s['length'] != 0]
        dataset = rand.choice(available_sets)
        dataset['length']-=1

        specs = np.load(path, allow_pickle=True)
        dataset['X'].extend(specs['X'])
        dataset['Y'].extend(specs['Y'])
        count += len(specs["Y"])
    return count

#INPUT: path to spectrogram data
def random_data_split(path="./"): #random test-train-validate datasets
    class0 = glob(f"{path}/spectogram_data/Class0/*")
    class1 = glob(f"{path}/spectogram_data/Class1/*")
    noise_noise = glob(f"{path}/spectogram_data/Random_Noise/noise/*")
    noise_people = glob(f"{path}/spectogram_data/Random_Noise/people/*")
    print(path)
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

    class1_count = 0
    class0_count = 0

    for person in class0:
        person_path = glob(f'{person}/*.npz')

        available_sets = [s for s in sets if s['length'] != 0]
        dataset = rand.choice(available_sets)
        dataset['length']-=1
        for path in person_path:
            specs = np.load(path, allow_pickle=True)
            dataset['X'].extend(specs['X'])
            dataset['Y'].extend(specs['Y'])
            class0_count += len(specs["Y"])

    for person in class1:
         person_path = glob(f'{person}/*.npz')
         train['length'] = 10
         test['length'] = 3
         validate['length'] = 2
         class1_count += helper(person_path, train, test, validate)

    #noise_people
    train['length'] =  9
    test['length'] = 3
    validate['length'] = 2
    class0_count += helper(noise_people,train,test,validate)

    #noise_noise
    train['length'] = 3
    test['length'] = 2
    validate['length'] = 1
    class0_count += helper(noise_noise, train, test, validate)

    print(len(train['X']),len(train['Y']))
    print(len(test['X']),len(test['Y']))
    print(len(validate['X']),len(validate['Y']))
    print(len(train['X']) + len(test['X']) +len(validate['X']))
    # print(train)
    # print(test)
    # print(validate)
    return {
        'train': (torch.tensor(np.array(train['X'])).unsqueeze(1), torch.tensor(np.array(train['Y']))),
        'test': (torch.tensor(np.array(test['X'])).unsqueeze(1), torch.tensor(np.array(test['Y']))),
        'validate': (torch.tensor(np.array(validate['X'])).unsqueeze(1), torch.tensor(np.array(validate['Y']))),
        "weight": class0_count / class1_count,
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
        S_pcen = librosa.pcen(S, sr=sr, time_constant=0.4,gain=0.4)
        spectograms.append(S_pcen)

    return spectograms


def convert_with_pitch_shift(audio, sr, clip_length):
    lower = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-3)
    upper = librosa.effects.pitch_shift(audio, sr=sr, n_steps=3)

    spectrogram = convert_to_spectograms(audio, sr, clip_length)
    spectrogram.extend(convert_to_spectograms(lower, sr, clip_length))
    spectrogram.extend(convert_to_spectograms(upper, sr, clip_length))

    return spectrogram
# print(random_data_split())