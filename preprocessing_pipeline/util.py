import torch
import numpy as np
import random as rand
import librosa, gc
from preprocessing_pipeline.augmentation import pitch_shift
from preprocessing_pipeline.preprocessing import preprocess_data
from preprocessing_pipeline.init import *
import math

"""
fix_len: Pads or truncates an audio signal to a fixed target length.
Input: audio (numpy array), target_len (int)
Output: Audio signal with exact target length (numpy array)
"""
def fix_len(audio, target_len):
    if len(audio) == target_len:
        return audio
    elif len(audio)>target_len:
        return audio[:target_len]

    repeats = target_len // len(audio) + 1
    extended = np.tile(audio, repeats)
    return extended[:target_len]

"""
get_stats: Computes mean and standard deviation of log-mel spectrogram values across the dataset.
Input: clip_length (int)
Output: Dataset spectrogram mean (float), standard deviation (float)
"""
def get_stats(clip_length=3):
    download_data()
    total = 0.0
    total_sq = 0.0
    count = 0

    for path in glob("./data/*/*/*.mp3"):
        audio, sr = librosa.load(path, sr=None)
        clips = preprocess_data(audio, sr, clip_length)

        for clip in clips:
            clip = fix_len(clip, sr * clip_length)

            S = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=128)
            S_log = librosa.power_to_db(S, ref=np.max)

            total += S_log.sum()
            total_sq += (S_log ** 2).sum()
            count += S_log.size

    mean = total / count
    std = np.sqrt(total_sq / count - mean ** 2) + 1e-6

    return mean, std

MEAN, STD = get_stats()

"""
get_name: Extracts speaker/person identifier from a file path.
Input: path (str)
Output: Speaker name (str)
"""
def get_name(path):
    path = path.replace("\\", "/").split("/")
    return path[-2]

"""
helper: Loads audio files, preprocesses them into clips, converts to spectrograms, and assigns them to datasets.
Input: path_lists (list), train/test/validate (dict), clip_length (int), label (int)
Output: Number of samples added (int)
"""
#convert voice memo to spectograms
def helper(path_lists, train, test, validate, clip_length, label):
    sets = [test, train, validate]
    count = 0
    for path in path_lists:
        available_sets = [s for s in sets if s['length'] != 0]
        dataset = rand.choice(available_sets)
        dataset['length'] -= 1
        specs = {
            'X': [],
            'Y': []
        }

        audio, sr = librosa.load(path)
        specs['X'].extend(preprocess_data(audio, sr, clip_length, augment=True if dataset['name'] == 'train' else False, label=label))
        specs['Y'].extend([label] * len(specs['X']))

        pitch_count = 0
        # if dataset['name'] == 'train':
        #     pitched = pitch_shift([audio], sr, pitch=3)
        #     specs['X'].extend(pitched)
        #     specs['Y'].extend([0] * len(pitched))
        #     pitch_count = len(pitched)

        specs['X'] = convert_to_spectrogram(specs['X'], sr, clip_length)
        dataset['X'].extend(specs['X'])
        dataset['Y'].extend(specs['Y'])
        count += len(specs["Y"])
        dataset['labels'].extend([get_name(path)] * (len(specs['Y']) - pitch_count))
        dataset['labels'].extend([f"pitched_{get_name(path)}"] * pitch_count)
        del audio, sr
    gc.collect()
    return count

"""
random_data_split: Splits dataset audio files into train/test/validate spectrogram tensors with optional augmentation.
Input: path (str), clip_length (int)
Output: Dictionary containing datasets, labels, and class weight (dict)
"""
#INPUT: path to spectrogram data
def random_data_split(path="./", clip_length=3): #random test-train-validate datasets
    download_data()
    class0 = glob(f"{path}/data/Class0/*")
    class1 = glob(f"{path}/data/Class1/*")
    noise_noise = glob(f"{path}/data/Random_Noise/noise/*")
    noise_people = glob(f"{path}/data/Random_Noise/people/*")
    #print(class1)
    #print(f"{path}/spectogram_data/Class1/*")

    train = {
        'X': [],
        'Y': [],
        'labels': [],
        'length': 14,
        'name': 'train'
    }
    test = {
        'X': [],
        'Y': [],
        'labels': [],
        'length': 4,
        'name': 'test'
    }
    validate = {
        'X': [],
        'Y': [],
        'labels': [],
        'length': 2,
        'name': 'validate'
    }
    #print(len(class0))

    class1_count = 0
    class0_count = 0

    for person in class0:
        person_path = glob(f'{person}/*.mp3')
        #NEW
        train['length'] = 2
        test['length'] = 1
        validate['length'] = 1
        class0_count += helper(person_path, train, test, validate, clip_length, label=0)
        ##OLD
        # available_sets = [s for s in sets if s['length'] != 0]
        # dataset = rand.choice(available_sets)
        # dataset['length']-=1
        # for path in person_path:
        #     specs = np.load(path, allow_pickle=True)
        #     dataset['X'].extend(specs['X'])
        #     dataset['Y'].extend(specs['Y'])
    print("Class0 done")

    for person in class1:
         person_path = glob(f'{person}/*.mp3')
         train['length'] = math.floor(0.7*len(person_path))
         validate['length'] = math.floor(0.1*len(person_path))
         test['length'] = len(person_path) - train['length'] - validate['length']
         class1_count += helper(person_path, train, test, validate, clip_length, label=1)
    print("Class1 done")

    #noise_people
    train['length'] =  math.floor(0.7*len(noise_people))
    validate['length'] = math.floor(0.1*len(noise_people))
    test['length'] =len(noise_people) - train['length'] - validate['length']
    class0_count += helper(noise_people,train,test,validate, clip_length, label=0)
    print("Noise people done")

    #noise_noise
    train['length'] = 3
    test['length'] = 2
    validate['length'] = 1
    class0_count += helper(noise_noise, train, test, validate, clip_length, label=0)
    print("Noise nosie done")


    print(len(train['X']),len(train['Y']))
    print(len(test['X']),len(test['Y']))
    print(len(validate['X']),len(validate['Y']))
    print(len(train['X']) + len(test['X']) +len(validate['X']))
    # print(len(train['labels']))
    # print(len(test['labels']))
    # print(len(validate['labels']))
    # print(train)
    # print(test)
    # print(validate)
    return {
        'train': (torch.tensor(np.array(train['X'])).unsqueeze(1), torch.tensor(np.array(train['Y']))),
        'test': (torch.tensor(np.array(test['X'])).unsqueeze(1), torch.tensor(np.array(test['Y']))),
        'validate': (torch.tensor(np.array(validate['X'])).unsqueeze(1), torch.tensor(np.array(validate['Y']))),
        "weight": class0_count / class1_count,
        'train_labels': train['labels'],
        'test_labels': test['labels'],
        'validate_labels': validate['labels'],
    }

"""
convert_to_spectrogram: Converts audio clips into normalized log-mel spectrograms.
Input: audio_clips (list), sr (int), clip_length (int)
Output: List of spectrogram arrays (list)
"""
def convert_to_spectrogram(audio_clips, sr, clip_length=3):
    #y_clips = preprocess_data(audio, sr, clip_length)
    spectrogram = []
    for sample in audio_clips:
        sample = fix_len(sample, sr*clip_length)
        s = librosa.feature.melspectrogram(y=sample, sr=sr, n_mels=128, )
        s_log = librosa.power_to_db(s, ref=np.max)
        s_norm = (s_log-MEAN)/STD
        spectrogram.append(s_norm)
    return spectrogram


##SPECTROGRAM UTIL
"""
save_spectrogram: Saves spectrogram data and labels into a compressed .npz file.
Input: path (str), target_dir (str), spectrogram (list/array), label (int)
Output: None (writes file to disk)
"""
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

"""
create_spectrogram_for_analysis: Generates spectrogram files from the dataset for offline analysis.
Input: clip_length (int)
Output: None (writes spectrogram dataset to ./spectrogram_data)
"""
def create_spectrogram_for_analysis(clip_length = 3):
    download_data()
    create_directories("./spectrogram_data")

    class0 = glob("./data/Class0/*/*.mp3")
    spectrogram_conversion_loop(class0, 0, clip_length)

    # augmented_class0 = glob("./data/Class0/*/*.mp3")[1::4]
    # spectrogram_conversion_loop(augmented_class0, 0, clip_length)

    noise = glob("./data/Random_Noise/*/*.mp3")
    spectrogram_conversion_loop(noise, 0, clip_length)

    class1 = glob("./data/Class1/*/*.mp3")
    spectrogram_conversion_loop(class1, 1, clip_length)

"""
spectrogram_conversion_loop: Converts audio files to spectrograms and saves them to disk.
Input: path_list (list), label (int), clip_length (int)
Output: None (writes spectrogram files)
"""
def spectrogram_conversion_loop(path_list, label, clip_length):
    for path in path_list:
        audio, sr = librosa.load(path)
        audio_clips = preprocess_data(audio, sr, clip_length)
        spectrogram = convert_to_spectrogram(audio_clips, sr)

        save_spectrogram(path, "./spectrogram_data", spectrogram, label)
        del audio, spectrogram
        gc.collect()

#print(random_data_split())