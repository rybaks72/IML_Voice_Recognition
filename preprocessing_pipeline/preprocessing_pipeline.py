import numpy as np
from glob import glob
import librosa
import librosa.display
import os
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


"""
save_spectrogram: Saves a spectrogram and its label into a compressed .npz file.
Input: path (str), target_dir (str), spectrogram (array-like), label (int)
Output: None (writes file to disk)
"""
def save_spectrogram(path,target_dir, spectrogram, label):
    path, _ = os.path.splitext(path)
    path = path.split("\\")
    name = path[-1] + ".npz"
    path.remove(path[-1])
    path.append(name.lower())
    path[-2] = path[-2].lower()
    path = path[2:]
    path = "\\".join([target_dir, *path])
    X = np.array(spectrogram)
    Y = np.array([label] * len(spectrogram))
    # print(f"X shape: {X.shape}")
    # print(f"Y shape: {Y.shape}")

    np.savez_compressed(path, X=X, Y=Y)

"""
create_raw_spectrogram: Converts all audio files into raw mel spectrograms and saves them.
Input: None
Output: None (writes spectrogram files to ./raw_spectrogram)
"""
def create_raw_spectrogram():
    dowload_data()
    create_directories(".\\raw_spectrogram")
    class0 = glob(".\\data\\Class0\\*\\*.mp3")
    for path in class0:
        y, sr = librosa.load(path)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, )
        spectrogram = librosa.amplitude_to_db(S, ref=np.max)
        save_spectrogram(path, ".\\raw_spectrogram", spectrogram, 0)

    class1 = glob(".\\data\\Class1\\*\\*.mp3")
    for path in class1:
        y, sr = librosa.load(path)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, )
        spectrogram = librosa.amplitude_to_db(S, ref=np.max)
        save_spectrogram(path, ".\\raw_spectrogram", spectrogram, 1)

"""
pitched_path: Generates a modified file path for pitch-augmented spectrogram outputs.
Input: path (str), prefix (str)
Output: New augmented path (str)
"""
def pitched_path(path, prefix):
    arr = path.split("\\")
    arr[2] = "Random_Noise"
    arr[3] = "people"
    arr[4] = prefix + arr[4]
    arr = ".\\".join(arr)
    return arr

"""
spectrogram_conversion_loop: Converts a list of audio files into spectrograms and saves them.
Input: path_list (list), label (int), clip_length (int), pitch (bool)
Output: None (writes spectrogram files to disk)
"""
def spectrogram_conversion_loop(path_list, label, clip_length, pitch=False, prefix=''):
    for path in path_list:
        audio, sr = librosa.load(path)
        if pitch:
            spectrograms = convert_with_pitch_shift(audio, sr, clip_length)
            for i in range(len(spectrograms)):
                save_spectrogram(pitched_path(path, f"pitched{i}_"), ".\\spectogram_data", spectrograms[i], label if pitch == False else 0)

        else:
            spectrogram = convert_to_spectograms(audio, sr, clip_length)
            save_spectrogram(path, ".\\spectogram_data", spectrogram, label if pitch==False else 0)
            del spectrogram

        del audio; gc.collect()




"""
create_spectrogram_from_data: Builds the full spectrogram dataset from all audio files.
Input: clip_length (int)
Output: None (writes processed spectrogram dataset to ./spectogram_data)
"""
def create_spectrogram_from_data(clip_length):
    #dowload_data()
    create_directories(".\\spectogram_data")

    class0 = glob(".\\data\\Class0\\*\\*.mp3")
    spectrogram_conversion_loop(class0, 0, clip_length)
    print("class0 done")

    augmented_class0 = glob(".\\data\\Class0\\*\\*.mp3")[0::4]
    #spectrogram_conversion_loop(augmented_class0, 0, clip_length, pitch=True)
    print("augmented class0 done")

    noise = glob(".\\data\\Random_Noise\\*\\*.mp3")
    spectrogram_conversion_loop(noise, 0, clip_length)
    print("noise done")

    noise_ppl_aug = glob(".\\data\\Random_Noise\\people\\*.mp3")
    #spectrogram_conversion_loop(noise_ppl_aug, 0, clip_length, pitch=True)
    print("augmented noise done")

    class1 = glob(".\\data\\Class1\\*\\*.mp3")
    spectrogram_conversion_loop(class1, 1, clip_length)
    print("class1 done")

    class1_aug = glob(".\\data\\Class1\\*\\*.mp3")[0::15]
    #spectrogram_conversion_loop(class1_aug, 0, clip_length, pitch=True)
    print("class1 done")


create_spectrogram_from_data(3)
#create_raw_spectrogram()