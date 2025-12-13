import numpy as np
from glob import glob
import librosa
import librosa.display
from preprocessing_pipeline.augmentation import augmentations
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

#save spectograms in .npz format
RMS = None

def get_rms():
    global RMS
    if RMS is not None:
        return RMS
    files = glob("./data/*/*/*.mp3")
    total = 0
    samples = 0
    for file in files:
        y, sr = librosa.load(file, sr=None)
        total += np.sum(y ** 2)
        samples += len(y)
    RMS = np.sqrt(total / samples)
    return np.sqrt(total / samples)


def rms_normalize(audio, rms):
    clip_rms = np.sqrt(np.mean(audio ** 2))
    scale = rms / clip_rms if clip_rms != 0 else rms
    return audio * scale


# preprocessing - normalize, trim, crop into 1 min audios, split into 3s clips
def preprocess_data(audio, sr, clip_length, rms=True, augment=False, label=0):
    # y, sr = librosa.load(audio) #NOTE: librosa.load by default standardizes the sr to 22050 HZ
    y_norm = rms_normalize(audio, get_rms()) if rms else librosa.util.normalize(audio)

    y_trimmed, _ = librosa.effects.trim(y_norm, top_db=20)
    intervals = librosa.effects.split(y_trimmed, top_db=20)
    y_no_silence = np.concatenate([y_trimmed[interval[0]:interval[1]] for interval in intervals])
    clip_length_samples = clip_length * sr
    y_clips = librosa.util.frame(y_no_silence, frame_length=clip_length_samples,
                                 hop_length=clip_length_samples).T.copy()
    y_clips = [np.array(c) for c in y_clips]

    if augment:
        aug_cpy = {k: {"fn": v["fn"], "count": v["count"]} for k, v in augmentations.items()}
        for i in range(len(aug_cpy)):
            if rand.random() < 0.4 if label == 0 else 0.7:
                a = rand.choice([aug for aug in aug_cpy.values() if aug["count"] != 0])
                a["count"] -= 1
                aug_specs = a["fn"](y_clips, sr)
                if len(aug_specs) != len(y_clips):
                    y_clips.extend(aug_specs)

    return y_clips

#DEPRECATED
# def create_raw_spectrogram():
#     dowload_data()
#     create_directories("./raw_spectrogram")
#     class0 = glob("./data/Class0/*/*.mp3")
#     for path in class0:
#         y, sr = librosa.load(path)
#         S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, )
#         spectrogram = librosa.amplitude_to_db(S, ref=np.max)
#         save_spectrogram(path, "./raw_spectrogram", spectrogram, 0)
#
#     class1 = glob("./data/Class1/*/*.mp3")
#     for path in class1:
#         y, sr = librosa.load(path)
#         S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, )
#         spectrogram = librosa.amplitude_to_db(S, ref=np.max)
#         save_spectrogram(path, "./raw_spectrogram", spectrogram, 1)
#
# def spectrogram_conversion_loop(path_list, label, clip_length, pitch=False):
#     for path in path_list:
#         audio, sr = librosa.load(path)
#         if pitch:
#             spectrogram = convert_with_pitch_shift(audio, sr, clip_length)
#         else:
#             spectrogram = convert_to_spectograms(audio, sr, clip_length)
#
#         save_spectrogram(path, "./spectogram_data", spectrogram, label if pitch==False else 0)
#         del audio, spectrogram
#         gc.collect()
#
# #function used to process gathered data
# def create_spectrogram_from_data(clip_length):
#     dowload_data()
#     create_directories("./spectogram_data")
#
#     class0 = glob("./data/Class0/*/*.mp3")
#     spectrogram_conversion_loop(class0, 0, clip_length)
#     class1 = glob("./data/Class1/*/*.mp3")
#
#     # augmented_class0 = glob("./data/Class0/*/*.mp3")[1::4]
#     # spectrogram_conversion_loop(augmented_class0, 0, clip_length)
#
#     noise = glob("./data/Random_Noise/*/*.mp3")
#     spectrogram_conversion_loop(noise, 0, clip_length)
#
#     class1 = glob("./data/Class1/*/*.mp3")
#     spectrogram_conversion_loop(class1, 1, clip_length)
# create_spectrogram_from_data(3)
#create_raw_spectrogram()