import numpy as np
from glob import glob
import librosa
import librosa.display
from preprocessing_pipeline.augmentation import augmentations
import random as rand

#save spectograms in .npz format
RMS = None

"""
get_rms: Computes and caches the global RMS loudness across all dataset audio files.
Input: None
Output: Global RMS value (float)
"""
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


"""
rms_normalize: Scales an audio clip to match a target RMS loudness level.
Input: audio (numpy array), rms (float)
Output: Loudness-normalized audio signal (numpy array)
"""
def rms_normalize(audio, rms):
    clip_rms = np.sqrt(np.mean(audio ** 2))
    scale = rms / clip_rms if clip_rms != 0 else rms
    scale = np.clip(scale, 0.5, 3)
    return audio * scale


"""
preprocess_data: Normalizes audio, removes silence, splits into fixed-length clips, and optionally applies augmentations.
Input: audio (numpy array), sr (int), clip_length (int), rms (bool), augment (bool), label (int)
Output: List of processed (and possibly augmented) audio clips (list of numpy arrays)
"""
def preprocess_data(audio, sr, clip_length, rms=True, augment=False, label=0):
    # y, sr = librosa.load(audio) #NOTE: librosa.load by default standardizes the sr to 22050 HZ
    y_norm = rms_normalize(audio, get_rms()) if rms else librosa.util.normalize(audio)

    y_trimmed, _ = librosa.effects.trim(y_norm, top_db=30)
    intervals = librosa.effects.split(y_trimmed, top_db=30)
    y_no_silence = np.concatenate([y_trimmed[interval[0]:interval[1]] for interval in intervals])
    clip_length_samples = clip_length * sr
    y_clips = librosa.util.frame(y_no_silence, frame_length=clip_length_samples,
                                 hop_length=clip_length_samples).T.copy()
    y_clips = [np.array(c) for c in y_clips]

    if augment:
        if label == 0:
            num_aug = np.random.choice([0, 1], p=[0.8, 0.2])
        else:
            num_aug = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])

        aug_candidates = [
            aug for aug in augmentations.values()
            if rand.random() < aug["prob"][label]
        ]
        
        selected = rand.sample(
            aug_candidates,
            min(num_aug, len(aug_candidates))
        )

        for aug in selected:
            y_clips = aug["fn"](y_clips, sr)

        #OLD
        # aug_cpy = {k: {"fn": v["fn"], "count": v["count"]} for k, v in augmentations.items()}
        # for i in range(len(aug_cpy)):
        #     a = rand.choice([aug for aug in aug_cpy.values() if aug["count"] != 0])
        #     if rand.random() < a["prob"][label]:
        #         a["count"] -= 1
        #         aug_specs = a["fn"](y_clips, sr,label)
        #         if len(aug_specs) != len(y_clips):
        #             y_clips.extend(aug_specs)
    return y_clips
