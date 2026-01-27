import librosa
import numpy as np, random as rand
from scipy.signal import fftconvolve

"""
generate_rir: Generates a random room impulse response for reverberation simulation.
Input: length (int), decay (float)
Output: Normalized RIR signal (numpy array)
"""
def generate_rir(length=256, decay=0.5):
    rir = np.random.randn(length)
    rir *= np.exp(-np.linspace(0, decay, length))  # exponential decay
    rir = rir / (np.sqrt(np.sum(rir**2)) + 1e-12)  # energy-normalize
    return rir

RIR = [generate_rir(length=np.random.randint(120, 280), decay=np.random.uniform(0.3, 0.7)) for _ in range(12)]

"""
reverb: Applies reverberation to an audio signal using a random impulse response.
Input: audio (numpy array)
Output: Reverberated audio signal (numpy array)
"""
def reverb(audio):
    rir = rand.choice(RIR)
    reverbed = fftconvolve(audio, rir, mode="full")
    reverbed = reverbed[:len(audio)]

    # normalize to avoid clipping
    peak = np.max(np.abs(reverbed)) + 1e-12
    if peak > 1.0:
        reverbed /= peak
    return reverbed

"""
reverberation: Applies reverberation augmentation to a list of audio clips and may generate additional augmented clips.
Input: audio_clips (list of numpy arrays), sr (int)
Output: Augmented audio clips with reverberation applied (list)
"""
def reverberation(audio_clips, sr):
    l = len(audio_clips)
    for i in range(l):
        # if rand.random() < 0.5:
        #     audio_clips.append(reverb(audio_clips[i].copy()))

        # audio_clips[i] = reverb(audio_clips[i])
        audio_clips.append(reverb(audio_clips[i].copy()))

    return audio_clips

"""
time_stretch: Randomly speeds up or slows down audio clips for augmentation and may generate additional augmented clips.
Input: audio_clips (list), sr (int), rate (float)
Output: Time-stretched audio clips (list)
"""
def time_stretch(audio_clips, sr, rate=0.1):
    if rate==0:
        return audio_clips
    # augment = []
    # for audio in audio_clips:
    #     augment.append(librosa.effects.time_stretch(audio ,rate= 1 -rate))
    #     augment.append(librosa.effects.time_stretch(audio ,rate= 1 +rate))
    #
    # return augment
    l = len(audio_clips)
    for i in range(l):
        sign = -1 if rand.random() < 0.5 else 1
        prob = rand.random()
        orig = audio_clips[i].copy()
        # if prob < 0.5:
        #     audio_clips.append(librosa.effects.time_stretch(orig, rate=1 + sign * 0.2))

        # audio_clips[i] = librosa.effects.time_stretch(orig, rate= 1 + sign*rate)
        audio_clips.append(librosa.effects.time_stretch(orig, rate=1 + sign * 0.2))
    return audio_clips

"""
pitch_shift: Shifts audio pitch up or down for augmentation and may generate additional augmented clips.
Input: audio_clips (list), sr (int), pitch (float)
Output: Pitch-shifted audio clips (list)
"""
def pitch_shift(audio_clips, sr, pitch=0.5):
    if pitch==0:
        return audio_clips

    pitch = abs(pitch)
    # augment = []
    # for audio in audio_clips:
    #     augment.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-pitch))
    #     augment.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch))
    #
    # return augment
    l = len(audio_clips)
    for i in range(l):
        orig = audio_clips[i].copy()
        sign = -1 if rand.random() < 0.5 else 1
        prob = rand.random()
        # if prob < 0.5:
        #     audio_clips.append(librosa.effects.pitch_shift(orig, sr=sr, n_steps=sign * 0.3))

        # audio_clips[i] = librosa.effects.pitch_shift(orig, sr=sr, n_steps=pitch * sign)
        audio_clips.append(librosa.effects.pitch_shift(orig, sr=sr, n_steps=sign * 0.3))

    return audio_clips

"""
guass_noise: Adds Gaussian noise to each audio clip with random SNR.
Input: audio_clips (list), sr (int), snr_range (tuple)
Output: Noisy audio clips (list)
"""
def guass_noise(audio_clips, sr, snr_range=(20.0, 40.0)):
    for i in range(len(audio_clips)):
        snr_db = float(np.random.uniform(*snr_range))
        audio_clips[i] = add_gaussian_noise(audio_clips[i], snr_db=snr_db)

    return audio_clips

"""
add_gaussian_noise: Adds Gaussian noise to a signal at a target SNR level.
Input: signal (numpy array), snr_db (float)
Output: Noisy signal (numpy array)
"""
def add_gaussian_noise(signal, snr_db):
    sig_rms =  np.sqrt(np.mean(signal**2) + 1e-12)
    # amplitude ratio from dB
    ratio = 10.0 ** (snr_db / 20.0)
    noise_rms = sig_rms / (ratio + 1e-12)

    noise = np.random.randn(len(signal))
    noise = noise * (noise_rms / (np.sqrt(np.mean(noise**2) + 1e-12) + 1e-12))

    noisy = signal + noise
    # soft peak normalization to avoid clipping
    peak = np.max(np.abs(noisy))
    if peak > 1.0:
        noisy = noisy / peak
    return noisy

# augmentations = {
#     "time_stretch": {
#         "fn": time_stretch,
#        # "count": 1,
#         "prob": [0.1, 0.3]
#     },
#     "pitch_shift": {
#         "fn": pitch_shift,
#         #"count": 1,
#         "prob": [0.05, 0.3]
#     },
#     "guass_noise": {
#         "fn": guass_noise,
#         #"count": 1,
#         "prob": [0.2, 0.3]
#     },
#     "reverberation": {
#         "fn": reverberation,
#         #"count": 1,
#         "prob": [0.1, 0.4]
#     }
# }

augmentations = {
    
    "guass_noise": {
        "fn": guass_noise,
        #"count": 1,
        "prob": [0.2, 0.3]
    },

    # "reverberation": {
    #     "fn": reverberation,
    #     #"count": 1,
    #     "prob": [0.1, 0.4]
    # }
    
    
}

