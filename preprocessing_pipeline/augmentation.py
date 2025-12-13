import librosa
import numpy as np, random as rand


def time_stretch(audio_clips, sr, rate=0.1):
    if rate == 0:
        return []

    # augment = []
    # for audio in audio_clips:
    #     augment.append(librosa.effects.time_stretch(audio ,rate= 1 -rate))
    #     augment.append(librosa.effects.time_stretch(audio ,rate= 1 +rate))
    #
    # return augment
    l = len(audio_clips)
    for i in range(l):
        sign = -1 if rand.random() < 0.5 else 1
        audio_clips[i] = librosa.effects.time_stretch(audio_clips[i] ,rate= 1 + sign*rate)

        prob = rand.random()
        if  prob < 0.25:
            audio_clips.append(librosa.effects.time_stretch(audio_clips[i] ,rate=sign*0.2))
    return audio_clips

def pitch_shift(audio_clips, sr, pitch=0.5):
    if pitch == 0:
        return []

    pitch = abs(pitch)
    # augment = []
    # for audio in audio_clips:
    #     augment.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-pitch))
    #     augment.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch))
    #
    # return augment
    l = len(audio_clips)
    for i in range(l):
        sign = -1 if rand.random() < 0.5 else 1
        audio_clips[i] = librosa.effects.pitch_shift(audio_clips[i], sr=sr, n_steps=pitch * sign)
        prob = rand.random()
        if  prob < 0.25:
            audio_clips.append(librosa.effects.pitch_shift(audio_clips[i], sr=sr, n_steps=sign * 0.3))

    return audio_clips

def guass_noise(audio_clips, sr, snr_range=(20.0, 40.0)):
    for i in range(len(audio_clips)):
        snr_db = float(np.random.uniform(*snr_range))
        audio_clips[i] = add_gaussian_noise(audio_clips[i], snr_db=snr_db)

    return audio_clips

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

augmentations = {
    "time_stretch": {
        "fn": time_stretch,
        "count": 1
    },
    "pitch_shift": {
        "fn": pitch_shift,
        "count": 1
    },
    "guass_noise": {
        "fn": guass_noise,
        "count": 1
    },
}