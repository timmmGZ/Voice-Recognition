import librosa
import numpy as np


def mel_to_frames(mel, frame_len=30, strides=8):
    frames = []
    for width in range(0, mel.shape[1], strides):
        frame = mel[:, width:width + frame_len]
        if frame.shape[1] < frame_len:
            break
        frames.append(frame)
    return np.array(frames)


def non_zero_hann(n_fft):
    return np.array([1e-5] + [0.5 * (1 - np.cos(2 * np.pi * i / (n_fft - 1))) for i in range(1, n_fft - 1)] + [1e-5])


def get_mel(x, fs, winLen=0.02):
    frame_length = winLen * fs
    frame_length = int(2 ** int(frame_length - 1).bit_length())
    hop_length = frame_length // 2
    S = librosa.feature.melspectrogram(y=x, win_length=frame_length, hop_length=hop_length, window=non_zero_hann, sr=fs,
                                       n_mels=128)
    S_dB = librosa.power_to_db(S, ref=102400000000)
    return S_dB
