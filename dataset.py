import glob
import os
import zipfile

import gdown
import numpy as np
import pandas as pd
from librosa.effects import trim
from scipy.io import wavfile
from tqdm import tqdm

from mel import get_mel, mel_to_frames


def download():
    if not os.path.exists("voice_recognition"):
        gdown.download_folder("https://drive.google.com/drive/folders/1-2oEmeKKRkSJ1xwAhtFX9mXdE1iNZpDZ")


def unzip():
    if not os.path.exists("dataset"):
        with zipfile.ZipFile("voice_recognition/voice_recognition.zip", 'r') as zip_ref:
            zip_ref.extractall(path="dataset")


def get_wavs(files):
    wavs = []
    for file in tqdm(files):
        fs, wav = wavfile.read(file)
        wav = wav.astype(np.float32)
        wav, l = trim(wav, top_db=25)
        wavs.append(wav)
    return np.array(wavs)


def get_dataset():
    download()
    unzip()
    wav_train_files = glob.glob('dataset/trainset/*.wav')
    wav_test_files = glob.glob('dataset/testset/*.wav')
    labels = pd.read_csv('dataset/labels.csv')
    return wav_train_files, wav_test_files, labels


def getXy(files, labels, wavs):
    X, y = [], []
    for i in tqdm(range(len(files))):
        fileID = files[i].split('\\')[-1]
        fileID = fileID.split('/')[-1]
        yi = [labels[labels['File ID'] == fileID].values[0][1]]
        mel = get_mel(wavs[i], 22050)
        mel_frames = mel_to_frames(mel)
        X.extend(mel_frames)
        y.extend(yi * len(mel_frames))
    return np.array(X), np.array(y)


def load_and_preprocess_dataset():
    if not os.path.exists("dataset/Xy.npz"):
        wav_train_files, wav_test_files, labels = get_dataset()
        wavs_train = get_wavs(wav_train_files)
        wavs_test = get_wavs(wav_test_files)
        fs, noise = wavfile.read('dataset/noise/noise.wav')
        fs, not_voice = wavfile.read('dataset/noise/not_voice.wav')
        noise = np.concatenate([noise, not_voice]).astype(np.float32).mean(axis=1)
        mel = get_mel(noise, fs)
        X_noise = mel_to_frames(mel)
        X_train, y_train = getXy(wav_train_files, labels, wavs_train)
        X_train = np.concatenate([X_train, X_noise], axis=0)
        y_noise = ["Not speaking"] * len(X_noise)
        y_train = np.concatenate([y_train, y_noise], axis=0)
        ids = np.random.permutation(len(X_train))
        X_train, y_train = X_train[ids], y_train[ids]
        X_test, y_test = getXy(wav_test_files, labels, wavs_test)

        train_mean = X_train.mean()
        train_std = X_train.std()
        X_train -= train_mean  # Using X_train = (X_train - train_mean) / train_std
        X_train /= train_std  # can result in "Unable to allocate xxx MiB for an array with shape xxx"
        X_test -= train_mean
        X_test /= train_std
        print("Saving as .npz")
        np.savez_compressed('dataset/Xy', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                            train_mean=train_mean, train_std=train_std)
        print("Saving done")
    else:
        Xy = np.load('dataset/Xy.npz')
        X_train = Xy['X_train']
        X_test = Xy['X_test']
        y_train = Xy['y_train']
        y_test = Xy['y_test']
        train_mean = Xy['train_mean']
        train_std = Xy['train_std']
    return X_train, X_test, y_train, y_test, train_mean, train_std
