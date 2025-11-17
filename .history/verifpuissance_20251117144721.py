import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from func import STFTabs, add_noise
from load_file import load_file
import librosa
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from modele import train_test_separation, train, test_estimation
from func import STFT_display, add_noise, data_sized

signals_sized = []
S_list = []
paths, signals, sr_list = load_file()
data_size = 5
fs=16000
n_fft = 1024
hop_length = int(n_fft * 0.2)
window = 'hann'
win_length = n_fft
test_size = 0.2
for i in range(len(signals)//10):
    d = data_sized(signals[i],data_size)
    if len(d)>100:
        signals_sized.append(d)
    else:    
        for j in d:
            signals_sized.append(j) 
for i in range(len(signals_sized)):
    S_list.append( STFTabs(signals_sized[i], hop_length, win_length, window, n_fft))

u,fs = librosa.load('babble_16k.wav',sr=fs)
U = STFTabs(u,hop_length,win_length,window,n_fft)/90
x_list = []
X_list = []
for i in range(len(signals_sized)):
    x_list.append(add_noise(signals_sized[i],u,-2))