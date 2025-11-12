import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from func import STFTabs, add_noise
from load_file import load_file
import librosa
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from load_file import load_file
from modele import train_test_separation, train, test_estimation
from load_file import load_file
from func import STFT_display


def power(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:               # stéréo -> mono
        x = x.mean(axis=0)
    return float(np.mean(x**2))  # puissance moyenne

# Bruit
u, fs = librosa.load('babble_16k.wav', sr=16000)
u = u/max(u)
Pu = power(u)

# Signaux (liste de tableaux 1D potentiellement de longueurs différentes)
paths, signals, sr_list = load_file()

Ps_list = [power(s/max(s)) for s in signals]   # puissance par signal
Psmax = max(Ps_list)
Psmin = min(Ps_list)
Ps = float(np.mean(Ps_list))            # moyenne sur l’ensemble

print("Pu =", Pu)
print("Ps =", Ps) 
print("puissance max dans les signaux = ", Psmax)
print("puissance minimale=", Psmin)