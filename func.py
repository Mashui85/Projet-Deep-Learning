import librosa
import numpy as np
import os
from os.path import basename, splitext, join
import random


def add_noise(s,noise,snr):
    """
    Ajoute du bruit au signal initial avec un SNR défini
    on ajoute une partie aléatoire du bruit pour éviter d'avoir tout le temps le même bruit et éviter l'overfitting
    """
    # SNR entre -10 et 10 dB

    s_len = len(s)
    n_len = len(noise)
    alpha = 10**(-snr/20)
    noise_start_indice = random.randint(0,n_len-s_len-1)
    noise = noise[noise_start_indice:noise_start_indice + s_len]
    x = s + alpha * noise

    return x



def STFTabs(y,hop_length,win_length,window,n_fft):
    D = librosa.stft(y,n_fft=n_fft,hop_length=hop_length,window=window,win_length=win_length, center=True)
    D_db = np.log(np.abs(D)**2)
    
    return D_db

def STFT_display():
    pass