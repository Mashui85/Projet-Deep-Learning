import librosa
import numpy as np
import os
from os.path import basename, splitext, join
import random
import yann as pd

def add_noise(s,noise,snr):
    """
    Ajoute du bruit au signal initial avec un SNR défini
    on ajoute une partie aléatoire du bruit pour éviter d'avoir tout le temps le même bruit et éviter l'overfitting
    """
    # SNR entre -10 et 10 dB
    #ajouter vérification de puissance équivalente entre bruit et signal
    noise = noise/max(noise)
    s_len = len(s)
    n_len = len(noise)
    alpha = 10**(-snr/20)
    noise_start_indice = random.randint(0,n_len-s_len-1)
    noise = noise[noise_start_indice:noise_start_indice + s_len]
    x = s + alpha * noise

    return x

def data_sized(x,time_limit):
    """Fonction qui met la taille des signaux à time_limit sec
    Ajoute des zéros si taille< time_limit sec et coupe le signal si > time_limit sec
    """
    Fs = 16000
    sample_limit = Fs * time_limit
    min_limit = sample_limit/2

    if len(x) < sample_limit:
        x_sized = x + np.zeros((sample_limit - len(x)),1)
    else:
        q = len(x)//sample_limit
        x_sized = x[:q*sample_limit]
        x_sized.reshape(q,sample_limit)
    return x_sized




def STFTabs(y,hop_length,win_length,window,n_fft):
    D = librosa.stft(y,n_fft=n_fft,hop_length=hop_length,window=window,win_length=win_length, center=True)
    D_db = np.log(np.abs(D)**2)
    
    return D_db

def STFT_display():
    pass

