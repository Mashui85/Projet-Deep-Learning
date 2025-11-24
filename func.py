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
    #ajouter vérification de puissance équivalente entre bruit et signal
    noise = noise/max(noise)
    s_len = len(s)
    n_len = len(noise)
    alpha = 10**(-snr/20)
    noise_start_indice = random.randint(0,n_len-s_len-1)
    noise = noise[noise_start_indice:noise_start_indice + s_len]
    Ps = np.mean(s**2)
    Pn = np.mean(noise**2)
    if Pn < 1e-12:
        noise = noise + 1e-6*np.random.randn(s_len)
        Pn = np.mean(noise**2)
    noise = noise * np.sqrt(Ps / Pn)
    #print("Ps=", Ps) Verfication
    #print("nouvelle puissance du bruit = ", np.mean(noise**2))
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
        x_sized = np.concatenate((x,np.zeros(sample_limit - len(x))))
    else:
        q = len(x)//sample_limit
        x_sized = x[:q*sample_limit]
        x_sized = x_sized.reshape((q,sample_limit))
    return x_sized




def STFTabs(y,hop_length,win_length,window,n_fft):
    D = librosa.stft(y,n_fft=n_fft,hop_length=hop_length,window=window,win_length=win_length, center=True)
    D_db = np.log((np.abs(D)+1e-6)**2)
    
    return D_db

def STFTabs_phase(y,hop_length,win_length,window,n_fft):
    D = librosa.stft(y,n_fft=n_fft,hop_length=hop_length,window=window,win_length=win_length, center=True)
    D_db = np.log((np.abs(D)+1e-6)**2)
    
    return D_db,np.exp(1j * np.angle(D))

def STFT_display():
    pass

