import librosa
import numpy as np
import os
from os.path import basename, splitext, join
import random

def add_noise(s,noise,snr):
    """
    Mélange un signal propre `s` avec un bruit `noise` pour obtenir un SNR cible (en dB).

    - On extrait un segment aléatoire du bruit (même longueur que `s`) pour diversifier
      les réalisations et limiter l'overfitting.
    - On normalise le segment de bruit pour qu'il ait la même puissance moyenne que `s`,
      puis on applique un facteur `alpha = 10^{-snr/20}` pour obtenir le SNR souhaité.

    Parameters
    
    s : np.ndarray
        Signal de parole (1D).
    noise : np.ndarray
        Signal de bruit (1D), plus long que `s`.
    snr : float
        SNR cible en dB (plus grand = moins de bruit).

    Retourne
    x : np.ndarray
        Signal bruité, même longueur que `s`.
    """
    # Normalisation amplitude du bruit (évite des échelles extrêmes avant ajustement puissance)
    noise = noise/max(noise)
    s_len = len(s)
    n_len = len(noise)
    
    # Conversion SNR(dB) -> facteur multiplicatif sur le bruit
    
    alpha = 10**(-snr/20)
    
    # Prend un extrait aléatoire du bruit de même longueur que le signal propre
    
    noise_start_indice = random.randint(0,n_len-s_len-1)
    noise = noise[noise_start_indice:noise_start_indice + s_len]
    
    # Calcul des puissances moyennes (énergie moyenne) du signal et du bruit
    
    Ps = np.mean(s**2)
    Pn = np.mean(noise**2)
    
    # Sécurise le cas d'un bruit quasi nul (évite division par zéro)
    
    if Pn < 1e-12:
        noise = noise + 1e-6*np.random.randn(s_len)
        Pn = np.mean(noise**2)
    
    # Mise à l'échelle du bruit pour égaliser la puissance avec celle du signal
    
    noise = noise * np.sqrt(Ps / Pn)
    
    # Mélange final : s + alpha * noise (alpha impose le SNR)
    
    x = s + alpha * noise

    return x

def data_sized(x,time_limit):
    """
    Met un signal audio à une durée fixe `time_limit` (en secondes) à Fs=16 kHz.

    - Si le signal est plus court : padding par zéros.
    - Si le signal est plus long : découpe en segments contigus de durée fixe.
      Le reste (partie non multiple) est ignoré.

    Parameters
    x : np.ndarray
        Signal audio 1D.
    time_limit : float
        Durée cible en secondes.

    Returns
    x_sized : np.ndarray
        - 1D si padding (len = Fs*time_limit)
        - 2D si découpage (shape = (q, Fs*time_limit)), avec q = nombre de segments
    """
    Fs = 16000
    
    # nombre d'échantillons correspondant à time_limit
    
    sample_limit = Fs * time_limit
    
    # Si le signal est trop court : padding pour obtenir exactement sample_limit

    if len(x) < sample_limit:
        x_sized = np.concatenate((x,np.zeros(sample_limit - len(x))))
    
    # Sinon : découpage en segments entiers de longueur sample_limit
    
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

