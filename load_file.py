# CONFIGURATION
import librosa
import os
from os.path import basename, splitext, join
from pathlib import Path
import numpy as np

#Charge les donnees des fichiers audio et les normalise
def load_file():
    output_dir = "./res/" #Output directory
    source_audio_dir = "C:/Users/Matthias/Documents/Cours/Phelma/3A/Projet Deep Learning/LibriSpeech/dev-clean"
    paths = [str(p) for p in Path(source_audio_dir).rglob("*")
            if p.suffix.lower() in (".wav", ".mp3", ".flac")]

    print("Fichiers trouvés :", len(paths))
    Fe = 16000
    signals = []
    sr_list = [] 

    for path in paths:
        y, sr = librosa.load(path, sr=Fe)
        y = y / np.max(np.abs(y))
        signals.append(y)
        sr_list.append(sr)
    return paths, signals, sr_list



