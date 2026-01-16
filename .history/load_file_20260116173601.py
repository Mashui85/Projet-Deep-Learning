# CONFIGURATION
import librosa
import os
from os.path import basename, splitext, join
from pathlib import Path
import numpy as np

#Charge les donnees des fichiers audio et les normalise
def load_file():
    """
    Charge des fichiers audio depuis un répertoire source (dataset) et les normalise.

    - Recherche récursive des fichiers audio (wav/mp3/flac) dans `source_audio_dir`.
    - Charge chaque fichier avec librosa à une fréquence d'échantillonnage cible Fe=16 kHz
      (resampling si nécessaire).
    - Normalise chaque signal en amplitude (division par max(|y|)) afin d'obtenir une échelle
      comparable entre fichiers.

    Remarques
    - La normalisation ici est une normalisation "peak" (amplitude max), pas une normalisation
      énergétique/RMS.
    - `output_dir` est défini mais n'est pas utilisé dans cette fonction (reste d'une étape
      précédente / futur usage).

    Retourne
    paths : list[str]
        Chemins des fichiers audio trouvés.
    signals : list[np.ndarray]
        Signaux audio mono normalisés (1D) à Fe=16 kHz.
    sr_list : list[int]
        Fréquences d'échantillonnage (vaut 16000 après re-echantillonage).
    """
    # Répertoire dataset : recherche récursive de fichiers audio
    source_audio_dir = "C:/Users/USER/Documents/Phelma/3A/LibriSpeech/dev-clean"
    paths = [str(p) for p in Path(source_audio_dir).rglob("*")
            if p.suffix.lower() in (".wav", ".mp3", ".flac")]
    # Indication du volume de données chargé (utile pour le debug)
    print("Fichiers trouvés :", len(paths))
    Fe = 16000          # fréquence d'échantillonnage cible
    signals = []
    sr_list = [] 

    for path in paths:
        # Chargement + reechantillonage à Fe
        y, sr = librosa.load(path, sr=Fe)
        # Normalisation en amplitude pour éviter de grosses variations d'échelle entre fichiers (robuste au silence)
        peak = np.max(np.abs(y))
        if peak > 1e-12:
            y = y / peak
        signals.append(y)
        sr_list.append(sr)
    return paths, signals, sr_list



