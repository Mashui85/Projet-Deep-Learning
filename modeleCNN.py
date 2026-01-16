import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import matplotlib.pyplot as plt

from func import STFTabs, add_noise, data_sized, STFTabs_phase
from load_file import load_file


# -------------------------
# Utils: save curves + csv
# -------------------------
def ensure_dir(path: str):

    os.makedirs(path, exist_ok=True)

def save_losses_csv(save_dir: str, train_losses, test_losses):
    """
    Sauvegarde les courbes d'apprentissage dans un fichier CSV.

    Cette fonction écrit, pour chaque epoch, la valeur de la perte sur
    l'ensemble d'entraînement et sur l'ensemble de test. Le fichier généré
    permet de tracer a posteriori les courbes d'apprentissage sans relancer
    l'entraînement.

    Paramètres
    ----------
    save_dir : str
        Dossier dans lequel le fichier CSV est enregistré.
    train_losses : list of float
        Liste des valeurs de la fonction de coût sur le jeu d'entraînement,
        indexées par epoch.
    test_losses : list of float
        Liste des valeurs de la fonction de coût sur le jeu de test,
        indexées par epoch.

    Retour
    ------
    csv_path : str
        Chemin vers le fichier CSV généré ("losses.csv").
    """
    ensure_dir(save_dir)
    csv_path = os.path.join(save_dir, "losses.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss"])
        for i, (tr, te) in enumerate(zip(train_losses, test_losses), start=1):
            writer.writerow([i, tr, te])
    return csv_path

def plot_and_save_losses(save_dir: str, train_losses, test_losses):
    """
    Trace et sauvegarde les courbes d'apprentissage (train et test).

    Cette fonction génère une figure représentant l'évolution de la fonction
    de coût sur l'ensemble d'entraînement et sur l'ensemble de test en fonction
    du nombre d'epochs. La figure est sauvegardée au format PNG afin de pouvoir
    être exploitée ultérieurement (rapport, analyse des performances).

    Paramètres
    ----------
    save_dir : str
        Dossier dans lequel la figure est enregistrée.
    train_losses : list of float
        Valeurs de la perte sur le jeu d'entraînement pour chaque epoch.
    test_losses : list of float
        Valeurs de la perte sur le jeu de test pour chaque epoch.

    Retour
    ------
    png_path : str
        Chemin vers le fichier image généré ("learning_curves.png").
    """
    ensure_dir(save_dir)
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning curves")
    plt.legend()
    plt.grid(True)

    png_path = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    return png_path


# 1. Séparation train / test
def train_test_separation():
        """
    Prépare les jeux de données d'entraînement et de test pour le débruitage de parole.

    Cette fonction charge des signaux de parole propres, les découpe en segments
    temporels de durée fixe, puis génère des versions bruitées en ajoutant un bruit
    de type babble à des rapports signal-sur-bruit (SNR) aléatoires. Les signaux
    propres et bruités sont ensuite transformés dans le domaine temps-fréquence
    à l'aide d'une STFT, normalisés et utilisés pour construire les paires
    (entrée bruitée, cible propre).

    Les données sont finalement séparées en jeux d'entraînement et de test.

    Étapes principales
    ------------------
    1. Chargement des signaux de parole.
    2. Découpage en segments de durée fixe (5 secondes).
    3. Filtrage des segments trop courts ou trop silencieux.
    4. Ajout de bruit babble avec un SNR aléatoire.
    5. Calcul des spectrogrammes (STFT), normalisation et clipping.
    6. Séparation aléatoire en ensembles d'entraînement et de test.

    Paramètres
    ----------
    Aucun paramètre d'entrée. Les paramètres de traitement (fréquence
    d'échantillonnage, taille de fenêtre, durée des segments, SNR) sont
    définis en interne.

    Retour
    ------
    X_train : torch.Tensor
        Spectrogrammes bruités normalisés du jeu d'entraînement,
        de dimension (N_train, F, T).
    X_test : torch.Tensor
        Spectrogrammes bruités normalisés du jeu de test,
        de dimension (N_test, F, T).
    y_train : torch.Tensor
        Spectrogrammes propres normalisés correspondant au jeu d'entraînement.
    y_test : torch.Tensor
        Spectrogrammes propres normalisés correspondant au jeu de test.
    x_list : list of np.ndarray
        Liste des signaux temporels bruités générés (utile pour des tests
        ou visualisations ultérieures).
    """
    fs = 16000
    n_fft = 1024
    hop_length = int(n_fft * 0.2)
    window = 'hann'
    win_length = n_fft
    test_size = 0.2
    data_size = 5  # durée des segments
    paths, signals, sr_list = load_file()

    signals_sized = []

    # On découpe les signaux
    for i in range(len(signals) // 10):
        d = data_sized(signals[i], data_size)
        if isinstance(d, np.ndarray):
            d = [d]
        elif isinstance(d, list) and len(d) > 0 and np.isscalar(d[0]):
            d = [np.array(d)]
        for seg in d:
            signals_sized.append(seg)

    # Bruit babble
    u, fs = librosa.load('babble_16k.wav', sr=fs)

    S_list = []
    X_list = []
    x_list = []

    for seg in signals_sized:
        seg = np.asarray(seg)
        if seg.ndim != 1:
            continue
        if len(seg) < fs:   # < 1s
            continue
        if np.mean(seg**2) < 1e-6:  # trop silencieux
            continue

        # Clean
        S = STFTabs(seg, hop_length, win_length, window, n_fft) / 90.0
        S = np.clip(S, -3.0, 1.0)

        # Noisy
        snr_db = np.random.uniform(5, 15)
        x_noisy = add_noise(seg, u, snr_db)
        X = STFTabs(x_noisy, hop_length, win_length, window, n_fft) / 90.0
        X = np.clip(X, -3.0, 1.0)

        S_list.append(S)
        X_list.append(X)
        x_list.append(x_noisy)

    S_array = np.array(S_list)  # (N, F, T)
    X_array = np.array(X_list)  # (N, F, T)

    X_train, X_test, y_train, y_test = train_test_split(
        X_array, S_array, test_size=test_size, random_state=42, shuffle=True
    )

    X_train = torch.from_numpy(X_train.astype(np.float32))  # (N, F, T)
    X_test  = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test  = torch.from_numpy(y_test.astype(np.float32))

    return X_train, X_test, y_train, y_test, x_list


# 2. CNN 2D pour débruitage
class CnnDenoiser(nn.Module):
    def __init__(self, n_channels=32, n_blocks=4):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        )

        blocks = []
        for _ in range(n_blocks):
            blocks.append(self._res_block(n_channels))
        self.blocks = nn.Sequential(*blocks)

        self.out = nn.Conv2d(n_channels, 1, kernel_size=1)

    def _res_block(self, n_channels):
        return nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels),
        )

    def forward(self, X_noisy_log_norm):
        """
        X_noisy_log_norm : (B, F, T)
        Sortie : log(|S_hat|^2)/normalisation (même forme)
        """
        x_in = X_noisy_log_norm    # (B, F, T)
        x = x_in.unsqueeze(1)      # (B,1,F,T)

        x = self.entry(x)          # (B,C,F,T)

        # blocs résiduels
        for block in self.blocks:
            res = x
            out = block(x)
            x = torch.relu(out + res)

        delta = self.out(x).squeeze(1)   # (B,F,T)
        S_hat_log_norm = x_in + delta
        S_hat_log_norm = torch.clamp(S_hat_log_norm, -3.0, 1.0)
        return S_hat_log_norm


# 3. Entraînement du CNN
def trainCNN(
    X_train, X_test, y_train, y_test,
    batch_size=16, epochs=80,
    save_dir="./res/training_logs",
    save_every=1,
    save_best=True
):
    """
    X_train, y_train : (N, F, T)
    Ajouts:
      - courbes train/test sauvegardées (PNG) + CSV
      - best model optionnel
    """
    ensure_dir(save_dir)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = CnnDenoiser().to(device)
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def smooth_loss(Y):
        return torch.mean(torch.abs(Y[:, :, 1:] - Y[:, :, :-1]))

    train_losses = []
    test_losses = []

    best_test = float("inf")
    best_path = os.path.join(save_dir, "best_model.pth")
    last_path = os.path.join(save_dir, "last_model.pth")

    for epoch in range(1, epochs + 1):
        # ----- TRAIN -----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)  # (B, F, T)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(xb)  # (B, F, T)
            loss = criterion(y_pred, yb) + 0.05 * smooth_loss(y_pred)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # ----- TEST -----
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                y_pred = model(xb)
                loss = criterion(y_pred, yb)
                test_loss += loss.item() * xb.size(0)
        test_loss /= len(test_loader.dataset)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"[{epoch:03d}/{epochs}] train={train_loss:.6f}  test={test_loss:.6f}")

        # --- save last model each epoch (pratique pour reprendre) ---
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "train_losses": train_losses,
            "test_losses": test_losses,
        }, last_path)

        # --- save best model ---
        if save_best and test_loss < best_test:
            best_test = test_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "train_losses": train_losses,
                "test_losses": test_losses,
                "best_test": best_test,
            }, best_path)

        # --- save curves/csv every k epochs ---
        if (epoch % save_every) == 0:
            save_losses_csv(save_dir, train_losses, test_losses)
            plot_and_save_losses(save_dir, train_losses, test_losses)

    # Sauvegarde finale (au cas où)
    save_losses_csv(save_dir, train_losses, test_losses)
    plot_and_save_losses(save_dir, train_losses, test_losses)

    print("Saved logs to:", save_dir)
    if save_best:
        print("Best model:", best_path)

    return model


# 4. Estimation sur un signal temporel
def test_estimationCNN(x, model):
    """
    Applique un modèle CNN entraîné pour débruiter un signal audio temporel.

    Cette fonction convertit le signal bruité dans le domaine temps-fréquence,
    applique la même normalisation que lors de l'entraînement, puis utilise le
    réseau pour estimer un spectrogramme débruité. Le signal temporel est ensuite
    reconstruit par ISTFT en réutilisant la phase du signal bruité.

    Étapes principales
    ------------------
    1. Calcul d'une représentation temps-fréquence : log(|X|^2) et extraction de la phase.
    2. Normalisation / clipping de l'entrée (cohérent avec le train).
    3. Inférence du CNN pour estimer le spectrogramme débruité (dans le domaine log-normalisé).
    4. Mélange perceptif entre l'entrée bruitée et la sortie réseau (paramètre beta).
    5. Dénormalisation et conversion en magnitude.
    6. Reconstruction complexe avec la phase bruitée, puis ISTFT pour revenir au domaine temporel.

    Paramètres
    ----------
    x : np.ndarray
        Signal audio bruité (1D) dans le domaine temporel.
    model : torch.nn.Module
        Modèle CNN entraîné (ex. CnnDenoiser) prenant en entrée un tenseur (B,F,T)
        et retournant une estimation de même forme dans le domaine log-normalisé.

    Retour
    ------
    x_pred : np.ndarray
        Signal audio débruité reconstruit dans le domaine temporel.

    Notes
    -----
    - La phase utilisée pour la reconstruction est celle du signal bruité (phase reuse),
      ce qui peut limiter la qualité perceptive finale mais simplifie la reconstruction.
    - Le paramètre beta contrôle le compromis entre conservation du signal d'entrée et
      correction apportée par le réseau.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    fs = 16000
    n_fft = 1024
    hop_length = int(n_fft * 0.2)
    window = 'hann'
    win_length = n_fft

    # 1) log(|X|^2) + phase, même représentation qu'au train
    X_log, phase = STFTabs_phase(x, hop_length, win_length, window, n_fft)  # (F,T)

    # 2) normalisation + clipping cohérent avec le train
    X_in = np.clip(X_log / 90.0, -3.0, 1.0).astype(np.float32)  # (F,T)

    X_in_torch = torch.from_numpy(X_in).unsqueeze(0).to(device)  # (1,F,T)

    # 3) passage dans le CNN
    with torch.no_grad():
        S_hat_log_norm = model(X_in_torch)   # (1,F,T)

    S_hat_log_norm = S_hat_log_norm.cpu().numpy()[0]  # (F,T)

    # 4) mélange perceptif
    beta = 0.3
    S_hat_log_norm = (1 - beta) * X_in + beta * S_hat_log_norm

    # 5) dénormalisation
    S_hat_log = S_hat_log_norm * 90.0
    S_hat_log = np.clip(S_hat_log, -200, 50)

    # 6) magnitude
    S_hat_mag = np.sqrt(np.exp(S_hat_log))  # (F,T)

    # 7) reconstruction complexe avec la phase bruitée
    phase_complex = np.exp(1j * phase)
    S_hat_complex = S_hat_mag * phase_complex

    x_pred = librosa.istft(
        S_hat_complex,
        hop_length=hop_length,
        n_fft=n_fft,
        window=window,
        win_length=win_length,
        length=len(x)
    )

    return x_pred
