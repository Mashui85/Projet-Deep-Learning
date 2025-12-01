import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import librosa

from func import STFTabs, add_noise, data_sized, STFTabs_phase
from load_file import load_file


# ---------------------------
# 1. Séparation train / test
# ---------------------------
def train_test_separation():
    fs = 16000
    n_fft = 1024
    hop_length = int(n_fft * 0.2)
    window = 'hann'
    win_length = n_fft
    test_size = 0.2
    data_size = 5  # durée des segments (en secondes, j'imagine)
    N = 270
    paths, signals, sr_list = load_file()
    paths, signals, sr_list = paths[:N], signals[:N], sr_list[:N]
    S_list = []
    signals_sized = []

    # On découpe les signaux
    for i in range(len(signals) // 10):
        d = data_sized(signals[i], data_size)
        if len(d) > 100:
            signals_sized.append(d)
        else:
            for j in d:
                signals_sized.append(j)

    # Calcul des STFT propres
    for i in range(len(signals_sized)):
        S_list.append(
            STFTabs(signals_sized[i], hop_length, win_length, window, n_fft)/90
        )

    # Bruit babble
    u, fs = librosa.load('babble_16k.wav', sr=fs)
    U = STFTabs(u, hop_length, win_length, window, n_fft) / 90.0  # inutilisé mais ok

    x_list = []
    X_list = []

    for i in range(len(signals_sized)):
        # On impose un SNR de -2 dB
        snr_db = np.random.uniform(-5, 10)  # par ex. entre -5 et 10 dB
        x_noisy = add_noise(signals_sized[i], u, -1)
        x_list.append(x_noisy)

        X_list.append(
            STFTabs(x_noisy, hop_length, win_length, window, n_fft) / 90.0
        )

    # X : données d'entrée (noisy/90), y : labels (clean)
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


# ---------------------------
# 2. CNN 2D pour débruitage
# ---------------------------
class CnnDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        self.conv_out = nn.Conv2d(64, 1, 1)

    def forward(self, X_noisy_log_norm):
        # X_noisy_log_norm : (B, F, T) = log(|X|^2)/90
        x = X_noisy_log_norm.unsqueeze(1)  # (B,1,F,T)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        res = x
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x + res                           # skip

        # Sortie = log(|S_clean|^2)/90, peut être négative
        S_hat_log_norm = self.conv_out(x).squeeze(1)  # (B,F,T)
        return S_hat_log_norm



# ---------------------------
# 3. Entraînement du CNN
# ---------------------------
def trainCNN(X_train, X_test, y_train, y_test, batch_size=16, epochs=30):
    """
    X_train, y_train : (N, F, T)
    """
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
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        # ----- TRAIN -----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)  # (B, F, T)
            yb = yb.to(device)

            optimizer.zero_grad()
            y_pred = model(xb)  # (B, F, T)
            loss = criterion(y_pred, yb)
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

        print(f"[{epoch:03d}/{epochs}] train={train_loss:.6f}  test={test_loss:.6f}")

    return model


# ---------------------------
# 4. Estimation sur un signal temporel
# ---------------------------
def test_estimationCNN(x, model):
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

    # 2) normalisation /90
    X_in = (X_log / 90.0).astype(np.float32)  # (F,T)

    X_in_torch = torch.from_numpy(X_in).unsqueeze(0).to(device)  # (1,F,T)

    # 3) passage dans le CNN : sortie = log(|S_hat|^2)/90
    with torch.no_grad():
        S_hat_log_norm = model(X_in_torch)   # (1,F,T)

    S_hat_log_norm = S_hat_log_norm.cpu().numpy()[0]  # (F,T)

    # 4) dénormalisation : on revient à log(|S_hat|^2)
    S_hat_log = S_hat_log_norm * 90.0

    # 5) clamp pour éviter les débordements numériques
    S_hat_log = np.clip(S_hat_log, -200, 50)

    # 6) magnitude = sqrt(exp(log(|S_hat|^2)))
    S_hat_mag = np.sqrt(np.exp(S_hat_log))  # (F,T)

    # 7) reconstruction complexe avec la phase du signal bruité
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
