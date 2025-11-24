import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import librosa

from func import STFTabs, add_noise, data_sized
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

    paths, signals, sr_list = load_file()

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
            STFTabs(signals_sized[i], hop_length, win_length, window, n_fft)
        )

    # Bruit babble
    u, fs = librosa.load('babble_16k.wav', sr=fs)
    U = STFTabs(u, hop_length, win_length, window, n_fft) / 90.0  # inutilisé mais ok

    x_list = []
    X_list = []

    for i in range(len(signals_sized)):
        # On impose un SNR de -2 dB
        snr_db = np.random.uniform(-5, 10)  # par ex. entre -5 et 10 dB
        x_noisy = add_noise(signals_sized[i], u, snr_db)
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

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        # Sortie : magnitude propre estimée
        self.conv_out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, X_noisy_mag_norm):
        """
        X_noisy_mag_norm : (B, F, T)  -> magnitude bruitée /90
        Retour : magnitude propre estimée (B, F, T)
        """
        x = X_noisy_mag_norm.unsqueeze(1)  # (B, 1, F, T)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        S_hat_mag = self.conv_out(x)       # (B, 1, F, T)
        S_hat_mag = torch.relu(S_hat_mag)  # >= 0
        S_hat_mag = S_hat_mag.squeeze(1)   # (B, F, T)

        return S_hat_mag


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
    criterion = nn.MSELoss()
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
    """
    x : signal audio 1D (noisy)
    model : CnnDenoiser entraîné
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    fs = 16000
    n_fft = 1024
    hop_length = int(n_fft * 0.2)
    window = 'hann'
    win_length = n_fft

    # STFT complexe pour récupérer la phase
    X_complex = librosa.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )

    mag_noisy = np.abs(X_complex)      # (F, T)
    phase = np.angle(X_complex)       # (F, T)

    # Normalisation cohérente avec le train : /90
    X_in = (mag_noisy / 90.0).astype(np.float32)  # (F, T)

    X_in_torch = torch.from_numpy(X_in).unsqueeze(0).to(device)  # (1, F, T)

    with torch.no_grad():
        S_hat_mag = model(X_in_torch)   # (1, F, T)

    S_hat_mag = S_hat_mag.cpu().numpy()[0]  # (F, T)

    # Reconstruction complexe avec phase du signal bruité
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
