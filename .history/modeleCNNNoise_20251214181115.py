import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import librosa

from func import add_noise, data_sized, STFTabs_phase
from load_file import load_file


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x
    # si (C, N) ou (N, C) -> moyenne des canaux
    if x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
        return x.mean(axis=0)
    return x.mean(axis=1)

def match_length(noise: np.ndarray, target_len: int) -> np.ndarray:
    noise = np.asarray(noise).reshape(-1)
    if len(noise) == target_len:
        return noise
    if len(noise) < target_len:
        reps = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, reps)
    return noise[:target_len]

# ---------------------------
# 0) Utils: loss de lissage
# ---------------------------
def smoothness_loss(Y: torch.Tensor) -> torch.Tensor:
    """
    Y: (B,F,T)
    Pénalise les variations brusques en fréquence et en temps.
    """
    df = torch.mean(torch.abs(Y[:, 1:, :] - Y[:, :-1, :]))
    dt = torch.mean(torch.abs(Y[:, :, 1:] - Y[:, :, :-1]))
    return df + dt


# ---------------------------
# 1) Dataset: prédire le bruit en log
#   X_log_norm = log(|X|^2)/90
#   N_log_norm = (X_log_norm - S_log_norm) = log(|X|^2)/90 - log(|S|^2)/90
# ---------------------------
def train_test_separation(
    test_size=0.2,
    data_size=5,
    fs=16000,
    n_fft=1024,
    snr_min=5.0,
    snr_max=15.0,
    max_signals=None,
):
    hop_length = int(n_fft * 0.2)
    window = "hann"
    win_length = n_fft

    paths, signals, sr_list = load_file()
    if max_signals is not None:
        signals = signals[:max_signals]

    # 1) segments propres
    segments = []
    for i in range(len(signals) // 10):
        d = data_sized(signals[i], data_size)
        # IMPORTANT: on force d à être itérable de segments
        # si data_sized renvoie un segment unique, enveloppe-le
        if isinstance(d, (np.ndarray, list)) and (not isinstance(d, list) or (len(d) > 0 and isinstance(d[0], (float, np.floating, int, np.integer)))):
            # d ressemble à un vecteur 1D -> un seul segment
            segments.append(np.asarray(d))
        else:
            # d est une liste de segments
            for seg in d:
                segments.append(np.asarray(seg))

    # 2) bruit
    u, _ = librosa.load("babble_16k.wav", sr=fs)

    X_list, N_list = [], []
    for seg in segments:
        # clean log-power
        S_log, _ = STFTabs_phase(seg, hop_length, win_length, window, n_fft)  # (F,T)

        # noisy
        snr_db = np.random.uniform(snr_min, snr_max)
        for seg in segments:
            seg = to_mono(seg)                     # <--- IMPORTANT
            seg = np.asarray(seg).reshape(-1)

            snr_db = np.random.uniform(5, 15)


            # aligne la longueur du bruit sur le segment
            noise_seg = match_length(u, len(seg))  # <--- IMPORTANT

            x_noisy = add_noise(seg, noise_seg, snr_db)

        # noisy log-power
        X_log, _ = STFTabs_phase(x_noisy, hop_length, win_length, window, n_fft)

        # normalisation /90
        X_log_norm = (X_log / 90.0).astype(np.float32)
        S_log_norm = (S_log / 90.0).astype(np.float32)

        # cible = bruit en log (résiduel)
        N_log_norm = (X_log_norm - S_log_norm).astype(np.float32)

        X_list.append(X_log_norm)
        N_list.append(N_log_norm)

    X_array = np.array(X_list, dtype=np.float32)  # (N,F,T)
    N_array = np.array(N_list, dtype=np.float32)  # (N,F,T)

    X_train, X_test, y_train, y_test = train_test_split(
        X_array, N_array, test_size=test_size, random_state=42, shuffle=True
    )

    X_train = torch.from_numpy(X_train)
    X_test  = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)  # N_log_norm
    y_test  = torch.from_numpy(y_test)

    return X_train, X_test, y_train, y_test


# ---------------------------
# 2) Modèle: CNN résiduel qui prédit N_log_norm
# ---------------------------
class NoisePredictorCNN(nn.Module):
    def __init__(self, n_channels=32, n_blocks=4):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(1, n_channels, 3, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList([self._res_block(n_channels) for _ in range(n_blocks)])
        self.out = nn.Conv2d(n_channels, 1, 1)

    def _res_block(self, c):
        return nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
        )

    def forward(self, X_log_norm):
        """
        X_log_norm: (B,F,T) = log(|X|^2)/90
        Retour: N_hat_log_norm (B,F,T)
        """
        x = X_log_norm.unsqueeze(1)  # (B,1,F,T)
        x = self.entry(x)
        for block in self.blocks:
            x = torch.relu(block(x) + x)
        n_hat = self.out(x).squeeze(1)  # (B,F,T)

        # clamp doux pour éviter des délires (à ajuster si besoin)
        n_hat = torch.clamp(n_hat, -3.0, 3.0)
        return n_hat


# ---------------------------
# 3) Entraînement
# ---------------------------
def trainCNN_noise(
    X_train, X_test, y_train, y_test,
    batch_size=16,
    epochs=60,
    lr=1e-3,
    lambda_smooth=0.05
):
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = NoisePredictorCNN(n_channels=32, n_blocks=4).to(device)
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        tr = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            y_pred = model(xb)  # N_hat_log_norm
            loss = mse(y_pred, yb) + lambda_smooth * smoothness_loss(y_pred)
            loss.backward()
            opt.step()
            tr += loss.item() * xb.size(0)
        tr /= len(train_loader.dataset)

        model.eval()
        te = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_pred = model(xb)
                loss = mse(y_pred, yb) + lambda_smooth * smoothness_loss(y_pred)
                te += loss.item() * xb.size(0)
        te /= len(test_loader.dataset)

        print(f"[{epoch:03d}/{epochs}] train={tr:.6f}  test={te:.6f}")

    return model


# ---------------------------
# 4) Test: reconstruire en soustrayant le bruit prédit
# ---------------------------
def test_estimationCNN_noise(x, model, fs=16000, n_fft=1024):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    hop_length = int(n_fft * 0.2)
    window = "hann"
    win_length = n_fft

    # X_log + phase
    X_log, phase = STFTabs_phase(x, hop_length, win_length, window, n_fft)  # (F,T)
    X_log_norm = (X_log / 90.0).astype(np.float32)

    X_in = torch.from_numpy(X_log_norm).unsqueeze(0).to(device)  # (1,F,T)

    with torch.no_grad():
        N_hat_log_norm = model(X_in).cpu().numpy()[0]  # (F,T)

    # Soustraction du bruit en log (norm)
    S_hat_log_norm = X_log_norm - N_hat_log_norm

    # Retour échelle log
    S_hat_log = S_hat_log_norm * 90.0
    S_hat_log = np.clip(S_hat_log, -200, 50)

    # magnitude + phase
    S_hat_mag = np.sqrt(np.exp(S_hat_log))
    S_hat_complex = S_hat_mag * np.exp(1j * phase)

    x_pred = librosa.istft(
        S_hat_complex,
        hop_length=hop_length,
        n_fft=n_fft,
        window=window,
        win_length=win_length,
        length=len(x),
    )
    return x_pred
