import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import librosa

from func import STFTabs, add_noise, data_sized, STFTabs_phase
from load_file import load_file


# ------------------------------------------------------------
# 1) Séparation train / test + construction du MASQUE BINAIRE
# ------------------------------------------------------------
def train_test_separation_masque():
    fs = 16000
    n_fft = 1024
    hop_length = int(n_fft * 0.2)
    window = "hann"
    win_length = n_fft
    test_size = 0.2
    data_size = 5  # durée des segments (s)

    paths, signals, sr_list = load_file()

    # Découpage en segments
    signals_sized = []
    for i in range(len(signals) // 10):
        d = data_sized(signals[i], data_size)
        if isinstance(d, np.ndarray):
            d = [d]
        elif isinstance(d, list) and len(d) > 0 and np.isscalar(d[0]):
            d = [np.array(d)]
        for seg in d:
            signals_sized.append(seg)

    # Bruit babble
    u, _ = librosa.load("babble_16k.wav", sr=fs)

    X_list = []   # features : STFTabs(noisy)/90, clip -> (F,T)
    M_list = []   # targets  : masque binaire -> (F,T)
    x_list = []   # noisy temporel (debug / écoute)

    for seg in signals_sized:
        seg = np.asarray(seg)
        if seg.ndim != 1:
            continue
        if len(seg) < fs:  # < 1 s
            continue
        if np.mean(seg**2) < 1e-6:  # trop silencieux
            continue

        # --- Clean STFT magnitude (même repr. que ton code)
        S = STFTabs(seg, hop_length, win_length, window, n_fft) / 90.0
        S = np.clip(S, -3.0, 1.0)

        # --- Noisy
        snr_db = np.random.uniform(5, 15)
        x_noisy = add_noise(seg, u, snr_db)
        X = STFTabs(x_noisy, hop_length, win_length, window, n_fft) / 90.0
        X = np.clip(X, -3.0, 1.0)

        # --- IBM (masque binaire) : M = 1 si speech >= noise, sinon 0
        # Ici on n'a pas directement |N|. On l'approxime en domaine "log(|.|^2)"
        # de la même façon que tes STFTabs:
        # log(|X|^2)=log(|S+N|^2) et log(|S|^2). Pour avoir une cible cohérente,
        # on reconstruit des magnitudes approx et on déduit |N|^2 ≈ max(|X|^2 - |S|^2, 0).
        #
        # Si STFTabs renvoie déjà log(|.|^2) (c'est cohérent avec ton istft: sqrt(exp(log))),
        # alors exp(...) donne |.|^2.
        S_pow = np.exp(S * 90.0)       # ≈ |S|^2
        X_pow = np.exp(X * 90.0)       # ≈ |X|^2
        N_pow = np.maximum(X_pow - S_pow, 0.0)  # ≈ |N|^2 (approx)
        M = (S_pow >= N_pow).astype(np.float32)  # (F,T) binaire

        X_list.append(X.astype(np.float32))
        M_list.append(M)
        x_list.append(x_noisy)

    X_array = np.array(X_list, dtype=np.float32)  # (N,F,T)
    M_array = np.array(M_list, dtype=np.float32)  # (N,F,T)

    X_train, X_test, y_train, y_test = train_test_split(
        X_array, M_array, test_size=test_size, random_state=42, shuffle=True
    )

    X_train = torch.from_numpy(X_train)  # (N,F,T)
    X_test  = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)
    y_test  = torch.from_numpy(y_test)

    return X_train, X_test, y_train, y_test, x_list


# ------------------------------------------------------------
# 2) CNN 2D pour prédire le masque (logits) : même squelette
# ------------------------------------------------------------
class CnnMasker(nn.Module):
    def __init__(self, n_channels=32, n_blocks=4):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
        )

        blocks = []
        for _ in range(n_blocks):
            blocks.append(self._res_block(n_channels))
        self.blocks = nn.Sequential(*blocks)

        # sortie : logits du masque (pas de sigmoid ici)
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
        X_noisy_log_norm : (B, F, T)  dans [-3, 1]
        Sortie : logits du masque (B, F, T) (réel)
        """
        x = X_noisy_log_norm.unsqueeze(1)  # (B,1,F,T)
        x = self.entry(x)                  # (B,C,F,T)

        for block in self.blocks:
            res = x
            out = block(x)
            x = torch.relu(out + res)

        logits = self.out(x).squeeze(1)    # (B,F,T)
        return logits


# ------------------------------------------------------------
# 3) Entraînement (BCEWithLogits) + régularisation temporelle
# ------------------------------------------------------------
def trainMasqueCNN(X_train, X_test, y_train, y_test, batch_size=16, epochs=150):
    """
    X_train : (N,F,T)  noisy log_norm
    y_train : (N,F,T)  masque binaire {0,1}
    """
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test, y_test),   batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = CnnMasker().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def smooth_loss_logits(L):
        return torch.mean(torch.abs(L[:, :, 1:] - L[:, :, :-1]))

    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)

            loss = criterion(logits, yb) + 0.05 * smooth_loss_logits(logits)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)  # note: sans le smooth term au test (comme ton code)
                test_loss += loss.item() * xb.size(0)
        test_loss /= len(test_loader.dataset)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"[{epoch:03d}/{epochs}] train={train_loss:.6f}  test={test_loss:.6f}")

    return model, train_losses, test_losses



# ------------------------------------------------------------
# 4) Estimation sur un signal temporel (apply mask + iSTFT)
# ------------------------------------------------------------
def test_estimationMasqueCNN(x, model, threshold=0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    fs = 16000
    n_fft = 1024
    hop_length = int(n_fft * 0.2)
    window = "hann"
    win_length = n_fft

    # 1) représentation cohérente : log(|X|^2) + phase
    X_log, phase = STFTabs_phase(x, hop_length, win_length, window, n_fft)  # (F,T)

    # 2) normalisation + clipping comme au train
    X_in = np.clip(X_log / 90.0, -3.0, 1.0).astype(np.float32)  # (F,T)
    X_t = torch.from_numpy(X_in).unsqueeze(0).to(device)         # (1,F,T)

    # 3) prédiction du masque
    with torch.no_grad():
        logits = model(X_t)                     # (1,F,T)
        prob = torch.sigmoid(logits)[0]         # (F,T) in [0,1]
        M = (prob > threshold).float().cpu().numpy()  # (F,T) binaire

    # 4) reconstruire magnitude estimée en appliquant le masque sur la magnitude du noisy
    # X_log = log(|X|^2), donc |X| = sqrt(exp(X_log))
    X_pow = np.exp(np.clip(X_log, -200, 50))    # |X|^2
    X_mag = np.sqrt(X_pow)                      # |X|
    S_hat_mag = M * X_mag                       # |S_hat|

    # 5) reconstruction complexe avec la phase bruitée
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
