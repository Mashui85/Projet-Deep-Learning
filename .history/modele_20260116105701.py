import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from func import STFTabs, add_noise, data_sized, STFTabs_phase
from load_file import load_file
import librosa
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import soundfile as sf


def train_test_separation():
    # Cette fonction charge es signaux de 
    fs = 16000
    n_fft = 1024
    hop_length = int(n_fft * 0.2)
    window = "hann"
    win_length = n_fft
    test_size = 0.2
    data_size = 5
    N = 270

    paths, signals, sr_list = load_file()
    paths, signals, sr_list = paths[:N], signals[:N], sr_list[:N]

    signals_sized = []
    for i in range(len(signals)):
        d = data_sized(signals[i], data_size)
        if len(d) > 100:
            signals_sized.append(d)
        else:
            for j in d:
                signals_sized.append(j)

    S_list = []
    for i in range(len(signals_sized)):
        D = STFTabs(signals_sized[i], hop_length, win_length, window, n_fft)
        S_list.append(D / 90.0)

    u, _ = librosa.load("babble_16k.wav", sr=fs)

    x_list = []
    X_list = []
    for i in range(len(signals_sized)):
        x_list.append(add_noise(signals_sized[i], u, 5))
        D = STFTabs(x_list[i], hop_length, win_length, window, n_fft)
        X_list.append(D / 90.0)

    S_array = np.array(S_list)
    X_array = np.array(X_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X_array, S_array, test_size=test_size, random_state=42, shuffle=True
    )

    # petit sanity check audio (optionnel mais tu l'avais)
    pipi = librosa.istft(
        np.sqrt(np.exp(X_list[0] * 90.0)),
        hop_length=hop_length,
        n_fft=n_fft,
        window=window,
        win_length=win_length,
        length=len(x_list[0]),
    )
    sf.write("test.wav", pipi, fs)

    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    return X_train, X_test, y_train, y_test, x_list


def train(X_train, X_test, y_train, y_test, epochs=150, batch_size=256, lr=1e-3):
    # Ici on suppose que X_train est déjà en (N_trames, 513) via le main
    input_dim = X_train.shape[1]

    model = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, input_dim),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_pred = model(xb)
                loss = criterion(y_pred, yb)
                test_loss += loss.item() * xb.size(0)
        test_loss /= len(test_loader.dataset)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"[{epoch:03d}/{epochs}] train={train_loss:.6f}  test={test_loss:.6f}")

    return model, train_losses, test_losses


def test_estimation(x, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    n_fft = 1024
    hop_length = int(n_fft * 0.2)
    window = "hann"
    win_length = n_fft

    X_log, phase = STFTabs_phase(x, hop_length, win_length, window, n_fft)  # (F,T)
    X_norm = (X_log / 90.0).astype(np.float32)  # (F,T)

    X_in = torch.from_numpy(X_norm.T).to(device)  # (T,F=513)

    with torch.no_grad():
        X_pred_norm = model(X_in)  # (T,513)

    X_pred_norm = X_pred_norm.cpu().numpy().T  # (F,T)
    X_pred_log = X_pred_norm * 90.0

    X_pred_mag = np.sqrt(np.exp(X_pred_log))
    phase_complex = np.exp(1j * phase)
    X_pred = X_pred_mag * phase_complex

    x_pred = librosa.istft(
        X_pred,
        hop_length=hop_length,
        n_fft=n_fft,
        window=window,
        win_length=win_length,
        length=len(x),
    )
    return x_pred
