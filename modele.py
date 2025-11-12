import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from func import STFTabs, add_noise
from load_file import load_file
import librosa
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def train_test_separation(): 
    fs=16000
    n_fft = 1024
    hop_length = int(n_fft * 0.2)
    window = 'hann'
    win_length = n_fft
    test_size = 0.2

    paths, signals, sr_list = load_file()
    S = STFTabs(signals,hop_length,win_length,window,n_fft)

    u,fs = librosa.load('babble_16k.wav',sr=fs)
    U = STFTabs(u,hop_length,win_length,window,n_fft)

    x = add_noise(signals,u,-2)
    X = STFTabs(x,hop_length,win_length,window,n_fft)

    # X : données d'entrée, y : labels (ou valeurs cibles)

    X_train, X_test, y_train, y_test = train_test_split(
    X, S, test_size=test_size, random_state=42, shuffle=True
    )
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test  = torch.from_numpy(X_test).float()
    y_test  = torch.from_numpy(y_test).float()
    return X_train, X_test, y_train, y_test, x

def train(X_train, X_test, y_train, y_test):
    batch_size = 256
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)
    input_dim = X_train.shape[1]  # nombre de fréquences (colonnes)
    
    #On definit le modele
    
    model = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, input_dim),
        nn.ReLU(),  # pour rester positif (magnitudes)
    )
    #Initialisation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 30
    
    #Boucle d'entrainement 
    
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
        
        #Evaluation
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                y_pred = model(xb)
                loss = criterion(y_pred, yb)
                test_loss += loss.item() * xb.size(0)
        test_loss /= len(test_loader.dataset)

        print(f"[{epoch:03d}/{epochs}] train={train_loss:.6f}  test={test_loss:.6f}")
    return model

    

def test_estimation(x,model):
    X = STFTabs(x,hop_length,win_length,window,n_fft)
    X_pred_mag = model.predict(X)
    X_pred_mag = np.sqrt(np.exp(X_pred_mag))
    phase = np.angle(X)
    phase_complex = np.exp(1j * phase_n)
    X_pred = X_pred_mag * phase_complex

    x_pred = librosa.istft(X_pred,hop_length=hop_length,n_fft=n_fft,window=window,win_length=win_length, length=len(x))
    
    return x_pred

    