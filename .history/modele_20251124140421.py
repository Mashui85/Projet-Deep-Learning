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
    fs=16000
    n_fft = 1024
    hop_length = int(n_fft * 0.2)
    window = 'hann'
    win_length = n_fft
    test_size = 0.2
    data_size = 5 # temps des signaux évalué
    paths, signals, sr_list = load_file()
    S_list = []
    signals_sized = []
    for i in range(len(signals)):
        d = data_sized(signals[i],data_size)
        if len(d)>100:
            signals_sized.append(d)
        else:    
            for j in d:
                signals_sized.append(j)
    # print(len(signals_sized))

    signals_sized_arr = np.array(signals_sized)
    
    for i in range(len(signals_sized)):
        D = STFTabs(signals_sized[i], hop_length, win_length, window, n_fft)
        S_list.append(D/90)



    u,fs = librosa.load('babble_16k.wav',sr=fs)
    U = STFTabs(u,hop_length,win_length,window,n_fft)
    U = U/90
    x_list = []
    X_list = []
    for i in range(len(signals_sized)):
        
        x_list.append(add_noise(signals_sized[i],u,-2)) # On impose une valeur de SNR
        D = STFTabs(x_list[i],hop_length,win_length,window,n_fft)
        X_list.append(D/90)

    # X : données d'entrée, y : labels (ou valeurs cibles)
    S_array = np.array(S_list)
    X_array = np.array(X_list)

    X_train, X_test, y_train, y_test = train_test_split(
    X_array, S_array, test_size=test_size, random_state=42, shuffle=True
    )

    print(X_train.shape)
    pipi = librosa.istft(np.sqrt(np.exp(X_list[0]*90)),hop_length=hop_length,n_fft=n_fft,window=window,win_length=win_length, length=len(x_list[0]))
    sf.write('test.wav',pipi,fs)

    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    X_test  = torch.from_numpy(X_test.astype(np.float32))
    y_test  = torch.from_numpy(y_test.astype(np.float32))
    return X_train, X_test, y_train, y_test, x_list

def train(X_train, X_test, y_train, y_test):
    batch_size = 256

    X_train = X_train.view(X_train.size(0), -1)
    X_test  = X_test.view(X_test.size(0), -1)
    y_train = y_train.view(y_train.size(0), -1)
    y_test  = y_test.view(y_test.size(0), -1)

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
    print("Using device:", device)
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

    

def test_estimation(x, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    n_fft = 1024
    hop_length = int(n_fft * 0.2)
    window = 'hann'
    win_length = n_fft

    # 1) STFT log-power + phase
    X_log, phase = STFTabs_phase(x, hop_length, win_length, window, n_fft)  # X_log shape: (F, T)
    F, T = X_log.shape

    # 2) Normalisation identique à train_test_separation : /90
    X_norm = (X_log / 90.0).astype(np.float32)  # (F, T)

    # 3) >>> correction dimension : aplatir en (1, F*T), comme dans train() <<<
    X_in = X_norm.reshape(1, -1)               # (1, F*T)
    X_in = torch.from_numpy(X_in).to(device)

    # 4) Passage dans le modèle
    with torch.no_grad():
        X_pred_norm = model(X_in)              # (1, F*T)

    # 5) On remet en (F, T)
    X_pred_norm = X_pred_norm.cpu().numpy().reshape(F, T)

    # 6) Revenir à la vraie échelle log : *90
    X_pred_log = X_pred_norm * 90.0           # log(|S_clean|^2)

    # 7) Magnitude + phase
    X_pred_mag = np.sqrt(np.exp(X_pred_log))  # |S_clean|
    phase_complex = np.exp(1j * phase)
    X_pred = X_pred_mag * phase_complex

    # 8) iSTFT
    x_pred = librosa.istft(
        X_pred,
        hop_length=hop_length,
        n_fft=n_fft,
        window=window,
        win_length=win_length,
        length=len(x)
    )
    print('caca', len(x_pred), len(x))
    return x_pred


    