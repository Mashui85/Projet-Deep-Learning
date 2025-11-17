from load_file import load_file
from modele import train_test_separation, train, test_estimation
from load_file import load_file
from func import STFT_display
import soundfile as sf

# Valeurs pour verifier chaque blocs
do_load = 0
do_tt_separation = 1
do_train = 1
do_test_estimation = 1

if __name__ == '__main__':
    
    if do_tt_separation: 
        print('loading files and git doing train test separation')
        X_train, X_test, y_train, y_test, x = train_test_separation()
        X_train = X_train.permute(0, 2, 1)
        X_test  = X_test.permute(0, 2, 1)
        y_train = y_train.permute(0, 2, 1)
        y_test  = y_test.permute(0, 2, 1)

        X_train = X_train.reshape(-1, X_train.shape[-1])
        X_test  = X_test.reshape(-1, X_test.shape[-1])
        y_train = y_train.reshape(-1, y_train.shape[-1])
        y_test  = y_test.reshape(-1, y_test.shape[-1])

    if do_train:
        print('training model')
        model = train(X_train, X_test, y_train, y_test)
        

    if do_test_estimation:
        print('doint test on model')
        x_pred = test_estimation(x[0], model)
        print(x_pred.shape)
        sf.write("input.wav",x[0], 16000)
        sf.write("output.wav", x_pred, 16000)