from load_file import load_file
from modele import train_test_separation, train, test_estimation
from load_file import load_file
from func import STFT_display

# Valeurs pour verifier chaque blocs
do_load = 0
do_tt_separation = 1
do_train = 0
do_test_estimation = 0

if __name__ == '__main__':
    
    if do_tt_separation: 
        print('loading files and doing train test separation')
        X_train, X_test, y_train, y_test, x = train_test_separation()
        
    if do_train:
        print('training model')
        model = train(X_train, X_test, y_train, y_test)
        

    if do_test_estimation:
        print('doint test on model')
        test_estimation(x, model)