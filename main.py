from load_file import load_file
from modele import train_test_separation, train, test_estimation
from load_file import load_file
from func import STFT_display

# Valeurs pour verifier chaque blocs
do_load = True
do_tt_separation = True
do_train = True
do_test_estimation = True

if __name__ == '__main__':
    
    if do_load: 
        print('Loading audio files')
        load_file()

    if do_tt_separation(): 
        print('doing train test separation')
    
    if do_train:
        train_ind, test_ind = train_test_split(range(nb_sentences),test_size=0.10)
        np.save(output_dir + '/train_ind.npy',train_ind)
        np.save(output_dir + '/test_ind.npy',test_ind)
        print('Train MLP regressor')
        do_train(train_ind)

    if step_test:
        print('MLP-based regression and audio synthesis')
        test_ind = np.load(output_dir + '/test_ind.npy')
        do_test(test_ind)
