from modeleCNN import train_test_separation, trainCNN, test_estimationCNN
import soundfile as sf

do_tt_separation     = 1
do_train             = 1
do_test_estimation   = 1

if __name__ == "__main__":

    if do_tt_separation:
        print("Loading data and train/test split...")
        X_train, X_test, y_train, y_test, x_list = train_test_separation()

        # IMPORTANT : NE RIEN PERMUTER, NE RIEN RESHAPER
        # Shapes attendues :
        # X_train : (N, F, T)
        # y_train : (N, F, T)

        print("Shapes :")
        print("X_train :", X_train.shape)
        print("y_train :", y_train.shape)


    if do_train:
        print("Training CNN model...")
        model = trainCNN(X_train, X_test, y_train, y_test)


    if do_test_estimation:
        print("Running test estimation...")
        noisy_example = x_list[0]                 # waveform bruitée
        x_pred = test_estimationCNN(noisy_example, model)

        sf.write("input.wav", noisy_example, 16000)
        sf.write("output.wav", x_pred, 16000)

        print("Saved input.wav and output.wav")
