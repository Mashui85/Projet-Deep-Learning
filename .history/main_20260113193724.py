from modele import train_test_separation, train, test_estimation
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
do_tt_separation = 1
do_train = 1
do_test_estimation = 1

if __name__ == "__main__":

    if do_tt_separation:
        print("loading files and doing train/test separation")
        X_train, X_test, y_train, y_test, x = train_test_separation()

        # (N,F,T) -> (N,T,F) puis flatten en (N*T, F)
        X_train = X_train.permute(0, 2, 1).reshape(-1, X_train.shape[1])
        X_test  = X_test.permute(0, 2, 1).reshape(-1, X_test.shape[1])
        y_train = y_train.permute(0, 2, 1).reshape(-1, y_train.shape[1])
        y_test  = y_test.permute(0, 2, 1).reshape(-1, y_test.shape[1])

        print("X_train:", X_train.shape, "y_train:", y_train.shape)
        print("X_test :", X_test.shape,  "y_test :", y_test.shape)

    if do_train:
        print("training model")
        model, train_losses, test_losses = train(X_train, X_test, y_train, y_test)

        plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(test_losses, label="test")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    if do_test_estimation:
        print("doing test on model")
        x_pred = test_estimation(x[0], model)

        sf.write("input.wav", x[0], 16000)
        sf.write("output.wav", x_pred, 16000)

        fs = 16000
        t = np.arange(len(x[0])) / fs

        plt.figure()
        plt.plot(t, x[0], label="Input (noisy)")
        plt.plot(t, x_pred, label="Output (denoised)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("MLP denoising – time-domain comparison")
        plt.legend()
        plt.grid(True)
        plt.show()

