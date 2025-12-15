from modeleCNN import (
    train_test_separation,
    trainCNN_noise,
    test_estimationCNN_noise
)
import soundfile as sf

do_tt_separation   = 1
do_train           = 1
do_test_estimation = 1

if __name__ == "__main__":

    if do_tt_separation:
        print("Loading data and train/test split (noise prediction)...")
        X_train, X_test, y_train, y_test = train_test_separation(
            snr_min=5,
            snr_max=15,
            max_signals=300
        )
        print("Shapes:")
        print("X_train :", X_train.shape)  # (N,F,T)
        print("y_train :", y_train.shape)  # (N,F,T) = bruit log/90

    if do_train:
        print("Training CNN (noise predictor)...")
        model = trainCNN_noise(
            X_train, X_test, y_train, y_test,
            batch_size=16,
            epochs=60,
            lr=1e-3
        )

    if do_test_estimation:
        print("Running test estimation...")
        # recrée un bruité pour test
        import librosa
        from func import add_noise

        clean, fs = librosa.load("test_clean.wav", sr=16000)
        noise, _ = librosa.load("babble_16k.wav", sr=16000)
        noisy = add_noise(clean, noise, snr_db=5)

        x_pred = test_estimationCNN_noise(noisy, model)

        sf.write("input_noisy.wav", noisy, 16000)
        sf.write("output_denoised.wav", x_pred, 16000)
        print("Saved input_noisy.wav and output_denoised.wav")
