from modeleMasqueCNN import (
    train_test_separation_masque,
    trainMasqueCNN,
    test_estimationMasqueCNN
)
import soundfile as sf
import torch
import numpy as np
import matplotlib.pyplot as plt


# Flags d'exécution
do_tt_separation   = 1
do_train           = 1
do_test_estimation = 1


if __name__ == "__main__":

    # 1) Séparation train / test + préparation données
    if do_tt_separation:
        print("=" * 60)
        print("[STEP 1] Loading data and building binary masks...")
        X_train, X_test, y_train, y_test, x_list = train_test_separation_masque()

        print("[INFO] Dataset ready")
        print("  X_train :", X_train.shape, " (noisy log-norm)")
        print("  y_train :", y_train.shape, " (binary mask)")
        print("  X_test  :", X_test.shape)
        print("  y_test  :", y_test.shape)

        # sanity check
        print("[CHECK] Mask statistics (train):")
        print("  mean(mask) =", y_train.mean().item())
        print("  min / max  =", y_train.min().item(), "/", y_train.max().item())

        print("=" * 60)

    # 2) Entraînement du CNN masque
    if do_train:
        print("=" * 60)
        print("[STEP 2] Training CNN mask estimator...")
        model, train_losses, test_losses = trainMasqueCNN(
            X_train, X_test,
            y_train, y_test,
            batch_size=16,
            epochs=80
        )


        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Model trained — trainable parameters: {n_params:,}")

        print("=" * 60)
        plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("Mask CNN — train/test loss")
    plt.legend()
    plt.grid(True)
    plt.show()



    # 3) Test sur un exemple temporel
    if do_test_estimation:
        print("=" * 60)
        print("[STEP 3] Running test estimation with binary mask...")

        noisy_example = x_list[0]
        print("[INFO] Noisy example length:", len(noisy_example), "samples")

        x_pred = test_estimationMasqueCNN(
            noisy_example,
            model,
            threshold=0.5
        )

        # noms explicites (important pour éviter confusion dans le rapport)
        input_wav  = "masque_input_noisy.wav"
        output_wav = "masque_output_denoised.wav"

        sf.write(input_wav, noisy_example, 16000)
        sf.write(output_wav, x_pred, 16000)

        print("[DONE] Audio files written:")
        print("  ->", input_wav)
        print("  ->", output_wav)
        print("=" * 60)
