import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, roc_auc_score
import os

def build_autoencoder(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    # Encoder
    x = keras.layers.Dense(32, activation="relu")(inputs)
    x = keras.layers.Dense(16, activation="relu")(x)
    encoded = keras.layers.Dense(8, activation="relu")(x)
    # Decoder
    x = keras.layers.Dense(16, activation="relu")(encoded)
    x = keras.layers.Dense(32, activation="relu")(x)
    decoded = keras.layers.Dense(input_dim, activation="linear")(x)

    autoencoder = keras.Model(inputs, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder

def train_autoencoder():
    X_train = np.load("data/processed/X_train.npy")
    X_test  = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test  = np.load("data/processed/y_test.npy")

    # Train ONLY on normal transactions
    X_train_normal = X_train[y_train == 0]

    print("Training Autoencoder on normal transactions only...")
    model = build_autoencoder(X_train.shape[1])
    model.fit(
        X_train_normal, X_train_normal,
        epochs=30, batch_size=256,
        validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )

    # Reconstruction error as anomaly score
    reconstructions = model.predict(X_test)
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

    # Threshold: 95th percentile of normal transaction errors
    train_recon = model.predict(X_train_normal)
    train_mse   = np.mean(np.power(X_train_normal - train_recon, 2), axis=1)
    threshold   = np.percentile(train_mse, 95)

    y_pred = (mse > threshold).astype(int)
    print(f"Threshold: {threshold:.6f}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, mse):.4f}")

    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/autoencoder.keras")
    np.save("saved_models/autoencoder_threshold.npy", np.array([threshold]))
    print("Saved: saved_models/autoencoder.keras")

if __name__ == "__main__":
    train_autoencoder()