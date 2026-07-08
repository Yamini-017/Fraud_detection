import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, roc_auc_score
import os

def train_lstm():
    X_train = np.load("data/processed/X_train.npy").astype(np.float32)
    X_test  = np.load("data/processed/X_test.npy").astype(np.float32)
    y_train = np.load("data/processed/y_train.npy")
    y_test  = np.load("data/processed/y_test.npy")

    # Reshape for LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm  = X_test.reshape((X_test.shape[0],  1, X_test.shape[1]))

    print(f"Training LSTM... X shape: {X_train_lstm.shape}")

    model = keras.Sequential([
        keras.layers.Input(shape=(1, X_train.shape[1])),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1,  activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    model.fit(
        X_train_lstm, y_train,
        epochs=20, batch_size=512,
        validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, monitor="val_loss")],
        verbose=1
    )

    y_prob = model.predict(X_test_lstm, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    print(f"\nSample predictions: {y_prob[:10]}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/lstm.keras")
    print("Saved: saved_models/lstm.keras")

if __name__ == "__main__":
    train_lstm()