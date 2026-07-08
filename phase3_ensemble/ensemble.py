import numpy as np
import joblib
from tensorflow import keras

MODEL_WEIGHTS = {
    "xgboost":          0.35,
    "isolation_forest": 0.20,
    "lstm":             0.25,
    "autoencoder":      0.20,
}

class FraudEnsemble:
    def __init__(self):
        print("[Ensemble] Loading models...")
        self.xgb          = joblib.load("saved_models/xgboost.pkl")
        self.iso          = joblib.load("saved_models/isolation_forest.pkl")
        self.lstm         = keras.models.load_model("saved_models/lstm.keras")
        self.ae           = keras.models.load_model("saved_models/autoencoder.keras")
        self.ae_threshold = float(np.load("saved_models/autoencoder_threshold.npy")[0])
        print(f"[Ensemble] All 4 models loaded. AE threshold: {self.ae_threshold:.6f}")

    def predict(self, X: np.ndarray) -> dict:
        # Ensure float32 for keras compatibility
        X = X.astype(np.float32)

        # 1. XGBoost
        try:
            xgb_prob = float(self.xgb.predict_proba(X)[0][1])
        except Exception as e:
            print(f"[XGBoost] Error: {e}")
            xgb_prob = 0.0

        # 2. Isolation Forest
        try:
            iso_raw  = self.iso.predict(X)[0]
            iso_prob = 1.0 if iso_raw == -1 else 0.0
        except Exception as e:
            print(f"[IsoForest] Error: {e}")
            iso_prob = 0.0

        # 3. LSTM — reshape to (1, timesteps=1, features)
        try:
            X_lstm    = X.reshape(1, 1, X.shape[1])
            lstm_prob = float(self.lstm.predict(X_lstm, verbose=0)[0][0])
        except Exception as e:
            print(f"[LSTM] Error: {e}")
            lstm_prob = 0.0

        # 4. Autoencoder — reconstruction error
        try:
            recon    = self.ae.predict(X, verbose=0)
            ae_mse   = float(np.mean(np.power(X - recon, 2)))
            ae_prob  = 1.0 if ae_mse > self.ae_threshold else 0.0
            print(f"[AE] MSE={ae_mse:.6f} threshold={self.ae_threshold:.6f} flag={ae_prob}")
        except Exception as e:
            print(f"[Autoencoder] Error: {e}")
            ae_mse  = 0.0
            ae_prob = 0.0

        # Weighted ensemble
        score = (
            MODEL_WEIGHTS["xgboost"]          * xgb_prob  +
            MODEL_WEIGHTS["isolation_forest"]  * iso_prob  +
            MODEL_WEIGHTS["lstm"]              * lstm_prob +
            MODEL_WEIGHTS["autoencoder"]       * ae_prob
        )

        print(f"[Ensemble] XGB={xgb_prob:.4f} ISO={iso_prob:.4f} LSTM={lstm_prob:.4f} AE={ae_prob:.4f} → score={score:.4f}")

        return {
            "xgboost_prob":     round(xgb_prob,  4),
            "isolation_forest": round(iso_prob,   4),
            "lstm_prob":        round(lstm_prob,  4),
            "autoencoder_flag": round(ae_prob,    4),
            "ensemble_score":   round(score,      4),
            "is_fraud":         score >= 0.5,
            "confidence":       round(score * 100, 2),
        }