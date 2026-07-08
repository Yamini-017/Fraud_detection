import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
import joblib, os

def train_isolation_forest():
    X_train = np.load("data/processed/X_train.npy")
    X_test  = np.load("data/processed/X_test.npy")
    y_test  = np.load("data/processed/y_test.npy")

    print("Training Isolation Forest...")
    # CORRECT — 0.17% fraud rate in dataset
    model = IsolationForest(n_estimators=200, contamination=0.0017, random_state=42, n_jobs=-1)
    model.fit(X_train)

    # IsolationForest returns -1 (anomaly) and 1 (normal); convert to 1/0
    raw_preds = model.predict(X_test)
    y_pred = np.where(raw_preds == -1, 1, 0)

    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}")

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, "saved_models/isolation_forest.pkl")
    print("Saved: saved_models/isolation_forest.pkl")

if __name__ == "__main__":
    train_isolation_forest()