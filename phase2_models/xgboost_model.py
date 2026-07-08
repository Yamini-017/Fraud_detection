import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib, os

def train_xgboost():
    X_train = np.load("data/processed/X_train.npy")
    X_test  = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test  = np.load("data/processed/y_test.npy")

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, "saved_models/xgboost.pkl")
    print("Saved: saved_models/xgboost.pkl")

if __name__ == "__main__":
    train_xgboost()