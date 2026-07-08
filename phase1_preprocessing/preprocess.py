import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib, os

DATA_PATH   = "data/creditcard.csv"
OUTPUT_PATH = "data/processed/"

def load_and_explore(path):
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    print(f"Missing values: {df.isnull().sum().sum()}")
    return df

def preprocess(df):
    df = df.drop_duplicates()
    # FIX: Two separate scalers — one per feature
    scaler_amount = StandardScaler()
    scaler_time   = StandardScaler()
    df['Amount_scaled'] = scaler_amount.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler_time.fit_transform(df[['Time']])
    df = df.drop(['Amount', 'Time'], axis=1)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    joblib.dump(scaler_amount, OUTPUT_PATH + "scaler_amount.pkl")
    joblib.dump(scaler_time,   OUTPUT_PATH + "scaler_time.pkl")
    print("Saved scaler_amount.pkl and scaler_time.pkl")
    return df

def split_and_balance(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Before SMOTE — Fraud in train: {y_train.sum()}")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After  SMOTE — Fraud in train: {y_train_bal.sum()}")
    return X_train_bal, X_test, y_train_bal, y_test

def save_splits(X_train, X_test, y_train, y_test):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    np.save(OUTPUT_PATH + "X_train.npy", X_train)
    np.save(OUTPUT_PATH + "X_test.npy",  X_test)
    np.save(OUTPUT_PATH + "y_train.npy", y_train)
    np.save(OUTPUT_PATH + "y_test.npy",  y_test)
    print("Saved all splits.")

if __name__ == "__main__":
    df = load_and_explore(DATA_PATH)
    df = preprocess(df)
    X_train, X_test, y_train, y_test = split_and_balance(df)
    save_splits(X_train, X_test, y_train, y_test)
    print("Phase 1 complete.")