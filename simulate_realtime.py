import pandas as pd
import requests
import time
import random
import sys

API_BASE  = "http://localhost:8000"
USERNAME  = "admin"
PASSWORD  = "admin123"
CSV_PATH  = "data/creditcard.csv"
MIN_DELAY = 0.5
MAX_DELAY = 1.5

def get_token():
    print("[Auth] Logging in...")
    try:
        r = requests.post(
            f"{API_BASE}/auth/login",
            data={"username": USERNAME, "password": PASSWORD},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=5
        )
        if r.status_code == 200:
            print("[Auth] Login successful")
            return r.json()["access_token"]
        elif r.status_code == 401:
            print("[Auth] Login failed — wrong username or password")
            print("[Auth] Register first: python -c \"import requests; r=requests.post('http://localhost:8000/auth/register',json={'username':'admin','password':'admin123'},headers={'Content-Type':'application/json'}); print(r.json())\"")
            sys.exit(1)
        else:
            print(f"[Auth] Unexpected: {r.status_code} {r.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("[Auth] Cannot connect — is uvicorn running?")
        sys.exit(1)

def load_data():
    print(f"[Data] Loading {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH).sample(frac=1).reset_index(drop=True)
        print(f"[Data] {len(df)} rows | {df['Class'].sum()} fraud | {(df['Class']==0).sum()} safe")
        return df
    except FileNotFoundError:
        print(f"[Data] File not found: {CSV_PATH}")
        sys.exit(1)

def simulate(df, token):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    print(f"\n[Sim] Starting — delay {MIN_DELAY}–{MAX_DELAY}s | Ctrl+C to stop\n")
    print("-" * 70)

    fraud_count = safe_count = 0

    for i, row in df.iterrows():
        payload = {f"V{j}": float(row[f"V{j}"]) for j in range(1, 29)}
        payload["Amount"] = float(row["Amount"])
        payload["Time"]   = float(row["Time"])

        try:
            r = requests.post(f"{API_BASE}/predict", json=payload, headers=headers, timeout=15)

            if r.status_code == 200:
                res = r.json()
                if res["is_fraud"]:
                    fraud_count += 1
                    label = "🚨 FRAUD"
                else:
                    safe_count += 1
                    label = "✅ SAFE "

                print(
                    f"[{i+1:>6}] {label} | "
                    f"conf:{str(res['confidence'])+'%':<8} | "
                    f"XGB:{res['xgboost_prob']:<7} "
                    f"LSTM:{res['lstm_prob']:<7} "
                    f"ISO:{res['isolation_forest']:<5} "
                    f"AE:{res['autoencoder_flag']:<5} | "
                    f"fraud total:{fraud_count}"
                )

            elif r.status_code == 401:
                print(f"[{i+1}] Token expired — re-logging in...")
                token   = get_token()
                headers["Authorization"] = f"Bearer {token}"

            else:
                print(f"[{i+1}] Error {r.status_code}: {r.text[:80]}")

        except requests.exceptions.ConnectionError:
            print(f"[{i+1}] Backend disconnected — retrying in 3s...")
            time.sleep(3)
            continue
        except requests.exceptions.Timeout:
            print(f"[{i+1}] Timeout — skipping")
            continue
        except KeyboardInterrupt:
            print_summary(i+1, fraud_count, safe_count)
            sys.exit(0)

        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    print_summary(len(df), fraud_count, safe_count)

def print_summary(total, fraud, safe):
    print("\n" + "="*70)
    print(f"  Total: {total} | Fraud: {fraud} | Safe: {safe} | Rate: {round(fraud/total*100,2) if total else 0}%")
    print("="*70)

if __name__ == "__main__":
    token = get_token()
    df    = load_data()
    simulate(df, token)