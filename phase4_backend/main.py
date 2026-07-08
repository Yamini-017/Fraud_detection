from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from typing import List
import numpy as np
import uuid, joblib
from datetime import datetime

from phase4_backend.models   import TransactionInput, PredictionResult, Token, UserCreate
from phase4_backend.database import transactions_col, alerts_col, users_col
from phase6_security.auth    import (
    hash_password, verify_password, create_access_token, get_current_user
)
from phase3_ensemble.ensemble import FraudEnsemble
from phase5_blockchain.blockchain import store_on_blockchain

app = FastAPI(title="Fraud Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models + scalers ONCE at startup ─────────────────────────────────────
ensemble      = FraudEnsemble()
scaler_amount = joblib.load("data/processed/scaler_amount.pkl")
scaler_time   = joblib.load("data/processed/scaler_time.pkl")

# ── WebSocket Manager ─────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        disconnected = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

# ── Auth ───────────────────────────────────────────────────────────────────────
@app.post("/auth/register", status_code=201)
async def register(user: UserCreate):
    existing = await users_col.find_one({"username": user.username})
    if existing:
        raise HTTPException(400, "Username already registered")
    await users_col.insert_one({
        "username": user.username,
        "password": hash_password(user.password)
    })
    return {"message": "User created"}

@app.post("/auth/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = await users_col.find_one({"username": form.username})
    if not user or not verify_password(form.password, user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    token = create_access_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}

# ── Predict ────────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResult)
async def predict(tx: TransactionInput, current_user: str = Depends(get_current_user)):
    # Use separate scalers — fixes the scaler bug
    amount_scaled = float(scaler_amount.transform([[tx.Amount]])[0][0])
    time_scaled   = float(scaler_time.transform([[tx.Time]])[0][0])

    features = np.array([[
        tx.V1,  tx.V2,  tx.V3,  tx.V4,  tx.V5,  tx.V6,  tx.V7,
        tx.V8,  tx.V9,  tx.V10, tx.V11, tx.V12, tx.V13, tx.V14,
        tx.V15, tx.V16, tx.V17, tx.V18, tx.V19, tx.V20, tx.V21,
        tx.V22, tx.V23, tx.V24, tx.V25, tx.V26, tx.V27, tx.V28,
        amount_scaled, time_scaled
    ]])

    result = ensemble.predict(features)
    tx_id  = str(uuid.uuid4())

    # Blockchain
    tx_hash = await store_on_blockchain(tx_id, result["is_fraud"], result["ensemble_score"])

    doc = {
        "transaction_id":     tx_id,
        "is_fraud":           result["is_fraud"],
        "confidence":         result["confidence"],
        "ensemble_score":     result["ensemble_score"],
        "xgboost_prob":       result["xgboost_prob"],
        "lstm_prob":          result["lstm_prob"],
        "isolation_forest":   result["isolation_forest"],
        "autoencoder_flag":   result["autoencoder_flag"],
        "blockchain_tx_hash": tx_hash,
        "timestamp":          datetime.utcnow(),
        "status":             "BLOCKED" if result["is_fraud"] else "APPROVED",
        "user":               current_user,
    }
    await transactions_col.insert_one(doc)

    if result["is_fraud"]:
        await alerts_col.insert_one({
            "transaction_id": tx_id,
            "confidence":     result["confidence"],
            "ensemble_score": result["ensemble_score"],
            "xgboost_prob":   result["xgboost_prob"],
            "lstm_prob":      result["lstm_prob"],
            "isolation_forest": result["isolation_forest"],
            "autoencoder_flag": result["autoencoder_flag"],
            "blockchain_tx_hash": tx_hash,
            "timestamp":      datetime.utcnow(),
        })

    # Broadcast full result to WebSocket
    await manager.broadcast({
        "transaction_id":   tx_id,
        "is_fraud":         result["is_fraud"],
        "confidence":       result["confidence"],
        "ensemble_score":   result["ensemble_score"],
        "xgboost_prob":     result["xgboost_prob"],
        "lstm_prob":        result["lstm_prob"],
        "isolation_forest": result["isolation_forest"],
        "autoencoder_flag": result["autoencoder_flag"],
        "status":           "BLOCKED" if result["is_fraud"] else "APPROVED",
        "blockchain_tx_hash": tx_hash,
        "timestamp":        datetime.utcnow().isoformat(),
    })

    return PredictionResult(**{k: v for k, v in doc.items() if k != "_id"})

# ── Query endpoints ────────────────────────────────────────────────────────────
@app.get("/transactions")
async def get_transactions(limit: int = 50, current_user: str = Depends(get_current_user)):
    cursor = transactions_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=limit)

@app.get("/alerts")
async def get_alerts(limit: int = 20, current_user: str = Depends(get_current_user)):
    cursor = alerts_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=limit)

@app.get("/stats")
async def get_stats(current_user: str = Depends(get_current_user)):
    total  = await transactions_col.count_documents({})
    frauds = await transactions_col.count_documents({"is_fraud": True})
    return {
        "total_transactions": total,
        "fraud_count":        frauds,
        "safe_count":         total - frauds,
        "fraud_rate":         round(frauds / total * 100, 2) if total > 0 else 0,
    }