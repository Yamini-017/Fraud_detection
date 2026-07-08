from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class TransactionInput(BaseModel):
    # V1-V28 are PCA features from the Credit Card dataset
    V1: float;  V2: float;  V3: float;  V4: float
    V5: float;  V6: float;  V7: float;  V8: float
    V9: float;  V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    Amount: float
    Time: float = 0.0

class PredictionResult(BaseModel):
    transaction_id: str
    is_fraud: bool
    confidence: float
    ensemble_score: float
    xgboost_prob: float
    lstm_prob: float
    isolation_forest: float
    autoencoder_flag: float
    blockchain_tx_hash: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = "BLOCKED" if False else "APPROVED"   # overridden in logic

class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str