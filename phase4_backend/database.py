from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db     = client["fraud_detection"]

transactions_col = db["transactions"]
alerts_col       = db["alerts"]
users_col        = db["users"]