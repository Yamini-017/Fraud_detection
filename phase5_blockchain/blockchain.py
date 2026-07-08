"""
For development: connects to Ganache (local Ethereum testnet).
Install Ganache: https://trufflesuite.com/ganache/
Then deploy FraudLedger.sol using Remix IDE or Truffle.
Paste the deployed contract address in .env as CONTRACT_ADDRESS.
"""
from web3 import Web3
from dotenv import load_dotenv
import os, json

load_dotenv()

GANACHE_URL       = os.getenv("GANACHE_URL", "http://127.0.0.1:7545")
CONTRACT_ADDRESS  = os.getenv("CONTRACT_ADDRESS", "")  # from Remix after deploy
PRIVATE_KEY       = os.getenv("PRIVATE_KEY", "")       # from Ganache account

# Minimal ABI — only the functions we call
CONTRACT_ABI = json.loads('''[
  {
    "inputs": [
      {"internalType": "string",  "name": "_txId",            "type": "string"},
      {"internalType": "bool",    "name": "_isFraud",         "type": "bool"},
      {"internalType": "uint256", "name": "_confidenceScore", "type": "uint256"}
    ],
    "name": "recordTransaction",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [{"internalType": "string", "name": "_txId", "type": "string"}],
    "name": "getRecord",
    "outputs": [
      {"internalType": "bool",    "name": "isFraud",         "type": "bool"},
      {"internalType": "uint256", "name": "confidenceScore", "type": "uint256"},
      {"internalType": "uint256", "name": "timestamp",       "type": "uint256"}
    ],
    "stateMutability": "view",
    "type": "function"
  }
]''')

w3 = Web3(Web3.HTTPProvider(GANACHE_URL))

async def store_on_blockchain(tx_id: str, is_fraud: bool, confidence: float) -> str:
    """
    Stores fraud result on blockchain. Returns transaction hash.
    Falls back gracefully if blockchain is not configured.
    """
    if not CONTRACT_ADDRESS or not PRIVATE_KEY:
        print("[Blockchain] Not configured — skipping.")
        return "not_configured"

    try:
        contract    = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
        account     = w3.eth.account.from_key(PRIVATE_KEY)
        conf_int    = int(confidence * 100)   # 87.5% → 8750

        tx = contract.functions.recordTransaction(tx_id, is_fraud, conf_int).build_transaction({
            "from":     account.address,
            "nonce":    w3.eth.get_transaction_count(account.address),
            "gas":      200000,
            "gasPrice": w3.to_wei("20", "gwei"),
        })
        signed    = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash   = w3.eth.send_raw_transaction(signed.rawTransaction)
        receipt   = w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt.transactionHash.hex()

    except Exception as e:
        print(f"[Blockchain] Error: {e}")
        return "error"