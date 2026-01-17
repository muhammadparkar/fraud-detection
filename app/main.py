from fastapi import FastAPI
import joblib
import pandas as pd
from app.schema import Transaction

app = FastAPI(title="Credit Card Fraud Detection API")

MODEL_PATH = "models/fraud_model.pkl"

# Load model once at startup
model = joblib.load(MODEL_PATH)

columns = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
    "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24",
    "V25", "V26", "V27", "V28", "Amount"
]

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    if len(transaction.features) != 30:
        return {"error": "Input must contain exactly 30 features"}

    df = pd.DataFrame([transaction.features], columns=columns)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": "Fraud" if prediction == 1 else "Legit",
        "fraud_probability": round(probability, 4)
    }
