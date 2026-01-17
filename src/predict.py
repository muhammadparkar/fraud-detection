import joblib
import pandas as pd
import os

MODEL_PATH = "../models/fraud_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train the model first.")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
    return model

def predict(transaction_data):
    """
    transaction_data: list or array of 30 values (V1...V28, Time, Amount)
    """
    model = load_model()

    columns = [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
        "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
        "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24",
        "V25", "V26", "V27", "V28", "Amount"
    ]

    df = pd.DataFrame([transaction_data], columns=columns)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    if prediction == 1:
        print(f"⚠️ Fraud Transaction Detected (Probability: {probability:.4f})")
    else:
        print(f"✅ Legit Transaction (Probability of fraud: {probability:.4f})")

    return prediction, probability


if __name__ == "__main__":
    # Example test transaction (take one row from your dataset without Class)
    example_transaction = [
        0, -1.359807, -0.072781, 2.536347, 1.378155,
        -0.338321, 0.462388, 0.239599, 0.098698,
        0.363787, 0.090794, -0.551600, -0.617801,
        -0.991390, -0.311169, 1.468177, -0.470401,
        0.207971, 0.025791, 0.403993, 0.251412,
        -0.018307, 0.277838, -0.110474, 0.066928,
        0.128539, -0.189115, 0.133558, -0.021053,
        149.62
    ]

    predict(example_transaction)
