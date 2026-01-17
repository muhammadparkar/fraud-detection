import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from preprocess import preprocess_data
import joblib
import os

def train_model():
    # Load and preprocess data
    X, y = preprocess_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Before SMOTE:")
    print(y_train.value_counts())

    # Apply SMOTE only on training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_resampled, y_train_resampled)

    print("\nModel training completed")

    # Save model
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/fraud_model.pkl")
    print("Model saved as models/fraud_model.pkl")

    # Quick evaluation
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model

if __name__ == "__main__":
    train_model()
