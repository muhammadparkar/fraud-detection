import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from data_loader import load_data

def preprocess_data():
    df = load_data("../data/creditcard.csv")

    # Plot class distribution
    plt.figure(figsize=(6,4))
    df["Class"].value_counts().plot(kind="bar")
    plt.title("Class Distribution (0 = Legit, 1 = Fraud)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Scale Amount feature
    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    print("Preprocessing completed")
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    return X, y

if __name__ == "__main__":
    X, y = preprocess_data()
