import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print("Dataset Loaded Successfully")
    print(df.head())
    print("\nClass distribution:")
    print(df["Class"].value_counts())
    return df

if __name__ == "__main__":
    df = load_data("../data/creditcard.csv")
