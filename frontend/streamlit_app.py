import streamlit as st
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"

REQUIRED_COLUMNS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
    "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24",
    "V25", "V26", "V27", "V28", "Amount"
]

st.title("💳 Credit Card Fraud Detection System")
st.write("Upload your CSV and map your columns to the required model features.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    st.subheader("Map Your Columns")
    mapping = {}

    for col in REQUIRED_COLUMNS:
        mapping[col] = st.selectbox(
            f"Select column for {col}",
            options=["-- None --"] + list(df.columns)
        )

    if st.button("Run Fraud Detection"):
        # Validate mapping
        selected_cols = list(mapping.values())
        if "-- None --" in selected_cols:
            st.error("Please map all required columns.")
        elif len(set(selected_cols)) != len(selected_cols):
            st.error("Each CSV column can only be mapped once.")
        else:
            ordered_df = df[[mapping[col] for col in REQUIRED_COLUMNS]]
            ordered_df.columns = REQUIRED_COLUMNS

            st.success("Column mapping successful. Running predictions...")

            results = []
            for _, row in ordered_df.iterrows():
                response = requests.post(API_URL, json={"features": row.tolist()})
                res = response.json()
                results.append(res)

            result_df = pd.concat(
                [df, pd.DataFrame(results)],
                axis=1
            )

            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Results",
                csv,
                "fraud_results.csv",
                "text/csv"
            )
