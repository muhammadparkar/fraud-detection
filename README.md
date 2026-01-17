# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning application that detects fraudulent credit card transactions using Python, Random Forest, and SMOTE.
The system is deployed with a FastAPI backend and a Streamlit frontend, supporting batch fraud detection via CSV upload, dynamic column mapping, and downloadable fraud analysis reports.

This project is designed to be production-ready and closely reflects how real-world fraud detection systems are built and deployed.

---

## 🚀 Features

* Data preprocessing and feature scaling
* Handling extreme class imbalance using **SMOTE**
* Fraud classification using **Random Forest**
* Model evaluation with:

  * Confusion Matrix
  * ROC Curve
  * AUC Score
* REST API built using **FastAPI**
* Interactive frontend using **Streamlit**
* Batch fraud detection using CSV upload
* Dynamic column mapping (works with any CSV format)
* Downloadable prediction reports

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* FastAPI
* Streamlit
* Matplotlib, Seaborn
* Joblib

---

## 📂 Project Structure

fraud-detection/
│
├── app/                  # FastAPI backend
│   ├── main.py
│   └── schema.py
│
├── frontend/             # Streamlit frontend
│   └── streamlit_app.py
│
├── src/                  # ML pipeline
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/               # Trained model (not committed)
├── data/                 # Dataset (not committed)
├── requirements.txt
├── README.md
└── .gitignore

---

## ⚙️ Setup Instructions

### 1. Clone the repository

git clone [https://github.com/your-username/fraud-detection.git](https://github.com/your-username/fraud-detection.git)
cd fraud-detection

### 2. Create and activate virtual environment

python -m venv venv
source venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

---

## 📊 Dataset

Download the dataset from Kaggle:

Credit Card Fraud Detection Dataset
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place the file inside:

data/creditcard.csv

⚠️ The dataset is not included in this repository due to size and licensing restrictions.

---

## 🧠 Train the Model

cd src
python train.py

This will:

* Preprocess the dataset
* Apply SMOTE to balance classes
* Train the Random Forest model
* Save the trained model to `models/fraud_model.pkl`

---

## 📈 Evaluate the Model

python evaluate.py

Generates:

* Confusion Matrix
* ROC Curve
* AUC Score

These metrics are crucial because accuracy alone is misleading for highly imbalanced fraud datasets.

---

## 🧪 Run the FastAPI Backend

From the project root:

uvicorn app.main:app --reload

Open in browser:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

You’ll get an interactive Swagger UI to test the API.

---

## 🌐 Run the Streamlit Frontend

Open a new terminal:

cd frontend
streamlit run streamlit_app.py

Open in browser:

[http://localhost:8501](http://localhost:8501)

---

## 📥 How the System Works

1. User uploads a CSV file containing transaction data
2. The app displays detected columns
3. User maps CSV columns to the required model features:
   Time, V1, V2, ..., V28, Amount
4. The system reorders and validates the data
5. Predictions are generated for all transactions
6. Output includes:

   * Fraud / Legit label
   * Fraud probability
7. User downloads a processed CSV report

This design allows the system to work with **any CSV format**, making it suitable for real-world datasets.

---

## 📌 Why Column Mapping?

Different users will upload CSVs with:

* Different column names
* Different column order
* Extra or missing columns

Column mapping ensures:

* Model input consistency
* Accurate predictions
* Production-grade reliability

This is how professional ML platforms handle data ingestion.

---

## 📜 License

This project is intended for educational and research purposes only.

---

