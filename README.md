# Fraud Detection MLOps Project

This project presents an end-to-end machine learning pipeline designed to detect fraudulent credit card transactions.  
It integrates data preprocessing, model training, evaluation, versioning, and deployment into a cohesive and reproducible workflow.

Key features include:

- Real-time fraud prediction served through a FastAPI web API  
- Experiment tracking and model versioning with MLflow  
- Dockerized architecture for consistent environment setup  
- Deployment to a live cloud environment using Render

The goal is to provide a robust and maintainable framework that bridges the gap between machine learning development and production-ready delivery.

## Project Overview

Credit card fraud is a growing issue with significant financial implications for businesses and consumers.  
This project tackles the problem by building a supervised machine learning model capable of identifying fraudulent transactions in real time.

The workflow follows key MLOps principles to ensure scalability, reproducibility, and maintainability.  
From raw data ingestion and feature engineering to model training and containerized deployment, the pipeline is built with production-readiness in mind.

The application is deployed as a web service where users can upload transaction data and receive predictions instantly.

## Technologies Used

- **Python 3.12** — main programming language
- **Scikit-learn** — baseline machine learning models
- **XGBoost** — powerful gradient boosting model
- **Pandas** — data manipulation and preprocessing
- **MLflow** — experiment tracking and model versioning
- **FastAPI** — building RESTful API for real-time inference
- **Uvicorn** — ASGI server to run FastAPI app
- **Docker** — containerization for consistent deployment
- **Render** — cloud platform used for hosting the API

## Project Structure

├── data/
│   ├── raw/                  # Raw input dataset (e.g., creditcard.csv)
│   ├── processed/            # Processed datasets (train/test splits)
│   └── sample_transaction.csv  # Sample transactions for testing the API
│
├── models/                   # Trained and serialized ML models
│
├── mlruns/                   # MLflow tracking and model registry artifacts
│
├── notebooks/                # Jupyter notebooks for EDA and experimentation
│
├── src/                      # Source code for model training and inference
│   ├── main.py               # FastAPI app entry point
│   ├── predict.py            # Prediction logic
│   ├── preprocess.py         # Data preprocessing
│   ├── train.py              # Standalone training script
│   ├── train_rf.py           # Random Forest training
│   ├── train_xgb.py          # XGBoost training
│   ├── train_pipeline.py     # Full training pipeline (with MLflow)
│   └── tune_xgb.py           # Hyperparameter tuning for XGBoost
│
├── tests/                    # Unit tests
│   └── test_preprocess.py    # Preprocessing test
│
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker setup for the API
├── .gitignore                # Files to ignore in git
└── README.md                 # Project documentation

## Setup & Installation

Follow these steps to set up and run the project locally:

### 1. Clone the repository
```bash
git clone https://github.com/rewpak/fraud-detection-mlops.git
cd fraud-detection-mlops
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI app locally
```bash
uvicorn src.main:app --reload
```

After running this command, navigate to:
	•	Local API root: http://127.0.0.1:8000
	•	Swagger UI docs: http://127.0.0.1:8000/docs

## Usage

### 1. Open the API in your browser

Visit the Swagger UI to test the API manually:

```bash
https://fraud-detection-mlops-o0vp.onrender.com/docs
```
There you can upload a CSV file under the /predict endpoint.

### 2. Using cURL from the terminal

You can also send a request using curl:

```bash
curl -X POST "https://fraud-detection-mlops-o0vp.onrender.com/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_transaction.csv"
```

### 3. Expected CSV format
Your file should contain the same columns as the training data used for the model. Example format:

```bash
Time,V1,V2,V3,...,V28,Amount
-1.359807,-0.072781,2.536347,...,0.133558,-0.021053,149.62
1.191857,0.266151,0.166480,...,-0.008983,0.014724,2.69
```



