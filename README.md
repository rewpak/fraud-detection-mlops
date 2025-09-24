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

```bash
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
```
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

### 3. Expected Input Format
Your file should contain the same columns as the training data used for the model. Example format:

```bash
Time,V1,V2,V3,...,V28,Amount
-1.359807,-0.072781,2.536347,...,0.133558,-0.021053,149.62
1.191857,0.266151,0.166480,...,-0.008983,0.014724,2.69
```

### 4. File Format

Ensure your CSV file matches the expected structure:
	•	Contains features used during training (e.g., Time, V1, V2, …, V28, Amount)
	•	Does not include the Class label
	•	Use the provided sample_transaction.csv as a template

## Model Tracking with MLflow

MLflow is used in this project to track experiments, log metrics, and version models during the training process. This helps ensure reproducibility and allows you to monitor the performance of different model configurations over time.

### What is Tracked

- Model parameters (e.g., number of estimators, learning rate)
- Evaluation metrics (e.g., accuracy, precision, recall, F1 score)
- Model artifacts (serialized model files)
- Training metadata and timestamps

### How to Launch MLflow Locally

To view and interact with experiment runs locally, use the following command:

```bash
mlflow ui
``` 

Then open your browser and navigate to:
```bash
http://127.0.0.1:5000
```
This will launch the MLflow tracking UI where you can explore past runs, compare metrics, and download model artifacts.

## Dockerization

Docker is used to containerize the application, ensuring consistency across different environments and simplifying the deployment process.

🧱 Building the Docker Image
To build the Docker image locally, run the following command from the root of the project:

```bash
docker build -t fraud-api .
```
This command creates a Docker image named fraud-api using the instructions in the Dockerfile.

🚀 Running the Container
Once the image is built, run the container using:
```bash
docker run -d -p 8000:8000 fraud-api
```
-d runs the container in detached mode
-p 8000:8000 maps your local port 8000 to the container’s port 8000

After that, the API will be accessible at:

```bash
http://localhost:8000
```
And the Swagger UI will be available at:
```bash
http://localhost:8000/docs
```

## 🚀 Deployment

This project is deployed on Render, a cloud platform that allows running web services with Docker support.

🔗 Live Demo

You can access the live API here:
fraud-detection-mlops.onrender.com

Use the /docs route to interact with the Swagger UI:
https://fraud-detection-mlops-o0vp.onrender.com/docs

🌍 How to Deploy on Render

✅ Before deploying, ensure your GitHub repo includes:
	•	Dockerfile
	•	requirements.txt
	•	src/main.py (FastAPI entry point)

Steps:
	1.	Create a free account at https://render.com

	2.	Go to Dashboard → New Web Service

	3.	Connect your GitHub repository

	4.	Fill in the settings:

	    •	Build Command: (leave empty if using Docker)
	    •	Start Command:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

        •	Environment: Docker
	    •	Port: 8000

	5.	Click Create Web Service
	6.	Wait for the build to complete and copy your public URL

🐳 Why Render?

Render makes it easy to deploy containerized applications without managing infrastructure. It supports automatic deploys from GitHub and provides HTTPS and CI/CD out of the box.






