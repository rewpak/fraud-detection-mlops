from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import os

app = FastAPI()

# Загрузка модели
MODEL_PATH = "models/pipeline_xgb_best.joblib"
model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "🚀 Fraud Detection API is up and running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Чтение CSV
    df = pd.read_csv(file.file)

    # Предсказание
    preds = model.predict(df)

    # Возвращаем результат
    return {"predictions": preds.tolist()}