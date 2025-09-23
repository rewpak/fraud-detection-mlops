import pandas as pd
import joblib
import argparse
import os

# Пути
MODEL_PATH = "models/pipeline_xgb_best.joblib"

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")
    return joblib.load(model_path)

def predict(model, input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл с входными данными не найден: {input_path}")
    
    # Загружаем транзакции
    X_new = pd.read_csv(input_path)
    
    # Предсказания
    preds = model.predict(X_new)
    proba = model.predict_proba(X_new)[:, 1]  # вероятность фрода

    # Печать результатов
    for i, (p, prob) in enumerate(zip(preds, proba)):
        label = "ФРОД" if p == 1 else "НЕ ФРОД"
        print(f"🧾 Транзакция #{i + 1}: {label} (вероятность фрода: {prob:.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скрипт для предсказания фрода")
    parser.add_argument("--input", type=str, required=True, help="Путь к CSV-файлу с транзакциями")
    args = parser.parse_args()

    model = load_model(MODEL_PATH)
    predict(model, args.input)