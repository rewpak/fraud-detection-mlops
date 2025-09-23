import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

RAW_DATA_PATH = "data/raw/creditcard.csv"
PROCESSED_DIR = "data/processed/"

def preprocess_data(raw_path=RAW_DATA_PATH, save_dir=PROCESSED_DIR):
    os.makedirs(save_dir, exist_ok=True)

    # Загрузка данных
    df = pd.read_csv(raw_path)

    # Стандартизация признаков 'Amount' и 'Time'
    scaler = StandardScaler()
    df[['Amount', 'Time']] = scaler.fit_transform(df[['Amount', 'Time']])

    # Делим на X и y
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Трен/тест сплит
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Сохраняем
    X_train.to_csv(f"{save_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{save_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{save_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{save_dir}/y_test.csv", index=False)

    return X_train, X_test, y_train, y_test

# ✅ Обёртка для безопасного запуска
def main():
    preprocess_data()
    print("✅ Препроцессинг завершён. Файлы сохранены в data/processed/")

if __name__ == "__main__":
    main()