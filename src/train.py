import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# Пути
PROCESSED_DIR = "data/processed/"
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# Загрузка данных
X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").values.ravel()

# Модель
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Метрики
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

# Сохраняем модель
joblib.dump(model, f"{MODEL_DIR}/model.joblib")
print("✅ Модель сохранена в models/model.joblib")