import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
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

print("✅ До SMOTE:", dict(pd.Series(y_train).value_counts()))

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("✅ После SMOTE:", dict(pd.Series(y_train).value_counts()))

# Модель
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
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
joblib.dump(model, f"{MODEL_DIR}/model_rf.joblib")
print("✅ Модель сохранена в models/model_rf.joblib")