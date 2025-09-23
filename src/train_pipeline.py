import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

# ✅ Импортируем MLflow и модуль для логирования sklearn-моделей
import mlflow
import mlflow.sklearn

# 📁 Пути к директориям с данными и для сохранения модели
PROCESSED_DIR = "data/processed/"
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)  # Создаём директорию models/, если её нет

# 📥 Загружаем предобработанные обучающие и тестовые данные
X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").values.ravel()

# ⚙️ Создаём pipeline: масштабируем данные и обучаем XGBoost
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Стандартизация признаков
    ('xgb', XGBClassifier(         # Модель XGBoost с базовыми параметрами
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ))
])

# 🚀 Стартуем логирование эксперимента в MLflow
with mlflow.start_run(run_name="xgb_pipeline_with_registry"):

    # 🧠 Обучаем модель
    pipeline.fit(X_train, y_train)

    # 📊 Получаем предсказания
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 📈 Считаем метрики
    roc_auc = roc_auc_score(y_test, y_proba)
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc:.4f}")

    # ✅ Логируем метрику ROC-AUC
    mlflow.log_metric("roc_auc", roc_auc)

    # ✅ Логируем параметры модели (для отслеживания экспериментов)
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    })

    # ✅ Логируем и регистрируем модель в MLflow Registry
    mlflow.sklearn.log_model(
        sk_model=pipeline,                  # Объект pipeline
        artifact_path="model",              # В какой папке сохранять в MLflow
        registered_model_name="fraud_detector"  # 📌 Имя модели в MLflow Registry
    )

# 💾 Параллельно сохраняем модель локально в формате .joblib
joblib.dump(pipeline, f"{MODEL_DIR}/pipeline_xgb.joblib")
print("✅ Pipeline сохранён в models/pipeline_xgb.joblib")