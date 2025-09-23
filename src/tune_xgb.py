import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
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

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Сетка параметров
param_grid = {
    'xgb__n_estimators': [50, 100],
    'xgb__max_depth': [4, 6, 8],
    'xgb__learning_rate': [0.05, 0.1],
    'xgb__subsample': [0.8],
    'xgb__colsample_bytree': [0.8]
}

# GridSearch
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Лучшие параметры и метрики
print("🔍 Best params:", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
y_proba = grid_search.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Сохраняем лучшую модель
joblib.dump(grid_search.best_estimator_, f"{MODEL_DIR}/pipeline_xgb_best.joblib")
print("✅ Лучшая модель сохранена в models/pipeline_xgb_best.joblib")