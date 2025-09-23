import os
import pandas as pd
from preprocess import preprocess_data

def test_preprocess_outputs():
    X_train, X_test, y_train, y_test = preprocess_data()

    # Проверим размеры
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # Проверим, что нужные колонки на месте
    assert 'Amount' in X_train.columns
    assert 'Time' in X_train.columns

    # Проверим, что сохранённые файлы существуют
    assert os.path.exists("data/processed/X_train.csv")
    assert os.path.exists("data/processed/y_train.csv")