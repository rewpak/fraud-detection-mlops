# 1. Используем официальный образ Python
FROM python:3.12-slim

# 2. Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# 3. Копируем файлы проекта в контейнер
COPY . .

# 4. Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# 5. Открываем порт (если нужно запускать FastAPI)
EXPOSE 8000

# 6. Команда запуска (по умолчанию FastAPI)
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]