FROM python:3.12-slim

WORKDIR /app

# Устанавливаем супер-быстрый uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Копируем зависимости и устанавливаем их
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# КОПИРУЕМ НАШ КОД И ДАННЫЕ (Это самое важное!)
COPY src/ ./src/
COPY static/ ./static/
COPY chroma_db/ ./chroma_db/

ENV PYTHONUNBUFFERED=1

# Открываем правильный порт 8080 (как просят в README)
EXPOSE 8080

# Запускаем именно main.py на порту 8080
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]