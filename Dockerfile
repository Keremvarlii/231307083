FROM python:3.11-slim

WORKDIR /app

# OpenCV ve YOLO için gereken sistem kütüphaneleri
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD gunicorn -b 0.0.0.0:${PORT:-5000} app:app

