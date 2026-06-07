FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all Python files
COPY *.py ./

# weights/ is NOT copied here — downloaded at startup from Google Drive

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]