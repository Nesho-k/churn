FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY . .

# Copy model artifacts to /app/model so inference.py finds them at MODEL_DIR
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/model /app/model
COPY src/serving/model/3b1a41221fc44548aed629fa42b762e0/artifacts/feature_columns.txt /app/model/feature_columns.txt

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
