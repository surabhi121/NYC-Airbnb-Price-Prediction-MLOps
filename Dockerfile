FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies before code: editing main.py doesn't bust the pip layer cache.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY schemas.py main.py ./

# No model is baked in. main.py loads models:/AirbnbPriceModel@champion from
# MLFLOW_TRACKING_URI at startup, so promoting a new champion only needs a
# container restart. MLFLOW_TRACKING_URI must be supplied at `docker run`.
#
# Fail fast if MLflow is unreachable: mlflow's defaults (7 retries, backoff 2)
# hang startup for ~4 minutes (measured in Task 6). 3 retries x 10 s timeout
# still rides out a brief MLflow restart (~14 s of backoff) but then exits
# with a clear error so Docker/Compose can restart the container.
ENV MLFLOW_HTTP_REQUEST_MAX_RETRIES=3 \
    MLFLOW_HTTP_REQUEST_TIMEOUT=10

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
