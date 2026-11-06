FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY ml_backend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Código do serviço vive em src/dados/ml_backend (contexto de build = raiz do repo)
COPY src/dados/ml_backend/model.py src/dados/ml_backend/app.py /app/

EXPOSE 9090

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9090"]
