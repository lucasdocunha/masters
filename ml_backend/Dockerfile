FROM python:3.12-slim

WORKDIR /app

# Instalar dependências básicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py app.py /app/

EXPOSE 9090

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:9090/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9090"]
