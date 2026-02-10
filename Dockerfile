FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY eval ./eval
COPY data ./data
COPY config.example.yaml ./config.example.yaml
COPY README.md ./README.md

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
