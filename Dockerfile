FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py train_models.py README.md ./
COPY src ./src
COPY scripts ./scripts
COPY service ./service

ENTRYPOINT ["python", "inference.py"]
