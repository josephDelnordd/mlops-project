FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY pyproject.toml .

RUN pip install --no-cache-dir --upgrade \
	pip \
	setuptools>=78.1.1 \
	msgpack>=1.2.1

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

RUN mkdir -p \
	models \
	artifacts \
	mlflow

CMD ["python", "-m", "src.train", "--config", "configs/logreg.yaml"]