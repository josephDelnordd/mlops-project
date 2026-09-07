# Utilisez une image de base légère
FROM python:3.12-slim

# Définissez les variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Définissez le répertoire de travail
WORKDIR /app

# Installez les dépendances système
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
	git \
	&& rm -rf /var/lib/apt/lists/*

# Copiez les fichiers requis
COPY requirements.txt .
COPY pyproject.toml .

# Mettez à jour pip et installez les packages critiques
RUN pip install --no-cache-dir --upgrade \
	pip \
	setuptools>=78.1.1 \
	msgpack>=1.2.1

# Installez les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiez le reste du code
COPY . .

# Installez le projet en mode développeur
RUN pip install -e .

# Créez les répertoires nécessaires
RUN mkdir -p \
	models \
	artifacts \
	mlflow

# Commande par défaut
CMD ["python", "-m", "src.train", "--config", "configs/logreg.yaml"]
