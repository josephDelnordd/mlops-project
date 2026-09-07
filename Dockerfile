# Utilisez une image de base légère
FROM python:3.12-slim

# Définissez les variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Installez les outils nécessaires (ping)
RUN apt-get update && \
	apt-get install -y iputils-ping && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

# Définissez le répertoire de travail
WORKDIR /app

# Copiez les fichiers requis
COPY requirements.txt .
COPY pyproject.toml .

# Installez les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiez le reste du code
COPY . .

# Créez les répertoires nécessaires
RUN mkdir -p models artifacts

# Définissez la commande par défaut
CMD ["python", "-m", "src.train", "--config", "configs/logreg.yaml"]
