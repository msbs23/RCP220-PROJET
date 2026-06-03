# Image de base légère avec Python
FROM python:3.10-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Copie et installation des dépendances d'abord (cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY api_radio/ ./api_radio/

# Variables d'environnement (surchargeables au runtime)
ENV MODEL_PATH=""
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_EXPERIMENT_NAME=""

# Port exposé
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "api_radio.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]