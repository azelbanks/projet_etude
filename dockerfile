# Image Python officielle légère
FROM python:3.13-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# --- CORRECTIF : Installation du compilateur GCC ---
# Nécessaire pour wordcloud et autres librairies C
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie des dépendances et installation
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copie de tout le code source dans le conteneur
COPY . .

# Commande par défaut : lancer le script de collecte
CMD ["python", "src/collection/collect_bluesky.py"]