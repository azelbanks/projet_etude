# Image Python officielle légère
FROM python:3.13-slim

# Identite de la revision construite. Injectee par .github/workflows/release.yml
# et exposee par l'endpoint /version : c'est ce qui permet de verifier qu'un
# deploiement ou un rollback a bien pris effet.
ARG GIT_SHA=unknown
ARG BUILD_TIME=
ENV GIT_SHA=${GIT_SHA} \
    BUILD_TIME=${BUILD_TIME} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Dossier de travail dans le conteneur
WORKDIR /app

# --- CORRECTIF : Installation du compilateur GCC ---
# Nécessaire pour wordcloud et autres librairies C
# curl sert a la sonde de disponibilite (HEALTHCHECK).
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie des dépendances et installation
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copie de tout le code source dans le conteneur
COPY . .

# Creer un utilisateur non-root pour la securite
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# Sonde sur /ready et non /health : /health repond des que le processus vit,
# /ready seulement quand le modele est charge et qu'une prediction est possible.
# start-period couvre le chargement du modele au demarrage.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/ready || exit 1

# Commande par défaut : lancer le script de collecte
CMD ["python", "src/collection/collect_bluesky.py"]