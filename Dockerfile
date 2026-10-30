# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Image de l'app "maintenance-agent" (marimo), prête pour HF Spaces.
# HF Spaces impose : port 7860 + conteneur exécuté en user id 1000.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# libgomp1 : OpenMP, requis par XGBoost. Sans lui -> `import xgboost` plante.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces : créer et basculer sur l'utilisateur 1000 (sinon erreurs de permission).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    LLM_BACKEND=groq \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# Couche dépendances AVANT le code -> cache réutilisé tant que requirements.txt est inchangé.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Code applicatif + artefacts du modèle (models/). La clé API n'est PAS copiée :
# GROQ_API_KEY est injecté par HF comme secret runtime et lu via os.environ.get().
COPY --chown=user . .

EXPOSE 7860
CMD ["marimo", "run", "app.py", "--host", "0.0.0.0", "-p", "7860"]
