# syntax=docker/dockerfile:1

# Python figé sur la version de développement (idéalement épinglé par @sha256 en production).
FROM python:3.12.2-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root (bonne pratique conteneur)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    LLM_BACKEND=groq \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# Dépendances figées (closure transitive) -> build reproductible.
COPY --chown=user requirements.lock .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.lock

# Embedder pré-téléchargé dans l'image -> conteneur AUTONOME (pas de download au 1er appel, marche hors-ligne).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY --chown=user . .

EXPOSE 7860
# Le dashboard doit répondre en HTTP (sans backend LLM, l'assistant s'affiche « indisponible »).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860', timeout=4).status==200 else 1)"
CMD ["marimo", "run", "app.py", "--host", "0.0.0.0", "-p", "7860"]
