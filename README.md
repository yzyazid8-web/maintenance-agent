# 🔧 Assistant de maintenance prédictive — agent IA embarqué dans un notebook marimo

> Un tableau de bord **réactif (marimo)** où un **agent IA conversationnel** répond en langage
> naturel sur un modèle de risque de panne : il interroge le modèle **en direct** (SHAP, what-if),
> diagnostique les causes via un **knowledge graph**, garde une **mémoire** des diagnostics et
> surveille la **dérive des données**.

**Stack :** Python · marimo · XGBoost · SHAP · Pydantic AI (Groq / Ollama) · chromadb · NetworkX · MLflow · Docker · GitHub Actions

<!-- Ajoute ici une capture d'écran ou un GIF de démo :
![Démo](assets/demo.gif)
-->

---

## 🎯 L'idée du projet

Le cœur de ce projet **n'est pas le modèle de prévision 'XGBoost' ** , mais il joue le rôle
de **« moteur »**. Le vrai sujet est de **transformer un modèle ML en produit réellement utilisable**,
en **greffant un agent IA directement dans le tableau de bord marimo**.

Au lieu d'un simple dashboard que l'utilisateur doit savoir lire, on peut **lui parler** :
*« Quelles machines sont à risque ? Pourquoi la machine 7593 ? Que faire ? Y a-t-il une dérive des
données ? »* — et l'agent répond en **appelant les vrais outils** (modèle, SHAP, graphe de pannes,
monitoring), sans rien inventer.

**C'est ce couplage agent ⇄ notebook ⇄ modèle qui est la contribution du projet.**

---

## 

Ce projet, de bout en bout :

- **IA agentique / LLM** : **agent à outils** (Pydantic AI), **GraphRAG** sur un knowledge
  graph de modes de panne, **mémoire** épisodique, **guardrails**, et **évaluation** (LLM-as-judge).
- **MLOps** : détection de **dérive** (PSI, Wasserstein, domain classifier), réentraînement,
  suivi **MLflow**, conteneurisation **Docker**, **CI** GitHub Actions (lint + tests + build).
- **ML & explicabilité** : XGBoost sur données déséquilibrées, **SHAP** (global + local).
- **Rigueur** : séparation train/serving, tests (**27** unitaires), code typé et linté.

**Résultats clés :** PR-AUC **0,87 ± 0,01** en validation croisée (0,90 sur le jeu de test,
**0,78** en découpage par ordre d'arrivée — plus réaliste) · ROC-AUC **0,98** ·
l'agent appelle le bon outil dans **100 %** des cas évalués.

---

## 🏭 Le problème 

En maintenance industrielle, on veut **anticiper les pannes** pour intervenir *avant* la casse.
Ce projet, à partir de mesures machine (couple, vitesse, usure outil, températures) :

1. **prédit** le risque de panne de chaque machine,
2. **explique** pourquoi (variables en cause, mode de panne probable),
3. **recommande** des actions correctives **ancrées** dans une base de connaissances,
4. **surveille** que les données restent conformes à l'entraînement (dérive).

---

## ✨ Fonctionnalités

| | |
|---|---|
| 🤖 **Assistant IA** | agent conversationnel dans un panneau latéral, répond sur l'app, les machines, les causes et les actions |
| 📊 **Dashboard** | KPIs, table des machines à risque (seuil réglable), importances SHAP globales, inspecteur SHAP par machine |
| 🕸️ **Diagnostic** | knowledge graph des modes de panne (FMEA) → causes et actions **traçables** |
| 🩺 **Supervision** | simulation de dérive + rapport (PSI / Wasserstein / AUC) + bouton de réentraînement |

---

## 🧠 L'agent IA — techniques mises en œuvre

- **Agentique (agent à outils / tool-calling)** : l'agent raisonne en plusieurs étapes et appelle des
  outils Python (`predict_risk`, `explain_prediction`, `whatif`, `diagnose_mode`,
  `recommend_actions`, `check_drift`, …) → il **calcule** les réponses, il ne les invente pas.
- **LLM hybride** : même code pour **Groq** (`openai/gpt-oss-120b`, démo) ou **Ollama** (local, hors-ligne),
  via un client OpenAI-compatible routé par variable d'environnement.
- **Knowledge Graph + GraphRAG** : ontologie FMEA (modes de panne → causes → signatures → actions)
  en NetworkX ; l'agent **diagnostique par seuils** puis lit les actions dans le graphe (anti-hallucination).
- **Mémoire épisodique** : les diagnostics sont stockés (chromadb + embeddings) et rappelés par similarité.
- **Guardrails** : refus déterministe des injections de prompt et des entrées invalides, avant tout appel LLM.
- **Évaluation** : *tool-call accuracy* (déterministe) + *groundedness* notée par un **LLM-as-judge**
  — avec un **modèle juge distinct de l'agent** (famille différente) pour éviter le biais d'auto-complaisance.

---

## 🏗️ Architecture

```mermaid
flowchart LR
    U([Utilisateur]) --> DASH
    subgraph APP["Tableau de bord marimo"]
      DASH[KPIs - table risques - SHAP - supervision]
      AG[Agent IA a outils]
    end
    AG -->|outils| MODEL[XGBoost + SHAP]
    AG -->|GraphRAG| KG[(Knowledge Graph FMEA)]
    AG -->|memoire| MEM[(chromadb)]
    AG -->|LLM| LLM[Groq gpt-oss-120b / Ollama]
    MODEL --> DATA[(AI4I 2020)]
```

---

## 📊 Résultats

Modèle entraîné sur **AI4I 2020** : 10 000 **points de fonctionnement** (relevés de machines-outils,
pas 10 000 machines distinctes), **3,4 %** de pannes — fortement déséquilibré.

| Modèle | PR-AUC | ROC-AUC |
|---|---|---|
| Régression logistique (baseline) | 0,43 | 0,94 |
| **XGBoost** (Optuna + validation stratifiée) | **0,90** | **0,98** |

> La **PR-AUC** (métrique adaptée au déséquilibre) révèle l'écart réel entre les deux modèles,
> là où la ROC-AUC seule le masquerait. Facteurs les plus influents (SHAP) :
> `rotational_speed`, `tool_wear`, et la feature physique dérivée `power` — cohérents avec la physique des pannes.

**Robustesse du chiffre (en toute transparence).** Le **0,90** est la PR-AUC sur *un* découpage
(jeu de test). En validation croisée 5-fold elle vaut **0,87 ± 0,01** (plage 0,86–0,90) ; et en
découpage par **ordre d'arrivée** des relevés — proxy d'un déploiement temporel — elle tombe à
**0,78**. C'est ce dernier chiffre, le plus pessimiste, qui préfigure le mieux la production : un
0,90 isolé serait trompeur.

### Et une simple baseline par règles « métier » ? (quand le ML sert vraiment)

Les modes de panne d'AI4I 2020 sont **générés par des seuils documentés**. Une baseline FMEA codée
à la main avec ces mêmes seuils (`src/agent/knowledge_graph.py`) les retrouve donc presque
parfaitement en **rappel** — HDF/PWF/OSF à ~1,00, et même TWF à 0,96 là où XGBoost le sous-détecte (0,33).

| Détection de panne | Précision | Rappel |
|---|---|---|
| Règles FMEA (seuils) | 0,33 | 0,97 |
| **XGBoost** (seuil par coût) | **0,61** | 0,88 |

**Ce que le ML apporte réellement ici** n'est donc pas un meilleur rappel, mais : une **précision
doublée** (deux fois moins de fausses alertes, à rappel comparable), une **probabilité calibrée et
classable** — indispensable au tableau de bord et à l'agent — et aucun seuil à maintenir par type de
machine. Les pannes **aléatoires** (RNF) restent imprévisibles pour les deux approches (rappel
~0,05–0,11) : c'est un plafond intrinsèque du jeu de données, pas un défaut du modèle.

*(Rapport d'évaluation de l'agent : `eval/report.md`, généré par `python -m src.agent.eval`.)*

---

## 🗂️ Structure du dépôt

```
app.py                     tableau de bord marimo (dashboard + agent en sidebar)
src/
  config.py                chemins, graine, schéma des colonnes
  data.py                  téléchargement (UCI) + nettoyage + split
  features.py              feature engineering (features physiques) + pipeline
  model.py                 entraînement baseline + XGBoost (Optuna, MLflow)
  explain.py               SHAP (global + local)
  drift.py                 détection de dérive (PSI, Wasserstein, domain classifier)
  agent/
    llm.py                 LLM hybride (Groq / Ollama)
    tools.py               outils de l'agent (fonctions pures)
    agent.py               agent à outils (Pydantic AI)
    knowledge_graph.py     graphe FMEA + GraphRAG
    memory.py              mémoire épisodique (chromadb)
    guardrails.py          garde-fous d'entrée
    chat.py                pont vers mo.ui.chat
    eval.py                évaluation (tool-accuracy + LLM-as-judge)
knowledge/failure_modes.yaml   ontologie des modes de panne
models/                    artefacts entraînés (pipeline + métriques)
tests/                     tests unitaires (pytest)
Dockerfile, .github/workflows/ci.yml   conteneurisation + CI
```

---

## ▶️ Lancer en local

```bash
# 1. Environnement
python -m venv .venv
.venv/Scripts/activate            # Windows ;  source .venv/bin/activate sur Linux/Mac
pip install -r requirements.txt   # (ou requirements-dev.txt pour entraîner / tester)

# 2. Backend LLM — au choix :
#    a) Groq (clé gratuite sur console.groq.com) : créer un fichier .env
#         LLM_BACKEND=groq
#         GROQ_API_KEY=gsk_...
#    b) Ollama local (hors-ligne) : installer Ollama + `ollama pull qwen2.5:3b`, puis .env
#         LLM_BACKEND=ollama

# 3. Lancer l'application
marimo run app.py
```

Les artefacts du modèle (`models/`) et le jeu de données sont fournis : l'app démarre sans
étape d'entraînement. Le dashboard reste utilisable même sans clé LLM (seul l'agent est alors inactif).

**Autres commandes :**
```bash
python -m src.model          # réentraîner (Optuna + MLflow)
python -m src.agent.eval     # évaluer l'agent -> eval/report.md
pytest -q                    # tests
```

---

## 🚀 Déploiement

L'application est **conteneurisée** (`Dockerfile`) et la **CI construit l'image** à chaque push
(preuve de reproductibilité). Elle n'est pas hébergée en continu (choix de gratuité), mais se lance
en une commande en local ou via Docker :

```bash
docker build -t maintenance-agent .
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... maintenance-agent
```

---

## ⚠️ Limites (en toute transparence)

- **AI4I 2020** est un dataset **synthétique** (réaliste mais généré) ; il inclut des pannes
  purement **aléatoires** (RNF) qui plafonnent la performance atteignable.
- La **dérive est simulée** (dérive covariable sur un batch fabriqué) pour démontrer la chaîne de
  surveillance ; en production réelle, on brancherait le flux de données entrant à la place.
- Le réentraînement se fait sur les mêmes données : il illustre la **capacité opérationnelle**,
  pas une amélioration de performance.

---

## 📚 Données

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — UCI Machine Learning Repository.
