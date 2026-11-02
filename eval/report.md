# Évaluation de l'agent

- Précision d'appel d'outil : **100%** (9/9)
- Groundedness moyenne (1-5) : **3.89**
- Juge (LLM-as-judge) : **qwen/qwen3.8-27b** — modèle distinct de l'agent (anti-biais).

| Question | Outil attendu | Outil OK | Groundedness |
|---|---|:--:|:--:|
| Combien de machines dans le parc et quel est le taux de pann | `dataset_summary` | ✅ | 5/5 |
| Quelles sont les performances du modèle ? | `model_metrics` | ✅ | 3/5 |
| Quels facteurs influencent le plus les pannes ? | `feature_importance` | ✅ | 4/5 |
| Quelles sont les 5 machines les plus à risque ? | `top_risk_machines` | ✅ | 2/5 |
| Quel est le risque de panne de la machine 42 ? | `predict_risk` | ✅ | 5/5 |
| Pourquoi la machine 4729 est-elle à risque ? | `explain_prediction` | ✅ | 2/5 |
| Quel mode de panne physique pour la machine 4729 ? | `diagnose_mode` | ✅ | 4/5 |
| Quelles actions correctives pour la machine 4729 ? | `recommend_actions` | ✅ | 5/5 |
| Si on baisse le couple de la machine 4729 à 30, que devient  | `whatif` | ✅ | 5/5 |