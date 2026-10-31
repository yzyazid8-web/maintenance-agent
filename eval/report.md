# Évaluation de l'agent

- Précision d'appel d'outil : **100%** (9/9)
- Groundedness moyenne (1-5) : **3.33**

| Question | Outil attendu | Outil OK | Groundedness |
|---|---|:--:|:--:|
| Combien de machines dans le parc et quel est le taux de pann | `dataset_summary` | ✅ | 1/5 |
| Quelles sont les performances du modèle ? | `model_metrics` | ✅ | 5/5 |
| Quels facteurs influencent le plus les pannes ? | `feature_importance` | ✅ | 5/5 |
| Quelles sont les 5 machines les plus à risque ? | `top_risk_machines` | ✅ | 5/5 |
| Quel est le risque de panne de la machine 42 ? | `predict_risk` | ✅ | 5/5 |
| Pourquoi la machine 7593 est-elle à risque ? | `explain_prediction` | ✅ | 5/5 |
| Quel mode de panne physique pour la machine 7593 ? | `diagnose_mode` | ✅ | 1/5 |
| Quelles actions correctives pour la machine 7593 ? | `recommend_actions` | ✅ | 2/5 |
| Si on baisse le couple de la machine 7593 à 30, que devient  | `whatif` | ✅ | 1/5 |