import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium", app_title="Maintenance prédictive")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import joblib
    import marimo as mo
    import matplotlib.pyplot as plt

    from src.config import MODELS_DIR
    from src.data import load_data
    from src.explain import explain_row, global_importance

    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    C_SAIN, C_PANNE = "#2a78d6", "#eb6834"
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )

    pipe = joblib.load(MODELS_DIR / "pipeline.joblib")
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    df = load_data()
    return C_PANNE, C_SAIN, MUTED, df, explain_row, global_importance, metrics, mo, pipe, plt


@app.cell
def _(df):
    import pandas as _pd

    from src.config import MODELS_DIR as _MODELS_DIR

    X = df.drop(columns=["machine_failure"])
    # Risque affiché = prédictions out-of-fold (hors-échantillon), pas un score sur les données vues.
    _oof = _pd.read_csv(_MODELS_DIR / "oof_scores.csv", index_col="machine_id")
    scored = df.assign(risk=_oof["risk"].to_numpy())
    return X, scored


@app.cell
def _(mo):
    mo.md(
        """
        # 🔧 Maintenance prédictive — tableau de bord

        Prédiction du risque de panne de machines-outils (XGBoost), avec explicabilité SHAP,
        diagnostic des modes de panne (FMEA) et surveillance de la dérive des données.
        L'assistant IA est dans le panneau de gauche.
        """
    )
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "❓ À quoi sert ce tableau de bord ?": mo.md(
                "Il prédit le **risque de panne** de chaque machine (modèle XGBoost, dataset AI4I 2020) "
                "pour aider à **prioriser la maintenance**. Les décisions sont **explicables** (SHAP) "
                "et les données **surveillées** (dérive)."
            ),
            "📊 Comment lire les sections ?": mo.md(
                "- **KPIs** : taille du parc, taux de panne, performance du modèle (PR-AUC, ROC-AUC).\n"
                "- **Machines à risque** : liste triée selon le *seuil de risque* (curseur).\n"
                "- **Importance SHAP globale** : les variables qui pèsent le plus dans les prédictions.\n"
                "- **Inspecteur** : pourquoi *une* machine précise est à risque (SHAP local).\n"
                "- **Supervision (MLOps)** : simulation de dérive des données + réentraînement."
            ),
            "🤖 L'assistant (panneau de gauche)": mo.md(
                "Un agent IA répond sur l'application, les machines, les causes de panne et les actions "
                "correctives — en interrogeant directement le modèle et le graphe FMEA (aucune réponse inventée)."
            ),
        }
    )
    return


@app.cell
def _(mo):
    try:
        from src.agent.chat import build_chat_model

        _assistant = mo.ui.chat(
            build_chat_model(),
            prompts=[
                "À quoi sert cette application ?",
                "Quelles sont les 5 machines les plus à risque ?",
                "Diagnostique la machine la plus à risque et propose des actions.",
                "Simule une dérive à sévérité 1 : est-elle détectée ?",
            ],
        )
        _out = mo.vstack([mo.md("### 🤖 Assistant"), _assistant])
    except Exception as _e:  # pas de backend LLM : le dashboard reste utilisable
        _out = mo.md(
            "### 🤖 Assistant\n"
            "_Indisponible : configure un backend LLM (GROQ_API_KEY, ou LLM_BACKEND=ollama)._\n\n"
            f"`{_e}`"
        )
    mo.sidebar(_out, width="440px")
    return


@app.cell
def _(metrics, mo):
    threshold = mo.ui.slider(
        0.01,
        0.95,
        value=round(metrics["threshold"], 2),
        step=0.01,
        label=f"Seuil de risque (défaut = seuil optimal par coût {metrics['cost_fn']:.0f}:{metrics['cost_fp']:.0f})",
    )
    threshold
    return (threshold,)


@app.cell
def _(metrics, mo, scored, threshold):
    flagged = int((scored["risk"] >= threshold.value).sum())
    mo.hstack(
        [
            mo.stat(f"{len(scored):,}", label="Machines"),
            mo.stat(f"{scored['machine_failure'].mean():.1%}", label="Taux de panne réel"),
            mo.stat(f"{metrics['xgboost']['pr_auc']:.2f}", label="PR-AUC (modèle)"),
            mo.stat(f"{metrics['xgboost']['roc_auc']:.2f}", label="ROC-AUC"),
            mo.stat(f"{flagged:,}", label=f"Signalées (≥ {threshold.value:.2f})"),
        ],
        justify="start",
        gap=1.5,
    )
    return


@app.cell
def _(mo, scored, threshold):
    at_risk = (
        scored[scored["risk"] >= threshold.value]
        .sort_values("risk", ascending=False)
        .round({"risk": 3})
    )
    display = mo.vstack(
        [
            mo.md(f"### Machines à risque (≥ {threshold.value:.2f}) — {len(at_risk)} machine(s)"),
            mo.ui.table(at_risk, page_size=10),
        ]
    )
    display
    return (at_risk,)


@app.cell
def _(mo):
    mo.md("### Fiabilité des probabilités (calibration)")
    return


@app.cell
def _(C_SAIN, MUTED, plt, scored):
    from sklearn.calibration import calibration_curve

    _frac, _mean = calibration_curve(
        scored["machine_failure"].to_numpy(),
        scored["risk"].to_numpy(),
        n_bins=10,
        strategy="quantile",
    )
    _fig, _ax = plt.subplots(figsize=(4.5, 4))
    _ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1, label="parfaitement calibré")
    _ax.plot(_mean, _frac, "o-", color=C_SAIN, label="modèle")
    _ax.set_xlabel("Probabilité prédite")
    _ax.set_ylabel("Taux de panne observé")
    _ax.set_title("Courbe de fiabilité (probas calibrées, out-of-fold)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _ax
    return


@app.cell
def _(mo):
    mo.md("### Ce qui pèse dans les décisions du modèle (SHAP global)")
    return


@app.cell
def _(C_SAIN, X, global_importance, pipe, plt):
    imp = global_importance(pipe, X.sample(500, random_state=0))
    _names = list(imp.keys())[::-1]
    _vals = list(imp.values())[::-1]
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.barh(_names, _vals, color=C_SAIN)
    _ax.set_title("Importance globale (moyenne des |SHAP|)")
    _ax.grid(axis="y", visible=False)
    _fig.tight_layout()
    _ax
    return


@app.cell
def _(at_risk, mo):
    _options = at_risk.head(30).index.astype(str).tolist()
    machine = mo.ui.dropdown(
        _options,
        value=_options[0] if _options else None,
        label="Inspecter une machine (top 30 à risque)",
    )
    machine
    return (machine,)


@app.cell
def _(C_PANNE, C_SAIN, MUTED, X, explain_row, machine, mo, pipe, plt, scored):
    if machine.value is None:
        out = mo.md("_Aucune machine au-dessus du seuil : baisse le seuil pour en inspecter une._")
    else:
        _idx = int(machine.value)
        _contrib = explain_row(pipe, X.loc[[_idx]])
        _items = sorted(_contrib.items(), key=lambda kv: abs(kv[1]))
        _names = [k for k, _ in _items]
        _vals = [v for _, v in _items]
        _colors = [C_PANNE if v > 0 else C_SAIN for v in _vals]
        _fig, _ax = plt.subplots(figsize=(6, 4))
        _ax.barh(_names, _vals, color=_colors)
        _ax.axvline(0, color=MUTED, linewidth=0.8)
        _ax.set_title(
            f"Machine {_idx} — risque affiché {scored.loc[_idx, 'risk']:.1%} "
            f"· lecture SHAP du modèle servi"
        )
        _ax.grid(axis="y", visible=False)
        _fig.tight_layout()
        out = _ax
    out
    return


@app.cell
def _():
    import pandas as pd

    from src.config import NUMERIC_FEATURES
    from src.drift import drift_report, simulate_drift

    return NUMERIC_FEATURES, drift_report, pd, simulate_drift


@app.cell
def _(mo):
    mo.md(
        """
        ## 🩺 Supervision (MLOps) — dérive des données & réentraînement

        Le batch de "production" est **simulé** (dérive covariable) pour démontrer la surveillance.
        """
    )
    return


@app.cell
def _(mo):
    severity = mo.ui.slider(0.0, 3.0, value=1.0, step=0.5, label="Sévérité de la dérive simulée")
    run_drift = mo.ui.run_button(label="Analyser la dérive")
    mo.hstack([severity, run_drift], justify="start", gap=1.5)
    return run_drift, severity


@app.cell
def _(NUMERIC_FEATURES, X, drift_report, mo, pd, run_drift, severity, simulate_drift):
    if not run_drift.value:
        _out = mo.md("_Règle la sévérité puis clique « Analyser la dérive »._")
    else:
        _report = drift_report(X, simulate_drift(X, severity.value), NUMERIC_FEATURES)
        _auc = _report["domain_classifier_auc"]
        _verdict = "🔴 forte" if _auc > 0.75 else "🟡 modérée" if _auc > 0.6 else "🟢 faible"
        _out = mo.vstack(
            [
                mo.md(f"**Dérive globale (AUC domain classifier) : {_auc:.2f} → {_verdict}**"),
                mo.ui.table(pd.DataFrame(_report["per_feature"]), page_size=5),
            ]
        )
    _out
    return


@app.cell
def _(C_PANNE, C_SAIN, X, plt, run_drift, severity, simulate_drift):
    if not run_drift.value:
        _out = None
    else:
        _current = simulate_drift(X, severity.value)
        _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.2))
        for _a, _f in zip(_axes, ["tool_wear_min", "torque_nm"], strict=False):
            _a.hist(X[_f], bins=40, alpha=0.6, color=C_SAIN, label="référence")
            _a.hist(_current[_f], bins=40, alpha=0.6, color=C_PANNE, label="production (dérivé)")
            _a.set_title(_f, fontsize=9)
            _a.grid(axis="x", visible=False)
            _a.legend(fontsize=7)
        _fig.suptitle("Distributions : référence vs batch dérivé", y=1.04)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell
def _(mo):
    run_retrain = mo.ui.run_button(label="Réentraîner le modèle (n_trials réduit)")
    mo.vstack([mo.md("### Réentraînement à la demande"), run_retrain])
    return (run_retrain,)


@app.cell
def _(mo, run_retrain):
    if not run_retrain.value:
        _out = mo.md(
            "_Clique pour relancer l'entraînement (≈ 15-30 s) et régénérer les artefacts._"
        )
    else:
        try:
            from src.model import run as run_training

            _xgb = run_training(n_trials=10)["xgboost"]
            _out = mo.md(
                f"✅ Réentraînement terminé — nouvelle **PR-AUC = {_xgb['pr_auc']:.3f}** "
                f"(ROC-AUC {_xgb['roc_auc']:.3f}).\n\n_Recharge l'app pour servir le nouveau modèle._"
            )
        except ModuleNotFoundError as _e:
            _out = mo.md(
                "⚠️ Réentraînement indisponible ici : les dépendances d'entraînement "
                "(`optuna`, `mlflow`) ne sont pas installées dans cet environnement. "
                "Installe-les avec `pip install -r requirements-dev.txt`.\n\n"
                f"`{_e}`"
            )
    _out
    return


if __name__ == "__main__":
    app.run()
