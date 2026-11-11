import marimo

__generated_with = "0.25.2"
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
    from src.explain import explain_row, global_importance_folds

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
    fold_models = joblib.load(
        MODELS_DIR / "fold_models.joblib"
    )  # 1 modèle par fold (SHAP cohérent)
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    _rp = MODELS_DIR / "report_metrics.json"
    report_metrics = json.loads(_rp.read_text()) if _rp.exists() else {}
    df = load_data()
    return (
        C_PANNE,
        C_SAIN,
        MUTED,
        df,
        explain_row,
        fold_models,
        global_importance_folds,
        metrics,
        mo,
        pipe,
        plt,
        report_metrics,
    )


@app.cell
def _(df):
    import pandas as _pd

    from src.config import MODELS_DIR as _MODELS_DIR

    X = df.drop(columns=["machine_failure"])
    # Risque affiché = prédictions out-of-fold (hors-échantillon), pas un score sur les données vues.
    _oof = _pd.read_csv(_MODELS_DIR / "oof_scores.csv", index_col="machine_id")
    scored = df.assign(
        risk=_oof["risk"].to_numpy(),
        risk_raw=_oof["raw"].to_numpy(),
        fold=_oof["fold"].astype(int).to_numpy(),
    )
    return X, scored


@app.cell
def _(mo):
    # Thème léger : couleur d'accent sur les titres de section (clair/sombre).
    mo.Html(
        "<style>"
        "h1, h2, h3 { color: #2f86e0; }"
        "h2 { border-bottom: 2px solid rgba(47, 134, 224, 0.25); padding-bottom: 0.2rem; }"
        "</style>"
    )
    return


@app.cell
def _(mo):
    _title = mo.md(
        """
        # 🔧 Maintenance prédictive — tableau de bord

        Estimation du risque de panne de machines-outils (XGBoost), avec explicabilité SHAP,
        diagnostic des modes de panne (FMEA) et surveillance de la dérive des données.
        L'assistant IA est dans le panneau de gauche.
        """
    )
    _note = mo.callout(
        mo.md(
            "Chaque ligne est un **relevé** (point de fonctionnement), pas une machine distincte ; "
            "le modèle **classe un état courant** à partir des capteurs — il n'anticipe pas à un horizon donné."
        ),
        kind="info",
    )
    mo.vstack([_title, _note])
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "❓ À quoi sert ce tableau de bord ?": mo.md(
                "Il estime le **risque de panne** de chaque relevé (modèle XGBoost, dataset AI4I 2020) "
                "pour aider à **prioriser la maintenance**. Les décisions sont **explicables** (SHAP) "
                "et les données **surveillées** (dérive)."
            ),
            "📊 Comment lire les sections ?": mo.md(
                "- **KPIs** : nombre de relevés, taux de panne, performance du modèle (PR-AUC).\n"
                "- **Relevés à risque** : liste triée selon le *seuil de risque* (curseur).\n"
                "- **Importance SHAP globale** : les variables qui pèsent le plus dans les prédictions.\n"
                "- **Inspecteur** : pourquoi *un* relevé précis est à risque (SHAP local).\n"
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
                "À quoi sert cette application et comment la lire ?",
                "Que montre la courbe de calibration (fiabilité) ?",
                "Quelles sont les 5 machines les plus à risque ?",
                "Diagnostique la machine la plus à risque et propose des actions.",
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
def _(metrics, mo, report_metrics, scored, threshold):
    from sklearn.metrics import average_precision_score as _ap

    flagged = int((scored["risk"] >= threshold.value).sum())
    # PR-AUC mise en avant = celle des scores RÉELLEMENT SERVIS (OOF calibré, hors-échantillon).
    _served = _ap(scored["machine_failure"], scored["risk"])
    _mo = report_metrics.get("model", {})
    _cap = (
        f"test {_mo.get('pr_auc_test', metrics['xgboost']['pr_auc']):.2f}"
        f" · CV {_mo.get('pr_auc_cv_mean', '?')} · ordre d'arrivée {_mo.get('pr_auc_temporal', '?')}"
        if _mo
        else "hors-échantillon (OOF)"
    )
    mo.hstack(
        [
            mo.stat(f"{len(scored):,}", label="Relevés", bordered=True),
            mo.stat(
                f"{scored['machine_failure'].mean():.1%}", label="Taux de panne réel", bordered=True
            ),
            mo.stat(f"{_served:.2f}", label="PR-AUC (servie, OOF)", caption=_cap, bordered=True),
            mo.stat(f"{metrics['xgboost']['roc_auc']:.2f}", label="ROC-AUC (test)", bordered=True),
            mo.stat(
                f"{flagged:,}",
                label=f"Signalés (≥ {threshold.value:.2f})",
                caption="à prioriser",
                bordered=True,
            ),
        ],
        justify="start",
        gap=1.5,
    )
    return


@app.cell
def _(mo, scored, threshold):
    at_risk = (
        scored[scored["risk"] >= threshold.value]
        .sort_values(["risk", "risk_raw"], ascending=False)  # départage les ex-æquo
        .drop(columns=["risk_raw", "fold"])
        .round({"risk": 3})
    )
    display = mo.vstack(
        [
            mo.md(f"### Relevés à risque (≥ {threshold.value:.2f}) — {len(at_risk)} relevé(s)"),
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

    # Bins UNIFORMES : la zone de décision (risque > ~0,2) est évaluée, au lieu d'être écrasée
    # par les 95 % de relevés quasi nuls (ce que donnaient les bins par quantiles).
    _frac, _mean = calibration_curve(
        scored["machine_failure"].to_numpy(),
        scored["risk"].to_numpy(),
        n_bins=10,
        strategy="uniform",
    )
    _fig, _ax = plt.subplots(figsize=(4.5, 4))
    _ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1, label="parfaitement calibré")
    _ax.plot(_mean, _frac, "o-", color=C_SAIN, label="modèle")
    _ax.set_xlabel("Probabilité prédite")
    _ax.set_ylabel("Taux de panne observé")
    _ax.set_title("Courbe de fiabilité (scores OOF, calibrateur Platt global)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _ax
    return


@app.cell
def _(mo):
    mo.md("### Ce qui pèse dans les décisions du modèle (SHAP global)")
    return


@app.cell
def _(C_SAIN, X, fold_models, global_importance_folds, plt):
    imp = global_importance_folds(fold_models, X.sample(500, random_state=0))
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
def _(mo, scored):
    _max = int(scored.index.max())
    machine = mo.ui.number(
        start=0,
        stop=_max,
        step=1,
        value=int(scored["risk"].idxmax()),  # par défaut : le relevé le plus à risque
        label=f"Inspecter n'importe quel relevé (index 0–{_max})",
    )
    machine
    return (machine,)


@app.cell
def _(C_PANNE, C_SAIN, MUTED, X, explain_row, fold_models, machine, mo, plt, scored):
    if machine.value is None or int(machine.value) not in scored.index:
        out = mo.md("_Saisis un index de relevé valide pour l'inspecter._")
    else:
        _idx = int(machine.value)
        _model = fold_models[int(scored.loc[_idx, "fold"])]  # modèle qui n'a pas vu cette ligne
        _contrib = explain_row(_model, X.loc[[_idx]])
        _items = sorted(_contrib.items(), key=lambda kv: abs(kv[1]))
        _names = [k for k, _ in _items]
        _vals = [v for _, v in _items]
        _colors = [C_PANNE if v > 0 else C_SAIN for v in _vals]
        _fig, _ax = plt.subplots(figsize=(6, 4))
        _ax.barh(_names, _vals, color=_colors)
        _ax.axvline(0, color=MUTED, linewidth=0.8)
        _ax.set_title(
            f"Relevé {_idx} — risque {scored.loc[_idx, 'risk']:.1%} "
            f"· SHAP du modèle hors-échantillon"
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
        _verdict = _report["verdict"]  # seuils définis dans src/drift.py, pas dans l'UI
        _cat = (
            " · ".join(
                f"{c['feature']} {c['status']} (p={c['p_value']})" for c in _report["categorical"]
            )
            or "—"
        )
        _kind = "danger" if "🔴" in _verdict else "warn" if "🟡" in _verdict else "success"
        _out = mo.vstack(
            [
                mo.callout(
                    mo.md(f"**Dérive globale (AUC domain classifier) : {_auc:.2f} → {_verdict}**"),
                    kind=_kind,
                ),
                mo.ui.table(pd.DataFrame(_report["per_feature"]), page_size=5),
                mo.md(f"**Catégorielles (χ²)** : {_cat}"),
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

            _res = run_training(n_trials=10)
            _p = _res["promotion"]
            if _p["promoted"]:
                _out = mo.callout(
                    mo.md(
                        f"✅ Challenger **promu** — PR-AUC {_p['challenger_pr_auc']:.3f} "
                        f"(champion précédent : {_p['champion_pr_auc']}).\n\n"
                        "_Recharge l'app pour servir le nouveau modèle._"
                    ),
                    kind="success",
                )
            else:
                _out = mo.callout(
                    mo.md(
                        f"🛟 Challenger **conservé au banc** (PR-AUC {_p['challenger_pr_auc']:.3f} "
                        f"< champion {_p['champion_pr_auc']:.3f}) : le modèle en production est "
                        "**inchangé**. C'est le garde-fou champion/challenger."
                    ),
                    kind="warn",
                )
        except ModuleNotFoundError as _e:
            _out = mo.callout(
                mo.md(
                    "⚠️ Réentraînement indisponible ici : les dépendances d'entraînement "
                    "(`optuna`, `mlflow`) ne sont pas installées dans cet environnement. "
                    "Installe-les avec `pip install -r requirements-dev.txt`.\n\n"
                    f"`{_e}`"
                ),
                kind="danger",
            )
    _out
    return


if __name__ == "__main__":
    app.run()
