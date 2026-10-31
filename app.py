import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


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
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    })

    pipe = joblib.load(MODELS_DIR / "pipeline.joblib")
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    df = load_data()
    return C_PANNE, C_SAIN, MUTED, df, explain_row, global_importance, metrics, mo, pipe, plt


@app.cell
def _(df, pipe):
    X = df.drop(columns=["machine_failure"])
    scored = df.assign(risk=pipe.predict_proba(X)[:, 1])
    return X, scored


@app.cell
def _(mo):
    mo.md(
        """
        # 🔧 Maintenance prédictive — tableau de bord

        Risque de panne par machine (XGBoost), avec explicabilité SHAP.
        *L'agent IA viendra se greffer ici au prochain jalon.*
        """
    )
    return


@app.cell
def _(mo):
    try:
        from src.agent.chat import build_chat_model

        _assistant = mo.ui.chat(
            build_chat_model(),
            prompts=[
                "Quelles sont les performances du modèle ?",
                "Quelles sont les 5 machines les plus à risque ?",
                "Pourquoi la machine 0 est-elle à risque ?",
            ],
        )
        _out = mo.vstack([mo.md("### 🤖 Assistant maintenance"), _assistant])
    except Exception as _e:  # pas de backend LLM : le dashboard reste utilisable
        _out = mo.md(
            "### 🤖 Assistant maintenance\n"
            "_Indisponible : configure un backend LLM (GROQ_API_KEY, ou LLM_BACKEND=ollama)._\n\n"
            f"`{_e}`"
        )
    _out
    return


@app.cell
def _(mo):
    threshold = mo.ui.slider(0.05, 0.95, value=0.5, step=0.05, label="Seuil de risque")
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
        _ax.set_title(f"Machine {_idx} — risque {scored.loc[_idx, 'risk']:.1%}")
        _ax.grid(axis="y", visible=False)
        _fig.tight_layout()
        out = _ax
    out
    return


if __name__ == "__main__":
    app.run()
