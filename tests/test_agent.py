"""Tests des outils de l'agent"""

from src.agent.tools import (
    build_deps,
    dataset_summary,
    explain_prediction,
    feature_importance,
    get_metrics,
    predict_risk,
    top_risk_machines,
    whatif,
)


def test_tools():
    deps = build_deps()

    assert "pr_auc" in get_metrics(deps)
    assert len(feature_importance(deps, top_n=3)) == 3
    assert dataset_summary(deps)["n_machines"] == 10_000

    top = top_risk_machines(deps, n=5)
    assert len(top) == 5

    mid = top[0]["machine_id"]
    assert 0.0 <= predict_risk(deps, mid) <= 1.0
    assert len(explain_prediction(deps, mid, top_n=4)) == 4

    # what-if : tolère les noms SHAP (num__...) et n'applique que les colonnes connues
    wi = whatif(deps, mid, {"num__torque_nm": 20.0})
    assert {"risk_before", "risk_after", "applied"} <= set(wi)
    assert wi["applied"] == {"torque_nm": 20.0}


def test_whatif_before_equals_displayed_risk():
    deps = build_deps()
    for mid in deps.scored.sort_values("risk", ascending=False).index[:5]:
        mid = int(mid)
        assert whatif(deps, mid, {})["risk_before"] == predict_risk(deps, mid)
