"""Tests des wrappers de l'agent (validation), sans LLM."""

from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic_ai import ModelRetry

from src.agent.agent import _require_machine, unknown_overrides


def _deps(n=3):
    return SimpleNamespace(scored=pd.DataFrame({"risk": [0.1] * n}, index=range(n)))


def test_require_machine_accepts_valid_index():
    _require_machine(_deps(3), 1)  # ne lève pas


def test_require_machine_rejects_unknown_index():
    with pytest.raises(ModelRetry):
        _require_machine(_deps(3), 999)


def test_unknown_overrides_rejects_type_and_unknown():
    # 'type' (catégorielle) et une variable inconnue -> rejetées
    assert unknown_overrides({"type": "L"}) == ["type"]
    assert unknown_overrides({"foo": 1}) == ["foo"]
    # variables numériques (y compris préfixées SHAP) -> acceptées
    assert unknown_overrides({"torque_nm": 30}) == []
    assert unknown_overrides({"num__torque_nm": 30}) == []
