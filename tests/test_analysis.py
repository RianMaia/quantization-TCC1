"""Análise pareada: o que precisa estar certo para que o verbo do artigo seja crível.

O bootstrap em si já é testado em `test_stats.py` contra respostas sintéticas
conhecidas. Aqui o que está sob teste é o que vem antes e depois dele: parear as
consultas certas, recusar agregar execuções incomparáveis, marcar o que é zero
por construção, e dizer qual diferença o desenho não teria enxergado.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantptbr import analysis, stats

METRIC = "nDCG@10"
QUERY_IDS = [str(i) for i in range(1, 21)]


def evaluation(values: dict[str, float], controlled: dict | None = None) -> dict:
    return {
        "controlled": {"envelope": {"model_id": "x"}} if controlled is None else controlled,
        "evaluation": {"per_query": {METRIC: values}},
    }


def constant(value: float) -> dict:
    return evaluation(dict.fromkeys(QUERY_IDS, value))


def test_pairing_follows_the_query_id_not_the_dictionary_order():
    """Parear na ordem errada destruiria o poder de n=49 sem nada parecer estranho."""
    baseline = evaluation({q: float(i) for i, q in enumerate(QUERY_IDS)})
    shuffled = dict(reversed(list(baseline["evaluation"]["per_query"][METRIC].items())))
    variant = evaluation(shuffled)

    left, right = analysis.aligned(baseline, variant, METRIC)
    assert np.array_equal(left, right), "a mesma medição não pode virar diferença"


def test_divergent_query_sets_are_refused():
    baseline = constant(0.5)
    variant = evaluation({q: 0.5 for q in QUERY_IDS[:-1]})
    with pytest.raises(ValueError, match="divergem"):
        analysis.aligned(baseline, variant, METRIC)


def test_incomparable_runs_abort_instead_of_being_aggregated(monkeypatch):
    """Somar nDCG de execuções com envelopes diferentes dá um número de experimento nenhum."""
    arms = [arm["id"] for arm in stats.matrix()["arms"]]

    def fake_load(arm_id, _mode):
        other = {"envelope": {"model_id": "outro"}}
        return evaluation(dict.fromkeys(QUERY_IDS, 0.5), other if arm_id == "A2" else None)

    monkeypatch.setattr(analysis, "load", fake_load)
    assert len(arms) > 2
    with pytest.raises(RuntimeError, match="não podem ser agregadas"):
        analysis.load_all("hnsw")


def test_minimum_detectable_difference_grows_with_the_spread():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0.5, 0.2, size=200)
    tight = analysis.minimum_detectable_difference(baseline, baseline + 0.01)
    noisy = analysis.minimum_detectable_difference(
        baseline, baseline + rng.normal(0.01, 0.1, size=200)
    )
    assert tight < noisy
    assert tight == pytest.approx(0.0, abs=1e-9), "diferença constante não tem variância"


def test_minimum_detectable_difference_matches_the_closed_form():
    """2,80 · sd/√n para 95% e potência 0,80 — a conta padrão do desenho pareado."""
    rng = np.random.default_rng(1)
    baseline = rng.normal(0.5, 0.2, size=49)
    variant = baseline + rng.normal(0.0, 0.05, size=49)

    differences = variant - baseline
    expected = 2.8015952 * differences.std(ddof=1) / np.sqrt(differences.size)
    assert analysis.minimum_detectable_difference(baseline, variant) == pytest.approx(
        expected, rel=1e-5
    )


def test_identical_runs_are_flagged_as_such_and_never_called_equivalent(monkeypatch):
    """Zero exato em toda consulta é o modo de busca devolvendo o mesmo run.

    O Qdrant ignora quantização sob `exact`, então os seis braços coincidem ali.
    Reportar isso como 'não distinguível de ruído' sem a marca convidaria a
    leitura de que a quantização saiu de graça.
    """
    monkeypatch.setattr(analysis, "load", lambda _arm, _mode: constant(0.5))
    rows = analysis.compare("exact", METRIC)

    assert rows, "a matriz tem braços além do A0"
    for row in rows:
        assert row["identical_by_construction"] is True
        assert row["mean_difference"] == 0.0
        assert row["verdict"] == stats.INDISTINGUISHABLE
        assert "equival" not in row["sentence"].lower()


def test_a_real_difference_is_not_flagged_as_identical(monkeypatch):
    rng = np.random.default_rng(2)
    degraded = evaluation(dict(zip(QUERY_IDS, rng.normal(0.2, 0.05, 20), strict=True)))
    monkeypatch.setattr(
        analysis, "load", lambda arm, _mode: constant(0.5) if arm == "A0" else degraded
    )
    rows = analysis.compare("hnsw", METRIC)
    for row in rows:
        assert row["identical_by_construction"] is False
        assert row["verdict"] == stats.DEGRADES
        assert row["ci_high"] < 0


def test_every_comparison_carries_an_interval_and_a_detectable_difference(monkeypatch):
    monkeypatch.setattr(analysis, "load", lambda _arm, _mode: constant(0.5))
    for row in analysis.compare("hnsw", METRIC):
        assert {"ci_low", "ci_high", "minimum_detectable_difference", "n"} <= set(row)
        assert row["n"] == len(QUERY_IDS)
