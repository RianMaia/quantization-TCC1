"""O instrumento de medição, validado contra respostas calculáveis à mão.

Todo resultado do artigo é uma *diferença* contra o A0. Se o próprio A0 estiver
errado — parsing de qrels, limiar trocado, consulta vazando — toda diferença é
computada contra uma referência quebrada, e nenhuma checagem de coerência
interna revelaria isso.
"""

from __future__ import annotations

import math

import ir_measures
import pytest
from ir_measures import R, nDCG

from quantptbr import corpus, evaluation, stats


def test_ndcg_matches_a_hand_computation():
    """Ganho linear, desconto log2(i+1). O caso todo cabe numa linha de conta."""
    qrels = {"q": {"d1": 3, "d2": 2, "d3": 1, "d4": 0}}
    run = {"q": {"d3": 0.9, "d1": 0.8, "d2": 0.7}}

    dcg = 1 / math.log2(2) + 3 / math.log2(3) + 2 / math.log2(4)
    ideal = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)

    measured = ir_measures.calc_aggregate([nDCG @ 10], qrels, run)[nDCG @ 10]
    assert measured == pytest.approx(dcg / ideal, abs=1e-9)


def test_gain_is_linear_and_not_exponential():
    """A convenção pré-registrada, travada contra uma troca silenciosa de biblioteca.

    Com ganho linear o grau 3 vale 3× o grau 1; com o exponencial `2^rel − 1`
    valeria 7×, e o mesmo run daria outro nDCG.
    """
    qrels = {"q": {"d1": 3, "d2": 1}}
    run = {"q": {"d2": 0.9, "d1": 0.8}}

    linear = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    exponential = ((2**1 - 1) + (2**3 - 1) / math.log2(3)) / (
        (2**3 - 1) + (2**1 - 1) / math.log2(3)
    )

    measured = ir_measures.calc_aggregate([nDCG @ 10], qrels, run)[nDCG @ 10]
    assert measured == pytest.approx(linear, abs=1e-9)
    assert measured != pytest.approx(exponential, abs=1e-3)


def test_recall_threshold_excludes_grade_one():
    """O corte ≥2 é a decisão de S1.2, e ele muda o denominador do recall."""
    qrels = {"q": {"d1": 3, "d2": 2, "d3": 1, "d4": 1}}
    run = {"q": {"d1": 0.9, "d2": 0.8}}
    assert ir_measures.calc_aggregate([R(rel=2) @ 100], qrels, run)[R(rel=2) @ 100] == 1.0
    assert ir_measures.calc_aggregate([R(rel=1) @ 100], qrels, run)[R(rel=1) @ 100] == 0.5


def test_library_scores_an_unscoreable_query_as_zero_and_keeps_it():
    """Comportamento determinado por teste, não presumido.

    Este é o fato que justifica a exclusão explícita da consulta 2: o
    `ir_measures` **não** omite uma consulta sem relevantes — ela entra na média
    como 0,0 e derruba o nDCG absoluto, que é exatamente a grandeza contra a qual
    o A0 é calibrado.
    """
    qrels = {"q1": {"d1": 3}, "q2": {"d1": 0, "d2": 0}}
    run = {"q1": {"d1": 0.9}, "q2": {"d1": 0.9, "d2": 0.8}}
    assert ir_measures.calc_aggregate([nDCG @ 10], qrels, run)[nDCG @ 10] == pytest.approx(0.5)


def test_measures_come_from_the_preregistered_matrix():
    threshold = stats.matrix()["evaluation"]["relevance_threshold_for_recall"]
    assert [str(m) for m in evaluation.measures()] == ["nDCG@10", f"R(rel={threshold})@100"]


def test_scoreable_set_is_the_preregistered_forty_nine():
    scoreable = evaluation.scoreable_query_ids()
    assert len(scoreable) == stats.matrix()["evaluation"]["scoreable_queries"]
    assert set(corpus.KNOWN_UNSCOREABLE_QUERY_IDS).isdisjoint(scoreable)


def test_evaluate_refuses_a_run_missing_a_judged_query():
    run = {q: {"p": 1.0} for q in evaluation.scoreable_query_ids()[:-1]}
    with pytest.raises(ValueError, match="sem resultados"):
        evaluation.evaluate(run)


def test_evaluate_refuses_a_run_with_an_unjudged_query():
    run = {q: {"p": 1.0} for q in evaluation.scoreable_query_ids()}
    run["999999"] = {"p": 1.0}
    with pytest.raises(ValueError, match="sem gabarito"):
        evaluation.evaluate(run)


def test_evaluate_filters_the_excluded_query_instead_of_refusing_it():
    """O harness recupera as 50; a avaliação escolhe quais entram na média."""
    run = {q: {"p": 1.0} for q in corpus.judged_query_ids()}
    assert evaluation.evaluate(run)["queries"] == len(evaluation.scoreable_query_ids())
    assert evaluation.evaluate(run, include_excluded=True)["queries"] == len(
        corpus.judged_query_ids()
    )


def test_evaluate_returns_per_query_values_for_the_bootstrap():
    """Média agregada não permite parear nada; o bootstrap consome estes valores."""
    run = {q: {"p": 1.0} for q in evaluation.scoreable_query_ids()}
    result = evaluation.evaluate(run)
    assert result["queries"] == len(evaluation.scoreable_query_ids())
    for measure in ("nDCG@10", "R(rel=2)@100"):
        assert set(result["per_query"][measure]) == set(evaluation.scoreable_query_ids())
