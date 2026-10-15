"""A trilha exaustiva: cada regra do documento de arquitetura contra uma conta conhecida.

Nada aqui precisa do corpus nem do artefato de vetores. O que está sob teste é a
aritmética das representações e a ordenação — o que, errado, produziria uma
tabela plausível com a condição errada dentro.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantptbr import exhaustive


def unit_vectors(rows: int, dims: int, seed: int = 0) -> np.ndarray:
    raw = np.random.default_rng(seed).standard_normal((rows, dims)).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def test_int8_reconstruction_stays_within_half_a_step_per_dimension():
    vectors = unit_vectors(500, 32)
    vectors[:, 5] = 0.0  # dimensão toda nula: a escala tem de virar 1, não 0

    representation = exhaustive.encode(vectors, "I8")
    scales = representation["scales"]
    rebuilt = representation["codes"] * scales

    assert scales[5] == 1.0
    assert representation["codes"].dtype == np.int8
    assert np.abs(representation["codes"]).max() <= 127
    assert (np.abs(rebuilt - vectors) <= scales / 2 + 1e-7).all()


@pytest.mark.parametrize("dims", [64, 70, 768])
def test_binary_score_is_the_negative_hamming_distance(dims: int):
    """Inclui uma dimensão que não é múltipla de 64: o preenchimento não pode contar."""
    vectors = unit_vectors(200, dims)
    query = unit_vectors(1, dims, seed=1)[0]

    measured = exhaustive.scores(exhaustive.encode(vectors, "B1"), query)
    expected = -((vectors > 0) != (query > 0)).sum(axis=1)

    assert np.array_equal(measured, expected)


def test_ties_at_the_cut_resolve_to_the_lowest_row():
    values = np.array([1, 5, 5, 5, 0, 5], dtype=np.int32)
    assert exhaustive.top_k(values, 2).tolist() == [1, 2]
    assert exhaustive.top_k(values, 5).tolist() == [1, 2, 3, 5, 0]


def test_float32_search_matches_a_direct_full_ranking():
    vectors, query = unit_vectors(3000, 48), unit_vectors(1, 48, seed=2)[0]
    direct = vectors @ query

    rows, values = exhaustive.search(exhaustive.encode(vectors, "F32"), query, k=10)

    assert rows.tolist() == np.argsort(-direct, kind="stable")[:10].tolist()
    assert np.allclose(values, direct[rows])


def test_lighter_conditions_keep_less_of_the_float32_top_ten():
    """Sanidade de ordem: F16 quase não mexe no ranking, I8 mexe pouco, B1 mexe mais."""
    vectors, queries = unit_vectors(5000, 128), unit_vectors(20, 128, seed=3)

    def top(kind: str) -> list[set[int]]:
        representation = exhaustive.encode(vectors, kind)
        return [set(exhaustive.search(representation, q, k=10)[0].tolist()) for q in queries]

    reference = top("F32")
    kept = {
        kind: np.mean([len(a & b) / 10 for a, b in zip(reference, top(kind), strict=True)])
        for kind in ("F16", "I8", "B1")
    }
    assert kept["F16"] >= 0.99
    assert kept["F16"] >= kept["I8"] > kept["B1"]


def test_sizes_follow_the_bits_per_component():
    vectors = unit_vectors(1000, 1024)
    sizes = {
        kind: exhaustive.size_bytes(exhaustive.encode(vectors, kind))
        for kind in exhaustive.CONDITIONS
    }
    assert sizes["F32"] == 1000 * 1024 * 4
    assert sizes["F16"] == sizes["F32"] // 2
    assert sizes["I8"] == 1000 * 1024 + 1024 * 4  # códigos mais as escalas float32
    assert sizes["B1"] == 1000 * 128  # 1.024 bits em 128 bytes, como o documento fixa


def test_metrics_follow_the_document_protocol():
    """Ganho 2^rel − 1, relevante = grau acima de zero, e ninguém sai do denominador."""
    qrels = {"1": {"a": 3, "b": 1, "c": 0, "d": 2}, "2": {"a": 0}}
    run = {"1": {"c": 4.0, "a": 3.0, "x": 2.0, "b": 1.0}, "2": {"a": 1.0}}

    report = exhaustive.evaluate(run, qrels)

    dcg = 7 / math.log2(3) + 1 / math.log2(5)
    ideal = 7 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert report["per_query"]["nDCG@10"]["1"] == pytest.approx(dcg / ideal)
    assert report["per_query"]["Recall@100"]["1"] == pytest.approx(2 / 3)
    assert report["per_query"]["MRR@100"]["1"] == pytest.approx(0.5)
    assert report["unscoreable"] == ["2"]
    assert report["aggregate"]["nDCG@10"] == pytest.approx(dcg / ideal / 2)


def test_measure_discards_warmup_and_shares_the_order_across_conditions():
    vectors, queries = unit_vectors(400, 32), unit_vectors(6, 32, seed=4)
    report = exhaustive.measure(
        exhaustive.encode(vectors, "B1"), queries, warmup=1, rounds=2, seed=7
    )
    assert report["samples"] == 2 * len(queries)
    assert report["latency_ms_p50"] <= report["latency_ms_p95"]
    assert report["qps_sequential"] > 0
