"""O harness de recuperação: um caminho só, parametrizado pelo braço.

Se cada braço tivesse o seu script, eles divergiriam em coisas que ninguém
pretendeu, e essas diferenças seriam reportadas como efeito de quantização.
Estes testes travam o que precisa ser idêntico entre braços e o que precisa
diferir exatamente conforme a matriz pré-registrada.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from quantptbr import corpus, embedding, retrieval, stats

pytestmark = pytest.mark.skipif(
    not retrieval.QUERY_MANIFEST_PATH.exists(),
    reason="consultas não codificadas; rode scripts/04_embed_queries.py",
)


@pytest.fixture(scope="module")
def query_vectors():
    return retrieval.load_queries()


def test_one_vector_per_judged_query_in_frozen_order(query_vectors):
    """A ordem das linhas é a de `judged_query_ids`; não há tabela de tradução."""
    ids, array = query_vectors
    assert ids == corpus.judged_query_ids()
    assert array.shape == (len(ids), embedding.Envelope.load().dimensions)


def test_query_vectors_are_unit_norm(query_vectors):
    _, array = query_vectors
    assert np.abs(np.linalg.norm(array, axis=1) - 1).max() < 1e-5


def test_manifest_matches_the_array_and_the_frozen_envelope():
    manifest = json.loads(retrieval.QUERY_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["array"]["sha256"] == corpus.sha256(retrieval.QUERY_VECTORS_PATH)
    assert manifest["envelope"] == json.loads(
        embedding.ENVELOPE_PATH.read_text(encoding="utf-8")
    ), "consultas codificadas com outro envelope que o corpus"


def test_queries_and_passages_use_different_prefixes():
    """Prefixo trocado degradaria todos os braços por igual — invisível entre braços."""
    envelope = embedding.Envelope.load()
    manifest = json.loads(retrieval.QUERY_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["envelope"]["query_prefix"] == envelope.query_prefix
    assert envelope.query_prefix != envelope.passage_prefix


def test_search_params_come_from_the_matrix_and_differ_only_where_declared():
    baseline = retrieval.search_params("A0", "hnsw")
    assert baseline.exact is False
    assert baseline.quantization is None, "o braço float32 não tem parâmetro de quantização"
    assert baseline.hnsw_ef == stats.matrix()["held_constant"]["hnsw"]["ef_search"]

    a4 = retrieval.search_params("A4", "hnsw")
    a5 = retrieval.search_params("A5", "hnsw")
    assert a4.quantization.rescore is False
    assert a5.quantization.rescore is True
    assert a5.quantization.oversampling == 4.0
    assert a4.hnsw_ef == a5.hnsw_ef, "A4 e A5 só podem diferir no rescoring"


def test_exact_mode_sets_the_exhaustive_flag_for_every_arm():
    for arm in stats.matrix()["arms"]:
        assert retrieval.search_params(arm["id"], "exact").exact is True
        assert retrieval.search_params(arm["id"], "hnsw").exact is False


def test_unregistered_arm_or_mode_is_refused():
    with pytest.raises(ValueError, match="não pré-registrado"):
        retrieval.search_params("A0", "approximate")
    with pytest.raises(KeyError):
        retrieval.search_params("A9", "hnsw")


def test_run_encodes_rank_not_score():
    """O desempate já foi decidido; passar o score deixaria a biblioteca desfazê-lo.

    A quantização binária produz distâncias inteiras com muitos empates exatos.
    Se o run levasse o score bruto, o `ir_measures` reordenaria os empatados por
    conta própria e a ordem passaria a depender do braço.
    """
    rows = [
        retrieval.Row("1", "pa", 0.9, 1),
        retrieval.Row("1", "pb", 0.9, 2),
        retrieval.Row("1", "pc", 0.9, 3),
    ]
    run = retrieval.as_run(rows)
    assert run["1"] == {"pa": -1.0, "pb": -2.0, "pc": -3.0}
    assert sorted(run["1"], key=lambda p: -run["1"][p]) == ["pa", "pb", "pc"]


def test_run_round_trips_through_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval, "RUNS_DIR", tmp_path)
    rows = [retrieval.Row("1", "pa", 0.9, 1), retrieval.Row("2", "pb", 0.8, 1)]
    retrieval.save_run(rows, "A0", "exact")
    assert retrieval.load_run("A0", "exact") == rows


def test_diagnostic_overrides_are_off_by_default():
    """Toda tabela reportada usa a matriz; afrouxar é opt-in e rotulado."""
    default = retrieval.search_params("A1", "hnsw")
    assert default.hnsw_ef == stats.matrix()["held_constant"]["hnsw"]["ef_search"]
    assert default.quantization.rescore is True


def test_decomposition_overrides_reach_the_search_params():
    """M4.3 precisa de ef alto e rescoring desligado, ou a medição se apaga."""
    probe = retrieval.search_params("A1", "hnsw", ef=8192, rescore=False)
    assert probe.hnsw_ef == 8192
    assert probe.quantization.rescore is False


def test_disabling_rescore_also_drops_oversampling():
    """Sem rescoring não há o que reordenar; oversampling só ampliaria candidatos."""
    a5 = retrieval.search_params("A5", "hnsw")
    assert a5.quantization.oversampling == 4.0
    assert retrieval.search_params("A5", "hnsw", rescore=False).quantization.oversampling is None


def test_float32_arm_has_no_quantization_params_under_any_override():
    assert retrieval.search_params("A0", "hnsw", ef=8192, rescore=False).quantization is None
