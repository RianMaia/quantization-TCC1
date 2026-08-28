"""Integridade da matriz pré-registrada.

A matriz é a fonte única consumida pelos scripts de build. Estes testes existem
para que a tabela do artigo e as execuções não possam divergir em silêncio.
"""

from __future__ import annotations

from quantptbr import corpus, embedding, stats

#: RAM total da máquina, menos folga para o sistema e o processo do servidor.
RESIDENCY_BUDGET_MB = 18_000


def test_matrix_declares_every_arm_in_the_tech_stack():
    ids = [arm["id"] for arm in stats.matrix()["arms"]]
    assert ids == ["A0", "A1", "A2", "A3", "A4", "A5"]
    assert len(set(ids)) == len(ids)


def test_every_arm_fits_resident_one_at_a_time():
    """S1.2 proíbe pré-registrar comparação que a máquina não executa."""
    for arm in stats.matrix()["arms"]:
        assert arm["predicted_resident_mb"] < RESIDENCY_BUDGET_MB, arm["id"]


def test_baseline_prediction_follows_from_the_frozen_envelope():
    """A previsão do A0 não é um número digitado: sai das dimensões congeladas."""
    envelope = embedding.Envelope.load()
    expected_mb = corpus.EXPECTED_PASSAGES * envelope.dimensions * 4 / 1024**2
    a0 = next(a for a in stats.matrix()["arms"] if a["id"] == "A0")
    assert abs(a0["predicted_resident_mb"] - expected_mb) / expected_mb < 0.05


def test_every_arm_names_a_question_and_both_outcomes():
    """Braço sem desfecho que mude alguma conclusão é descartado, não mantido por simetria."""
    for arm in stats.matrix()["arms"]:
        for field in ("question", "confirms_english_expectation", "contradicts"):
            assert arm[field].strip(), f"{arm['id']} não declara {field}"


def test_a5_shares_a4s_collection():
    arms = {a["id"]: a for a in stats.matrix()["arms"]}
    assert arms["A5"]["collection"] == arms["A4"]["collection"]
    assert arms["A4"]["search_overrides"]["rescore"] is False
    assert arms["A5"]["search_overrides"]["rescore"] is True


def test_excluded_query_is_the_one_the_corpus_flags():
    evaluation = stats.matrix()["evaluation"]
    assert tuple(evaluation["excluded_query_ids"]) == corpus.KNOWN_UNSCOREABLE_QUERY_IDS
    assert evaluation["scoreable_queries"] == evaluation["judged_queries"] - 1


def test_relevance_threshold_is_on_the_declared_grade_scale():
    threshold = stats.matrix()["evaluation"]["relevance_threshold_for_recall"]
    assert threshold in corpus.RELEVANCE_GRADES
    assert threshold == 2, "o corte é semântico: a rubrica quebra entre não-responde e responde"


def test_calibration_compares_against_the_qrels_the_published_number_used():
    """O 0,3955 é da Tabela 6: run de 1M pontuado contra os qrels de 10M.

    Confrontá-lo contra o nosso nDCG sobre os qrels de 1M compararia grandezas
    diferentes — 97,78 contra 38,66 julgamentos por consulta — e foi exatamente
    o erro que S1.2 cometeu.
    """
    target = stats.matrix()["evaluation"]["calibration_target"]
    assert target["arm"] == "A0" and target["search_mode"] == "exact"
    assert target["qrels"] == corpus.CALIBRATION_QRELS
    assert target["qrels"] != "qrels", "os qrels do estudo não são os da linha publicada"
    assert "Tabela 6" in target["source"]


def test_range_check_uses_the_studys_own_qrels_and_brackets_a_dense_biencoder():
    band = stats.matrix()["evaluation"]["calibration_target"]["range_check"]
    assert band["floor"]["ndcg10"] < band["nearest_above"]["ndcg10"] < band["ceiling"]["ndcg10"]
    assert band["floor"]["system"] == "BM25"


def test_calibration_qrels_are_a_frozen_hashed_input():
    """Se eles não estivessem no manifesto, a calibração dependeria de um arquivo solto."""
    files = corpus.read_manifest()["files"]
    assert corpus.CALIBRATION_QRELS in files
    assert files[corpus.CALIBRATION_QRELS]["sha256"]
