"""Contrato do envelope de embedding congelado em S1.4.

O teste que carrega peso não é o round-trip: é o que amarra a **decisão**
gravada em `config/embedding_envelope.json` à **evidência** gravada em
`runs/s1.4_pilot.json`. Um envelope editado à mão, ou um piloto refeito sem
regravar o envelope, quebra aqui em vez de contaminar 1M de vetores em silêncio.
"""

from __future__ import annotations

import json

import pytest

from quantptbr import embedding

PILOT_PATH = embedding.REPO_ROOT / "runs" / "s1.4_pilot.json"

#: Fração do efeito do int8 que o fp16 pode ocupar sem ser confundidor.
#:
#: Um teto absoluto em cosseno seria arbitrário — 1e-4 é pouco ou muito? A
#: pergunta só tem resposta nas unidades do estudo. O critério é relativo: a
#: perturbação de ranking do fp16 tem de ficar uma ordem de grandeza abaixo da
#: do braço **mais suave** da matriz, a quantização escalar int8. Se chegar
#: perto dela, o fp16 confunde o próprio efeito medido e o passe roda em fp32.
MAX_FP16_SHARE_OF_INT8 = 0.1


@pytest.fixture(scope="module")
def pilot() -> dict:
    if not PILOT_PATH.exists():
        pytest.skip("piloto não executado; rode scripts/02_pilot_embeddings.py")
    return json.loads(PILOT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def envelope() -> embedding.Envelope:
    if not embedding.ENVELOPE_PATH.exists():
        pytest.skip("envelope não congelado; rode scripts/02_pilot_embeddings.py")
    return embedding.Envelope.load()


def test_envelope_round_trips_through_json(tmp_path):
    original = embedding.Envelope(
        model_id=embedding.MODEL_ID,
        model_revision=embedding.MODEL_REVISION,
        dimensions=embedding.DIMENSIONS,
        max_length=256,
        batch_size=32,
        dtype="float16",
        pooling="mean",
        normalize=True,
        query_prefix=embedding.QUERY_PREFIX,
        passage_prefix=embedding.PASSAGE_PREFIX,
    )
    path = tmp_path / "envelope.json"
    original.save(path)
    assert embedding.Envelope.load(path) == original


def test_model_revision_is_a_pinned_commit():
    """`main` se move; um sha de 40 hex não."""
    assert len(embedding.MODEL_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in embedding.MODEL_REVISION)


def test_envelope_matches_the_pilot_that_produced_it(envelope, pilot):
    assert pilot["envelope"] == json.loads(embedding.ENVELOPE_PATH.read_text(encoding="utf-8"))
    assert envelope.max_length == pilot["token_lengths"]["chosen_max_length"]
    assert envelope.batch_size in [
        row["batch_size"] for row in pilot["batch_sweep"] if not row["oom"]
    ]


def test_envelope_is_the_frozen_model_and_geometry(envelope):
    assert envelope.model_id == embedding.MODEL_ID
    assert envelope.model_revision == embedding.MODEL_REVISION
    assert envelope.dimensions == embedding.DIMENSIONS
    assert envelope.normalize is True, "a métrica de distância pressupõe vetores normalizados"


def test_max_length_covers_the_corpus_tail_up_to_the_model_limit(pilot):
    """Não é um número redondo: é a cauda real do corpus, limitada pelo modelo.

    A cauda vem das `longest_k` passagens mais longas de 1M, não da amostra
    aleatória — uma amostra de 10.000 não enxerga o extremo de um milhão. Onde a
    cauda ultrapassa o limite posicional do modelo, truncar é inevitável, e o que
    o teste exige é que a fração truncada esteja **registrada** em vez de
    silenciada.
    """
    lengths = pilot["token_lengths"]
    expected = min(lengths["corpus_token_max"], embedding.MODEL_MAX_LENGTH)
    assert lengths["chosen_max_length"] >= expected
    assert lengths["chosen_max_length"] <= embedding.MODEL_MAX_LENGTH
    if lengths["chosen_max_length"] < lengths["corpus_token_max"]:
        assert lengths["tail_fraction_truncated"] > 0, (
            "o corpus tem passagens acima do limite do modelo, mas nenhuma truncagem foi registrada"
        )


def test_envelope_carries_the_e5_prefixes(envelope):
    """O E5 degrada em silêncio sem os prefixos, então eles são parte do envelope."""
    assert envelope.query_prefix == embedding.QUERY_PREFIX
    assert envelope.passage_prefix == embedding.PASSAGE_PREFIX
    assert envelope.query_prefix != envelope.passage_prefix


def test_not_truncating_was_free(pilot):
    """A escolha de cobrir a cauda tem preço medido, e o preço é ~zero.

    Sob padding dinâmico, `max_length` trunca mas quase não encarece. Se um dia
    passar a encarecer, a decisão de S1.4 deixa de ser gratuita e precisa ser
    retomada em vez de herdada.
    """
    chosen = max(r["passages_per_second"] for r in pilot["batch_sweep"] if not r["oom"])
    reference = [r for r in pilot["batch_sweep_reference"] if not r["oom"]]
    if not reference:
        pytest.skip("referência coincide com o valor escolhido")
    assert chosen >= 0.95 * max(r["passages_per_second"] for r in reference)


def test_fp16_is_not_a_confound_for_the_effect_being_measured(envelope, pilot):
    """A decisão de precisão, julgada nas unidades do estudo: o ranking.

    Compara o que o fp16 faz ao top-10 das consultas julgadas com o que a
    quantização int8 faz ao mesmo top-10. Não é um limiar escolhido a priori —
    é uma razão entre o ruído da implementação e o sinal que o artigo mede.
    """
    if envelope.dtype != "float16":
        pytest.skip("envelope não usa fp16; a checagem de confundidor não se aplica")
    fp16, int8 = pilot["rank_agreement"]["fp16"], pilot["rank_agreement"]["simulated_int8"]
    assert int8["mean_overlap"] < 1.0, "a régua não mediu nada; o int8 simulado está inerte"

    # Duas leituras da mesma perturbação: quanto do top-10 troca (contínua) e em
    # quantas consultas o conjunto muda (discreta). Ambas como razão contra o
    # int8, porque nenhum valor absoluto das duas significa algo isolado.
    for metric in ("mean_overlap", "identical_set"):
        fp16_loss, int8_loss = 1 - fp16[metric], 1 - int8[metric]
        assert fp16_loss <= MAX_FP16_SHARE_OF_INT8 * int8_loss, (
            f"em {metric} o fp16 perturba {fp16_loss:.4f} contra {int8_loss:.4f} do int8, "
            f"acima da fração de {MAX_FP16_SHARE_OF_INT8:.0%}: rode em fp32"
        )
