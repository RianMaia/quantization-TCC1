"""Contrato do artefato de vetores de S1.5.

O teste que justifica o arquivo é o de retomada. Um passe de 1M interrompido e
recomeçado que pule ou repita um bloco produz um artefato indetectavelmente
errado: a forma bate, as normas batem, e todos os seis braços leem os mesmos
vetores deslocados. Nenhuma comparação entre braços revelaria isso.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from quantptbr import corpus, embedding, vectors


@pytest.mark.gpu
def test_kill_and_resume_is_byte_identical(tmp_path, monkeypatch):
    """Simula a queda na pior janela: bloco gravado, progresso ainda não.

    É o caso que mais importa porque a retomada precisa **reescrever** um bloco
    já presente no disco. Se a reescrita não for idêntica, o artefato passa a
    depender de onde o passe foi interrompido.
    """
    envelope = embedding.Envelope.load()
    monkeypatch.setattr(corpus, "EXPECTED_PASSAGES", 1500)
    monkeypatch.setattr(vectors, "CHUNK_SIZE", 500)
    monkeypatch.setattr(vectors, "VECTORS_DIR", tmp_path)

    monkeypatch.setattr(vectors, "ARRAY_PATH", tmp_path / "whole.npy")
    monkeypatch.setattr(vectors, "PROGRESS_PATH", tmp_path / "whole.json")
    vectors.run_pass(envelope)
    uninterrupted = np.load(tmp_path / "whole.npy")

    monkeypatch.setattr(vectors, "ARRAY_PATH", tmp_path / "resumed.npy")
    monkeypatch.setattr(vectors, "PROGRESS_PATH", tmp_path / "resumed.json")
    write_progress = vectors.write_progress
    crashed = {"yet": False}

    def crash_before_recording(rows_done: int) -> None:
        if not crashed["yet"]:
            crashed["yet"] = True
            raise KeyboardInterrupt("queda após gravar o bloco, antes de registrá-lo")
        write_progress(rows_done)

    monkeypatch.setattr(vectors, "write_progress", crash_before_recording)
    with pytest.raises(KeyboardInterrupt):
        vectors.run_pass(envelope)
    assert vectors.read_progress() == 0, "progresso não deveria ter sido registrado"

    vectors.run_pass(envelope)
    resumed = np.load(tmp_path / "resumed.npy")

    assert resumed.shape == uninterrupted.shape
    assert np.array_equal(resumed, uninterrupted), (
        "o passe retomado difere do ininterrupto: a retomada não é reprodutível"
    )


def test_normalise_produces_unit_vectors():
    """fp16 devolve normas fora de 1; é a renormalização que o índice exige."""
    rough = (np.random.default_rng(0).normal(size=(64, 768)) * 3).astype(np.float16)
    norms = np.linalg.norm(vectors.normalise(rough), axis=1)
    assert np.abs(norms - 1).max() < 1e-6


pytestmark_artifact = pytest.mark.skipif(
    not vectors.MANIFEST_PATH.exists(), reason="passe de 1M não executado"
)


@pytestmark_artifact
def test_manifest_matches_the_array_on_disk():
    manifest = json.loads(vectors.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["array"]["sha256"] == corpus.sha256(vectors.ARRAY_PATH)
    assert manifest["corpus_sha256"] == corpus.read_manifest()["files"]["corpus"]["sha256"]


@pytestmark_artifact
def test_array_has_one_row_per_passage_under_the_frozen_envelope():
    envelope = embedding.Envelope.load()
    array = vectors.load()
    assert array.shape == (corpus.EXPECTED_PASSAGES, envelope.dimensions)
    assert array.dtype == np.float32
