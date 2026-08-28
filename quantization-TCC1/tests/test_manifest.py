"""Manifesto de execução e a checagem que impede agregar o inagregável."""

from __future__ import annotations

import copy
import subprocess

import pytest

from quantptbr import corpus, embedding, manifest, server, stats


@pytest.fixture()
def captured() -> dict:
    return manifest.RunManifest.capture(
        arm_id="A0", search_mode="exact", allow_dirty=True
    ).as_dict()


def test_manifest_records_the_whole_controlled_surface(captured):
    assert set(captured["controlled"]) == set(manifest.CONTROLLED_FIELDS)
    assert captured["controlled"]["envelope"]["model_id"] == embedding.MODEL_ID
    assert captured["controlled"]["image_digest"] == server.DIGEST
    assert captured["controlled"]["judged_query_ids"] == corpus.judged_query_ids()
    assert captured["controlled"]["hnsw"] == stats.matrix()["held_constant"]["hnsw"]


def test_manifest_records_the_arm_configuration_not_just_its_name(captured):
    """O nome do braço não descreve nada; a configuração descreve."""
    assert captured["run"]["arm_config"]["collection"] == "quati_a0_float32"
    assert captured["run"]["arm_config"]["quantization"] is None


def test_manifest_records_whether_the_tree_was_dirty(captured):
    """Execução com árvore suja não é citável, e o manifesto tem de dizer isso.

    O que se afirma é a fidelidade do campo, não um valor fixo: fixar `True`
    quebrava o teste sempre que a árvore estivesse limpa, que é justamente o
    estado em que se quer rodar a suíte antes de entregar.
    """
    assert captured["run"]["git_dirty"] == manifest.git_is_dirty()
    assert len(captured["run"]["git_commit"]) == 40


def test_dirty_tree_is_refused_without_an_explicit_override():
    if not manifest.git_is_dirty():
        pytest.skip("árvore limpa; nada a recusar")
    with pytest.raises(RuntimeError, match="árvore git suja"):
        manifest.RunManifest.capture(arm_id="A0")


def test_identical_runs_may_be_aggregated(captured):
    manifest.require_same_envelope([captured, copy.deepcopy(captured)])


@pytest.mark.parametrize(
    "field",
    ["envelope", "corpus_sha256", "server_version", "image_digest", "hnsw", "judged_query_ids"],
)
def test_aggregation_aborts_on_any_divergent_controlled_field(captured, field):
    """Aborta, nunca avisa e continua.

    Somar nDCG de execuções que consumiram vetores diferentes, ou que rodaram
    contra outra versão do servidor, produz um número que não corresponde a
    experimento nenhum — e que parece perfeitamente normal na tabela.
    """
    other = copy.deepcopy(captured)
    other["controlled"][field] = "divergente"
    with pytest.raises(RuntimeError, match=field):
        manifest.require_same_envelope([captured, other])


def test_differing_arm_or_repetition_does_not_block_aggregation(captured):
    """Braço e repetição são o que *varia*; bloqueá-los impediria o estudo inteiro."""
    other = copy.deepcopy(captured)
    other["run"]["arm"] = "A4"
    other["run"]["repetition"] = 3
    manifest.require_same_envelope([captured, other])


def test_empty_comparison_is_refused():
    with pytest.raises(ValueError, match="nada a comparar"):
        manifest.require_same_envelope([])


def test_manifest_round_trips_to_disk(tmp_path, captured):
    import json

    path = manifest.RunManifest(
        run=captured["run"], controlled=captured["controlled"], host=captured["host"]
    ).save(tmp_path / "m.json")
    assert json.loads(path.read_text(encoding="utf-8"))["controlled"] == captured["controlled"]


def test_evidence_written_by_the_run_does_not_count_as_a_dirty_tree(tmp_path, monkeypatch):
    """Sem isto, toda execução bloqueia a si mesma.

    O run grava em `runs/` — que é versionado de propósito — e só depois pede o
    manifesto. Contar essa escrita como árvore suja faria o `require_clean_tree`
    recusar exatamente a execução que acabou de produzir a evidência.
    """
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    monkeypatch.setattr(manifest.corpus, "REPO_ROOT", tmp_path)

    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "eval_a0_exact.json").write_text("{}", encoding="utf-8")
    assert not manifest.git_is_dirty(), "evidência recém-escrita não é código sujo"
    manifest.require_clean_tree()

    (tmp_path / "quantptbr").mkdir()
    (tmp_path / "quantptbr" / "index.py").write_text("x = 1", encoding="utf-8")
    assert manifest.git_is_dirty()
    with pytest.raises(RuntimeError, match="árvore git suja"):
        manifest.require_clean_tree()
